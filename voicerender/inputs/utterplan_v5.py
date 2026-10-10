"""Strict consumer reader for UtterPlan v5 flow TOML.

This is intentionally a small wire reader, not a copy of UtterPlan's compiler.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..errors import (
    PlanError,
    ProfileError,
    UnsupportedSynthesisFeatureError,
)
from ..models import RenderSegment, SourceRef, VoiceProfile

PLAN_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
FLOW_HASH_SCHEMA = "utterplan-flow-v1"
SEGMENT_HASH_SCHEMA = "voicerender-segment-semantics.v1"
ROOT_KEYS = {
    "format",
    "schema_version",
    "plan_id",
    "language",
    "unit",
    "hash_schema",
    "document",
    "linguistics",
    "flow",
    "producer",
    "warnings",
}
DIRECTIVE_KEYS = {
    "voice",
    "pronunciation",
    "prosody",
    "emphasis",
    "say_as",
    "substitution",
    "audio",
    "extensions",
}
SEGMENT_KEYS = {
    "text",
    "language",
    "pause_before",
    "pause_after",
    "marker",
    "heading",
    "token",
    *DIRECTIVE_KEYS,
}
PAUSE_CLASSES = {
    "none",
    "x-weak",
    "weak",
    "medium",
    "strong",
    "x-strong",
    "clause",
    "sentence",
    "paragraph",
    "parenthetical",
    "voice_change",
}
TOKEN_COLUMNS = {"span", "pos", "tag", "lemma", "morph"}


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PlanError(
            f"UtterPlan semantic value is not canonical JSON: {exc}"
        ) from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _nonempty(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{what} must be a non-empty string")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


_DIRECTIVE_SHAPES = {
    "voice": {"reference", "name", "languages", "gender", "age", "variant"},
    "pronunciation": {"phonemes", "alphabet"},
    "prosody": {"rate", "pitch", "volume"},
    "emphasis": {"level"},
    "say_as": {"interpret_as", "format", "detail"},
    "substitution": {"alias"},
    "audio": {
        "src",
        "alt_text",
        "clip_begin",
        "clip_end",
        "speed",
        "repeat_duration",
        "repeat_count",
        "sound_level",
        "description",
    },
}


def _directives(raw: Mapping[str, Any], where: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in DIRECTIVE_KEYS:
        if name not in raw:
            continue
        value = raw[name]
        if name == "extensions":
            if not isinstance(value, list):
                raise PlanError(f"{where}.extensions must be an array")
            extensions = []
            for index, item in enumerate(value):
                item_path = f"{where}.extensions[{index}]"
                if not isinstance(item, dict) or set(item).difference(
                    {"name", "params"}
                ):
                    raise PlanError(f"{item_path} has an invalid extension shape")
                extension_name = _nonempty(item.get("name"), f"{item_path}.name")
                params = item.get("params", {})
                if not isinstance(params, dict):
                    raise PlanError(f"{item_path}.params must be an object")
                extensions.append({"name": extension_name, "params": _plain(params)})
            if extensions:
                result[name] = extensions
            continue
        if not isinstance(value, dict):
            raise PlanError(f"{where}.{name} must be an object")
        unknown = set(value).difference(_DIRECTIVE_SHAPES[name])
        if unknown:
            raise PlanError(f"Unknown {where}.{name} fields: {sorted(unknown)}")
        if name == "voice":
            normalized = {
                key: _plain(item) for key, item in value.items() if item is not None
            }
        elif name == "pronunciation":
            normalized = {
                "phonemes": _nonempty(
                    value.get("phonemes"), f"{where}.pronunciation.phonemes"
                ),
                "alphabet": value.get("alphabet", "ipa"),
            }
        elif name == "prosody":
            normalized = {key: value.get(key) for key in ("rate", "pitch", "volume")}
        elif name == "emphasis":
            normalized = {
                "level": _nonempty(value.get("level"), f"{where}.emphasis.level")
            }
        elif name == "say_as":
            normalized = {
                "interpret_as": _nonempty(
                    value.get("interpret_as"), f"{where}.say_as.interpret_as"
                ),
                "format": value.get("format"),
                "detail": value.get("detail"),
            }
        elif name == "substitution":
            normalized = {
                "alias": _nonempty(value.get("alias"), f"{where}.substitution.alias")
            }
        else:  # AudioDirective's typed defaults are semantically retained as null.
            normalized = {
                "src": _nonempty(value.get("src"), f"{where}.audio.src"),
                "alt_text": value.get("alt_text"),
                "clip_begin": value.get("clip_begin"),
                "clip_end": value.get("clip_end"),
                "speed": value.get("speed"),
                "repeat_duration": value.get("repeat_duration"),
                "repeat_count": value.get("repeat_count"),
                "sound_level": value.get("sound_level"),
            }
            if value.get("description") is not None:
                normalized["description"] = value["description"]
        result[name] = normalized
    return result


def _pause(value: Any, what: str) -> Any:
    if isinstance(value, str):
        if value not in PAUSE_CLASSES:
            raise PlanError(f"Unknown {what} pause intent {value!r}")
        return value
    if isinstance(value, dict):
        if set(value) != {"type", "time"} or value.get("type") != "timed":
            raise PlanError(
                f"{what} pause must be a known class or exact timed-pause table"
            )
        time = _nonempty(value.get("time"), f"{what}.time")
        return {"type": "timed", "time": time}
    raise PlanError(f"{what} pause must be a string or timed-pause table")


def _column_values(raw: Any, *, field: str, count: int, path: str) -> dict[int, str]:
    if not isinstance(raw, list):
        raise PlanError(f"{path} must be an array")
    if field in {"lemma", "morph"}:
        sparse = True
    else:
        sparse = bool(raw and isinstance(raw[0], list))
    if not sparse:
        if len(raw) != count or any(not isinstance(value, str) for value in raw):
            raise PlanError(f"{path} dense values must match the token span count")
        return dict(enumerate(raw))

    result: dict[int, str] = {}
    previous = -1
    for index, pair in enumerate(raw):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or type(pair[0]) is not int
            or not 0 <= pair[0] < count
            or pair[0] <= previous
            or not isinstance(pair[1], str)
        ):
            raise PlanError(
                f"{path}[{index}] must be an ascending [token_index, string] pair"
            )
        result[pair[0]] = pair[1]
        previous = pair[0]
    return result


def _tokens(raw: Any, text: str, path: str) -> tuple[dict[str, Any], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise PlanError(f"{path} must be a token table")
    unknown = set(raw).difference(TOKEN_COLUMNS)
    if unknown:
        raise PlanError(f"Unknown {path} columns: {sorted(unknown)}")
    spans = raw.get("span")
    if not isinstance(spans, list) or not spans:
        raise PlanError(f"{path}.span is required when token annotations exist")
    normalized: list[tuple[int, int]] = []
    previous_end = 0
    for index, span in enumerate(spans):
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(value) is not int for value in span)
        ):
            raise PlanError(
                f"{path}.span[{index}] must be an integer [start, end] pair"
            )
        start, end = span
        if start < previous_end or end <= start or end > len(text):
            raise PlanError(
                f"{path}.span[{index}] is unordered or outside segment text"
            )
        normalized.append((start, end))
        previous_end = end

    columns: dict[str, dict[int, str]] = {}
    for name in ("pos", "tag", "lemma", "morph"):
        if name in raw:
            columns[name] = _column_values(
                raw[name], field=name, count=len(normalized), path=f"{path}.{name}"
            )
    output = []
    for index, (start, end) in enumerate(normalized):
        token: dict[str, Any] = {"start": start, "end": end}
        lemma_values = columns.get("lemma", {})
        token["lemma"] = lemma_values.get(index, text[start:end].casefold())
        for name in ("pos", "tag", "morph"):
            if index in columns.get(name, {}):
                token[name] = columns[name][index]
        output.append(token)
    return tuple(output)


def _segment_projection(segment: RenderSegment) -> dict[str, Any]:
    result: dict[str, Any] = {
        "text": segment.text,
        "language": segment.language,
        "directives": _plain(segment.directives) or {},
        "tokens": [_plain(token) for token in segment.tokens],
        "markers": list(segment.markers),
    }
    if segment.pause_before is not None:
        result["pause_before"] = _plain(segment.pause_before)
    if segment.pause_after is not None:
        result["pause_after"] = _plain(segment.pause_after)
    if segment.heading is not None:
        result["heading"] = segment.heading
    return result


def _flow_plan_id(
    language: str,
    unit: str,
    hash_schema: str,
    flows: Iterable[FlowUnit],
    document_semantics: Mapping[str, Any],
) -> str:
    identity = {
        "format": "utterplan",
        "schema_version": 5,
        "language": language,
        "unit": unit,
        "hash_schema": hash_schema,
        "document_semantics": _plain(document_semantics),
        "flow": [
            {"segments": [_segment_projection(segment) for segment in flow.segments]}
            for flow in flows
        ],
    }
    return _digest(identity)


def semantic_segment_digest(segment: RenderSegment) -> str:
    """Hash one renderer-relevant segment using the versioned VoiceRender projection."""
    return _digest(
        {"schema": SEGMENT_HASH_SCHEMA, "segment": _segment_projection(segment)}
    )


def _voice_reference(directives: Mapping[str, Any], where: str) -> str | None:
    voice = directives.get("voice")
    if voice is None:
        return None
    if isinstance(voice, str):
        return _nonempty(voice, f"{where}.voice")
    if not isinstance(voice, Mapping):
        raise PlanError(f"{where}.voice must be a logical role name or object")
    reference = voice.get("reference")
    return (
        _nonempty(reference, f"{where}.voice.reference")
        if reference is not None
        else None
    )


@dataclass(frozen=True, slots=True)
class FlowUnit:
    index: int
    flow_hash: str
    segments: tuple[RenderSegment, ...]


@dataclass(frozen=True, slots=True)
class UtterPlanV5:
    plan_id: str
    language: str
    unit: str
    hash_schema: str
    file_sha256: str
    flows: tuple[FlowUnit, ...]
    path: Path | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def segments(self) -> tuple[RenderSegment, ...]:
        return tuple(segment for flow in self.flows for segment in flow.segments)

    @property
    def source_file_sha256(self) -> str:
        return "sha256:" + self.file_sha256


class SourceOperationKind(str, Enum):
    SPEECH = "speech"
    EXTERNAL_AUDIO = "external_audio"
    UNSUPPORTED = "unsupported"


def classify_source_operation(segment: RenderSegment) -> SourceOperationKind:
    """Classify one source segment without resolving media or opening an engine."""
    if "audio" in segment.directives:
        return SourceOperationKind.EXTERNAL_AUDIO
    if segment.text.strip():
        return SourceOperationKind.SPEECH
    return SourceOperationKind.UNSUPPORTED


def source_operation_record(segment: RenderSegment) -> dict[str, Any]:
    """Return the stable manifest record for a selected source operation."""
    kind = classify_source_operation(segment)
    source = {
        "locator": segment.source.locator,
        "flow_index": segment.source.flow_index,
        "segment_index": segment.source.segment_index,
        "flow_hash": segment.source.flow_hash,
        "semantic_segment_hash": segment.semantic_segment_hash,
    }
    result: dict[str, Any] = {"source": source, "kind": kind.value}
    if kind is SourceOperationKind.EXTERNAL_AUDIO:
        result["audio_directive"] = _plain(segment.directives["audio"])
    return result


@dataclass(frozen=True, slots=True)
class RenderSelection:
    kind: str
    segments: tuple[RenderSegment, ...]
    take: int | None = None
    locators: tuple[str, ...] = ()
    is_full_plan: bool = False

    @property
    def speech_segments(self) -> tuple[RenderSegment, ...]:
        return tuple(
            segment
            for segment in self.segments
            if classify_source_operation(segment) is SourceOperationKind.SPEECH
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "order": "source",
            "selected_count": len(self.speech_segments),
            "selected_source_count": len(self.segments),
            "selected_speech_count": len(self.speech_segments),
            "is_full_plan": self.is_full_plan,
        }
        if self.take is not None:
            result["take"] = self.take
        if self.kind == "locators":
            result["locators"] = list(self.locators)
        return result


def _parse(
    data: Mapping[str, Any], *, file_sha256: str, path: Path | None
) -> UtterPlanV5:
    unknown = set(data).difference(ROOT_KEYS)
    if unknown:
        raise PlanError(f"Unknown UtterPlan v5 root fields: {sorted(unknown)}")
    if (
        data.get("format") != "utterplan"
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 5
    ):
        raise PlanError("Expected format='utterplan' schema_version=5")
    plan_id = _nonempty(data.get("plan_id"), "plan.plan_id")
    if PLAN_ID_RE.fullmatch(plan_id) is None:
        raise PlanError("plan.plan_id must use sha256:<64 lowercase hex>")
    language = _nonempty(data.get("language"), "plan.language")
    unit = _nonempty(data.get("unit"), "plan.unit")
    hash_schema = data.get("hash_schema")
    if hash_schema != FLOW_HASH_SCHEMA:
        raise PlanError(f"Unsupported UtterPlan flow hash schema: {hash_schema!r}")
    flow_data = data.get("flow")
    if not isinstance(flow_data, list):
        raise PlanError("plan.flow must be an array")

    flows: list[FlowUnit] = []
    for flow_index, raw_flow in enumerate(flow_data):
        if not isinstance(raw_flow, dict) or set(raw_flow).difference(
            {"hash", "segment"}
        ):
            raise PlanError(f"flow[{flow_index}] has an invalid shape")
        flow_hash = _nonempty(raw_flow.get("hash"), f"flow[{flow_index}].hash")
        if PLAN_ID_RE.fullmatch(flow_hash) is None:
            raise PlanError(
                f"flow[{flow_index}].hash must use sha256:<64 lowercase hex>"
            )
        raw_segments = raw_flow.get("segment")
        if not isinstance(raw_segments, list):
            raise PlanError(f"flow[{flow_index}].segment must be an array")

        segments: list[RenderSegment] = []
        for segment_index, raw_segment in enumerate(raw_segments):
            where = f"flow[{flow_index}].segment[{segment_index}]"
            if not isinstance(raw_segment, dict):
                raise PlanError(f"{where} must be a table")
            unsupported = set(raw_segment).difference(SEGMENT_KEYS)
            if unsupported:
                raise PlanError(f"Unknown {where} fields: {sorted(unsupported)}")
            text = raw_segment.get("text")
            if not isinstance(text, str):
                raise PlanError(f"{where}.text must be a string")
            segment_language = _nonempty(
                raw_segment.get("language", language), f"{where}.language"
            )
            directives = _directives(raw_segment, where)
            tokens = _tokens(raw_segment.get("token"), text, f"{where}.token")
            markers_raw = raw_segment.get("marker", [])
            if not isinstance(markers_raw, list) or any(
                not isinstance(marker, str) or not marker for marker in markers_raw
            ):
                raise PlanError(f"{where}.marker must be an array of non-empty strings")
            heading = raw_segment.get("heading")
            if heading is not None and (type(heading) is not int or heading < 1):
                raise PlanError(f"{where}.heading must be a positive integer")
            pause_before = (
                _pause(raw_segment["pause_before"], f"{where}.pause_before")
                if "pause_before" in raw_segment
                else None
            )
            pause_after = (
                _pause(raw_segment["pause_after"], f"{where}.pause_after")
                if "pause_after" in raw_segment
                else None
            )
            logical_voice = _voice_reference(directives, where)
            source = SourceRef(plan_id, flow_index, segment_index, flow_hash)
            segment = RenderSegment(
                source=source,
                text=text,
                language=segment_language,
                logical_voice=logical_voice,
                directives=directives,
                tokens=tokens,
                markers=tuple(markers_raw),
                heading=heading,
                pause_before=pause_before,
                pause_after=pause_after,
            )
            segments.append(
                RenderSegment(
                    source=segment.source,
                    text=segment.text,
                    language=segment.language,
                    logical_voice=segment.logical_voice,
                    directives=segment.directives,
                    tokens=segment.tokens,
                    markers=segment.markers,
                    heading=segment.heading,
                    pause_before=segment.pause_before,
                    pause_after=segment.pause_after,
                    semantic_segment_hash=semantic_segment_digest(segment),
                )
            )

        # UtterPlan flow.hash uses the same segment projection over the entire unit.
        flow_semantics = []
        for segment in segments:
            value = _segment_projection(segment)
            value.pop("heading", None) if segment.heading is None else None
            flow_semantics.append(value)
        actual_flow_hash = _digest({"segments": flow_semantics})
        if actual_flow_hash != flow_hash:
            raise PlanError(
                f"flow hash mismatch at f{flow_index:06d}: expected {flow_hash}, got {actual_flow_hash}"
            )
        flows.append(FlowUnit(flow_index, flow_hash, tuple(segments)))

    if not flows or not any(flow.segments for flow in flows):
        raise PlanError("UtterPlan v5 must contain at least one source segment")
    document_data = data.get("document", {})
    if not isinstance(document_data, Mapping):
        raise PlanError("plan.document must be a table")
    document_semantics = document_data.get("semantics", {})
    if not isinstance(document_semantics, Mapping):
        raise PlanError("plan.document.semantics must be a table")
    calculated_plan_id = _flow_plan_id(
        language, unit, hash_schema, flows, document_semantics
    )
    if calculated_plan_id != plan_id:
        raise PlanError(
            f"plan_id mismatch: expected {plan_id}, computed {calculated_plan_id}"
        )
    if (
        not isinstance(file_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", file_sha256) is None
    ):
        raise PlanError(
            "source file SHA-256 must be 64 lowercase hexadecimal characters"
        )
    metadata = {
        key: _plain(data[key])
        for key in ("document", "linguistics", "producer", "warnings")
        if key in data
    }
    return UtterPlanV5(
        plan_id=plan_id,
        language=language,
        unit=unit,
        hash_schema=hash_schema,
        file_sha256=file_sha256,
        flows=tuple(flows),
        path=path,
        metadata=metadata,
    )


def parse_utterplan_v5(
    data: Mapping[str, Any], *, file_sha256: str, path: str | Path | None = None
) -> UtterPlanV5:
    """Parse an already-decoded v5 table using the exact original-file fingerprint."""
    if not isinstance(data, Mapping):
        raise PlanError("UtterPlan v5 root must be a table")
    location = Path(path).expanduser().resolve() if path is not None else None
    return _parse(data, file_sha256=file_sha256, path=location)


def load_utterplan_v5(path: str | Path) -> UtterPlanV5:
    location = Path(path).expanduser().resolve()
    try:
        source_bytes = location.read_bytes()
        data = tomllib.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PlanError(f"Cannot read UtterPlan v5 {location}: {exc}") from exc
    return _parse(
        data,
        file_sha256=hashlib.sha256(source_bytes).hexdigest(),
        path=location,
    )


def select_segments(
    plan: UtterPlanV5,
    *,
    take: int | None = None,
    select: Iterable[str] | str | None = None,
) -> RenderSelection:
    if take is not None and (type(take) is not int or take < 1):
        raise PlanError("--take must be a positive integer")
    if take is not None and select is not None:
        raise PlanError("Use take or select, not both")
    all_segments = plan.segments
    if select is None:
        selected = all_segments[:take] if take is not None else all_segments
        kind = "take" if take is not None else "all"
        selection = RenderSelection(
            kind=kind,
            segments=selected,
            take=take,
            is_full_plan=len(selected) == len(all_segments),
        )
    else:
        raw = select.split(",") if isinstance(select, str) else list(select)
        if not raw or any(
            not isinstance(item, str) or not item.strip() for item in raw
        ):
            raise PlanError("--select requires one or more flow locators")
        wanted: set[tuple[int, int]] = set()
        for locator in raw:
            match = re.fullmatch(r"f(\d+)\.s(\d+)", locator.strip())
            if match is None:
                raise PlanError(
                    f"Invalid v5 source locator {locator!r}; expected fN.sN"
                )
            address = (int(match.group(1)), int(match.group(2)))
            if address in wanted:
                raise PlanError(f"Duplicate selected source locator {locator!r}")
            wanted.add(address)
        indexed = {
            (segment.source.flow_index, segment.source.segment_index): segment
            for segment in all_segments
        }
        missing = wanted.difference(indexed)
        if missing:
            raise PlanError(f"Unknown v5 source locator(s): {sorted(missing)}")
        selected = tuple(
            segment
            for segment in all_segments
            if (segment.source.flow_index, segment.source.segment_index) in wanted
        )
        selection = RenderSelection(
            kind="locators",
            segments=selected,
            locators=tuple(segment.source.locator for segment in selected),
            is_full_plan=len(selected) == len(all_segments),
        )
    if not selection.segments:
        raise PlanError("No source segments selected")
    return selection


def resolve_logical_voice(segment: RenderSegment, profile: VoiceProfile) -> str:
    role = segment.logical_voice or profile.default_role
    if role is None:
        if len(profile.roles) == 1:
            role = next(iter(profile.roles))
        else:
            raise ProfileError(
                f"No logical voice/default role for source segment {segment.source.locator}; "
                "configure defaults.role or a voice reference"
            )
    if role not in profile.roles:
        raise ProfileError(
            f"No voice binding for logical role {role!r} at {segment.source.locator}"
        )
    return role


_SSMD_RATE_FACTORS = {
    "very-slow": 0.65,
    "slow": 0.80,
    "moderate": 0.90,
    "normal": 1.00,
    "brisk": 1.10,
    "fast": 1.25,
    "very-fast": 1.50,
}
_SSMD_PITCH_PERCENTAGES = {
    "very-low": -20.0,
    "low": -12.0,
    "moderate-low": -6.0,
    "normal": 0.0,
    "moderate-high": 6.0,
    "high": 12.0,
    "very-high": 20.0,
}
_SSMD_VOLUME_DB = {
    "silent": -60.0,
    "x-soft": -12.0,
    "soft": -6.0,
    "medium": 0.0,
    "loud": 6.0,
    "x-loud": 12.0,
    "0": -60.0,
    "1": -12.0,
    "2": -6.0,
    "3": 0.0,
    "4": 6.0,
    "5": 12.0,
}


def _ssmd_rate_factor(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        natural = _SSMD_RATE_FACTORS.get(normalized)
        if natural is not None:
            return natural
        match = re.fullmatch(r"([+-]?)(\d+(?:\.\d+)?)%", normalized)
        if match is not None:
            percentage = float(match.group(2))
            sign = match.group(1)
            factor = (
                1.0 + percentage / 100.0
                if sign == "+"
                else 1.0 - percentage / 100.0
                if sign == "-"
                else percentage / 100.0
            )
            if math.isfinite(factor) and factor > 0:
                return factor
    raise ValueError(f"unsupported rate value: {value!r}")


def _ssmd_pitch_semitones(value: Any) -> float:
    percentage: float | None = None
    if isinstance(value, str):
        normalized = value.strip().lower()
        percentage = _SSMD_PITCH_PERCENTAGES.get(normalized)
        if percentage is None:
            match = re.fullmatch(r"([+-]?)(\d+(?:\.\d+)?)%", normalized)
            if match is not None:
                magnitude = float(match.group(2))
                percentage = -magnitude if match.group(1) == "-" else magnitude
    if percentage is not None:
        ratio = 1.0 + percentage / 100.0
        if math.isfinite(ratio) and ratio > 0:
            return 12.0 * math.log2(ratio)
    raise ValueError(f"unsupported pitch value: {value!r}")


def _ssmd_volume_db(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _SSMD_VOLUME_DB:
            return _SSMD_VOLUME_DB[normalized]
        match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)db", normalized)
        if match is not None:
            decibels = float(match.group(1))
            if math.isfinite(decibels):
                return decibels
    raise ValueError(f"unsupported volume value: {value!r}")


def _unsupported_feature(
    feature: str,
    value: Any,
    segment: RenderSegment,
    capabilities: Any,
) -> None:
    raise UnsupportedSynthesisFeatureError(
        f"Unsupported UtterPlan semantic {feature!r} at {segment.source.locator}",
        engine=getattr(capabilities, "id", None),
        request_id=segment.source.locator,
        details={"feature": feature, "value": repr(value)},
    )


def _lower_prosody(
    value: Mapping[str, Any], segment: RenderSegment, capabilities: Any
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    converters = {
        "rate": ("tempo", "factor", _ssmd_rate_factor),
        "pitch": ("pitch_shift", "semitones", _ssmd_pitch_semitones),
        "volume": ("gain", "db", _ssmd_volume_db),
    }
    for field, (operation, amount_name, convert) in converters.items():
        authored = value.get(field)
        if authored is None:
            continue
        try:
            amount = convert(authored)
        except (TypeError, ValueError, OverflowError):
            _unsupported_feature(f"prosody.{field}", authored, segment, capabilities)
        output[operation] = {amount_name: amount}
    return output


def lower_directives(
    segment: RenderSegment, capabilities: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lower supported request semantics or fail before engine inference."""
    native: dict[str, Any] = {}
    composition: dict[str, Any] = {}
    unsupported: list[str] = []
    for name, value in segment.directives.items():
        if name == "voice":  # Logical role routing is resolved before this phase.
            if isinstance(value, Mapping):
                unsupported.extend(
                    f"voice.{key}" for key in set(value).difference({"reference"})
                )
            elif not isinstance(value, str):
                unsupported.append("voice")
            continue
        if name == "pronunciation":
            alphabet = (
                value.get("alphabet", "ipa") if isinstance(value, Mapping) else None
            )
            supported_alphabets = frozenset(
                getattr(capabilities, "pronunciation_alphabets", ())
            )
            if (
                getattr(capabilities, "supports_pronunciation_overrides", False)
                and alphabet in supported_alphabets
            ):
                native[name] = _plain(value)
            else:
                unsupported.append(name)
            continue
        if name == "prosody":
            composition.update(_lower_prosody(value, segment, capabilities))
            continue
        if name == "extensions":
            # v0.1 has no registered extension handlers; never imply these were honored.
            unsupported.extend(
                f"extensions.{item.get('name', 'unknown')}"
                for item in value
                if isinstance(item, Mapping)
            )
            continue
        unsupported.append(name)

    if segment.tokens:
        if getattr(capabilities, "supports_linguistic_tokens", False):
            native["linguistic_tokens"] = [
                {
                    **dict(_plain(token)),
                    "text": segment.text[token["start"] : token["end"]],
                }
                for token in segment.tokens
            ]
        else:
            unsupported.append("tokens")
    if unsupported:
        raise UnsupportedSynthesisFeatureError(
            f"Unsupported UtterPlan semantics at {segment.source.locator}: "
            f"{', '.join(sorted(unsupported))}",
            engine=getattr(capabilities, "id", None),
            request_id=segment.source.locator,
            details={"features": sorted(unsupported)},
        )
    return native, composition
