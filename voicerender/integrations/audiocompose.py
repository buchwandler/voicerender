"""Materialize verified UtterPlan v5 renders as producer-neutral AudioJobs.

AudioCompose is imported only when this optional integration is invoked; core
rendering and manifest inspection do not depend on it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from ..errors import ArtifactError, PausePolicyError, PlanError, ProfileError
from ..inputs.utterplan_v5 import (
    RenderSelection,
    UtterPlanV5,
    load_utterplan_v5,
    select_segments,
)
from ..manifest import read_manifest, validate_manifest_source
from ..models import RenderSegment

PAUSE_SCHEMA = "voicerender.pause-policy.v1"
PAUSE_CLASSES = frozenset(
    {
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
)
_DURATION = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(ms|s)$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _duration(value: Any, where: str) -> float:
    if not isinstance(value, str):
        raise PausePolicyError(
            f"{where} must be a non-negative duration such as '320ms'"
        )
    match = _DURATION.fullmatch(value)
    if match is None:
        raise PausePolicyError(f"{where} has invalid duration {value!r}; use ms or s")
    number = value[: -len(match.group(1))]
    amount = float(number)
    seconds = amount / 1000.0 if match.group(1) == "ms" else amount
    if not math.isfinite(seconds) or seconds < 0:
        raise PausePolicyError(f"{where} must be a finite duration >= 0")
    return seconds


def _parse_pause_table(value: Any, where: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise PausePolicyError(f"{where} must be a TOML table")
    unknown = set(value).difference(PAUSE_CLASSES)
    if unknown:
        raise PausePolicyError(
            f"{where} has unknown pause class(es): {sorted(unknown)}"
        )
    parsed = {name: _duration(item, f"{where}.{name}") for name, item in value.items()}
    if "none" in parsed and parsed["none"] != 0:
        raise PausePolicyError(f"{where}.none must resolve to zero")
    return parsed


@dataclass(frozen=True, slots=True)
class PausePolicy:
    """Validated symbolic-to-numeric pause policy."""

    default: Mapping[str, float]
    voices: Mapping[str, Mapping[str, float]]
    merge: str
    policy_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.default, Mapping) or not isinstance(
            self.voices, Mapping
        ):
            raise PausePolicyError(
                "PausePolicy defaults and voice overrides must be mappings"
            )
        if any(not isinstance(role, str) or not role.strip() for role in self.voices):
            raise PausePolicyError(
                "PausePolicy voice override names must be non-empty strings"
            )
        if self.merge != "max":
            raise PausePolicyError("PausePolicy merge must be 'max'")
        if (
            not isinstance(self.policy_id, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.policy_id) is None
        ):
            raise PausePolicyError(
                "PausePolicy policy_id must be a canonical SHA-256 identity"
            )
        tables = {
            "default": self.default,
            **{f"voice.{role}": values for role, values in self.voices.items()},
        }
        for where, table in tables.items():
            if not isinstance(table, Mapping) or set(table).difference(PAUSE_CLASSES):
                raise PausePolicyError(
                    f"{where} contains an invalid pause-class mapping"
                )
            for name, seconds in table.items():
                if (
                    isinstance(seconds, bool)
                    or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds)
                    or seconds < 0
                ):
                    raise PausePolicyError(
                        f"{where}.{name} must be finite seconds >= 0"
                    )
            if table.get("none", 0) != 0:
                raise PausePolicyError(f"{where}.none must resolve to zero")

    @classmethod
    def load(cls, path: str | Path) -> PausePolicy:
        location = Path(path).expanduser()
        if location.is_symlink():
            raise PausePolicyError(f"Pause policy must not be a symlink: {location}")
        try:
            raw_bytes = location.read_bytes()
            document = tomllib.loads(raw_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise PausePolicyError(
                f"Cannot read pause policy {location}: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise PausePolicyError("Pause policy root must be a TOML table")
        unknown = set(document).difference({"schema", "merge", "default", "voice"})
        if unknown:
            raise PausePolicyError(
                f"Pause policy has unknown root field(s): {sorted(unknown)}"
            )
        if document.get("schema") != PAUSE_SCHEMA:
            raise PausePolicyError(f"Expected pause policy schema {PAUSE_SCHEMA!r}")
        if document.get("merge") != "max":
            raise PausePolicyError("Pause policy merge must be 'max'")
        default = _parse_pause_table(document.get("default"), "default")
        raw_voices = document.get("voice", {})
        if not isinstance(raw_voices, Mapping):
            raise PausePolicyError("voice must be a table of logical-voice overrides")
        voices: dict[str, dict[str, float]] = {}
        for role, table in raw_voices.items():
            if not isinstance(role, str) or not role.strip():
                raise PausePolicyError("voice override names must be non-empty strings")
            voices[role] = _parse_pause_table(table, f"voice.{role}")
        return cls(
            default=default,
            voices=voices,
            merge="max",
            policy_id="sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
        )

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> PausePolicy:
        """Build a policy from a decoded mapping, useful for library callers."""
        if not isinstance(document, Mapping):
            raise PausePolicyError("Pause policy must be a mapping")
        unknown = set(document).difference({"schema", "merge", "default", "voice"})
        if unknown:
            raise PausePolicyError(
                f"Pause policy has unknown root field(s): {sorted(unknown)}"
            )
        if document.get("schema") != PAUSE_SCHEMA or document.get("merge") != "max":
            raise PausePolicyError(
                "Pause policy requires the v1 schema and merge='max'"
            )
        default = _parse_pause_table(document.get("default"), "default")
        raw_voices = document.get("voice", {})
        if not isinstance(raw_voices, Mapping):
            raise PausePolicyError("voice must be a mapping of logical-voice overrides")
        voices = {
            role: _parse_pause_table(table, f"voice.{role}")
            for role, table in raw_voices.items()
            if isinstance(role, str) and role.strip()
        }
        if len(voices) != len(raw_voices):
            raise PausePolicyError("voice override names must be non-empty strings")
        normalized = {
            "schema": PAUSE_SCHEMA,
            "merge": "max",
            "default": dict(default),
            "voice": {name: dict(table) for name, table in sorted(voices.items())},
        }
        return cls(
            default=default,
            voices=voices,
            merge="max",
            policy_id="sha256:"
            + hashlib.sha256(_canonical_json(normalized)).hexdigest(),
        )

    def resolve(self, intent: Any, *, voice: str, where: str) -> float:
        if intent is None:
            return 0.0
        if isinstance(intent, str):
            if intent not in PAUSE_CLASSES:
                raise PausePolicyError(f"{where}: unknown symbolic pause {intent!r}")
            if intent == "none":
                return 0.0
            value = self.voices.get(voice, {}).get(intent, self.default.get(intent))
            if value is None:
                raise PausePolicyError(
                    f"{where}: no duration configured for pause {intent!r} "
                    f"(logical voice {voice!r})"
                )
            return value
        if isinstance(intent, Mapping):
            if set(intent) != {"type", "time"} or intent.get("type") != "timed":
                raise PausePolicyError(
                    f"{where}: timed pause must contain exactly type and time"
                )
            return _duration(intent["time"], f"{where}.time")
        raise PausePolicyError(f"{where}: invalid pause intent {intent!r}")


def load_pause_policy(
    value: PausePolicy | str | Path | Mapping[str, Any],
) -> PausePolicy:
    if isinstance(value, PausePolicy):
        return value
    if isinstance(value, (str, Path)):
        return PausePolicy.load(value)
    if isinstance(value, Mapping):
        return PausePolicy.from_mapping(value)
    raise PausePolicyError(
        "pause_policy must be a policy path, mapping, or PausePolicy"
    )


def _selection_from_metadata(
    plan: UtterPlanV5,
    metadata: Mapping[str, Any],
) -> RenderSelection:
    kind = metadata["kind"]
    if kind == "all":
        selected = select_segments(plan)
    elif kind == "take":
        selected = select_segments(plan, take=metadata["take"])
    elif kind == "locators":
        selected = select_segments(plan, select=metadata["locators"])
    else:  # guarded by manifest validation
        raise ArtifactError(f"Unsupported render selection kind: {kind!r}")
    if selected.as_dict() != dict(metadata):
        raise ArtifactError(
            f"Manifest selection metadata does not match the plan selection for {plan.plan_id}"
        )
    return selected


def _requested_selection(
    plan: UtterPlanV5,
    requested: Any,
    authoritative: RenderSelection,
) -> RenderSelection:
    if requested is None:
        return authoritative
    if isinstance(requested, RenderSelection):
        locators = [segment.source.locator for segment in requested.segments]
        chosen = select_segments(plan, select=locators)
        if chosen.as_dict() != requested.as_dict():
            raise PlanError(
                "Requested RenderSelection is not a canonical selection of this plan"
            )
    elif type(requested) is int:
        chosen = select_segments(plan, take=requested)
    elif isinstance(requested, Iterable):
        chosen = select_segments(plan, select=requested)
    else:
        raise PlanError(
            "selection must be a take count, source locator(s), or RenderSelection"
        )
    if chosen.as_dict() != authoritative.as_dict():
        raise ArtifactError(
            "Requested AudioJob selection differs from the render manifest"
        )
    return chosen


def _operation_from_record(
    record: Mapping[str, Any], *, locator: str, index: int, ac: Any
) -> Any:
    if set(record) != {"name", "value"}:
        raise ArtifactError(
            f"{locator}: composition directive {index} has an invalid shape"
        )
    name = record["name"]
    value = record["value"]
    if not isinstance(name, str) or not name:
        raise ArtifactError(f"{locator}: composition directive {index} has no name")
    if not isinstance(value, Mapping):
        raise ArtifactError(
            f"{locator}: composition directive {name!r} must carry a typed numeric operation"
        )
    kind = value.get("type")
    if not isinstance(kind, str):
        raise ArtifactError(
            f"{locator}: composition directive {name!r} has no operation type"
        )
    fields = {
        "gain": ("db",),
        "tempo": ("factor",),
        "pitch": ("semitones",),
        "fade_in": ("seconds",),
        "fade_out": ("seconds",),
    }
    if kind in fields:
        expected = {"type", *fields[kind]}
        if set(value) != expected:
            raise ArtifactError(f"{locator}: malformed numeric {kind!r} operation")
        if name not in {kind, "prosody", "emphasis", "operation"}:
            raise ArtifactError(
                f"{locator}: composition directive {name!r} does not match {kind!r} operation"
            )
        constructor = {
            "gain": ac.Gain,
            "tempo": ac.Tempo,
            "pitch": ac.PitchShift,
            "fade_in": ac.FadeIn,
            "fade_out": ac.FadeOut,
        }[kind]
        try:
            return constructor(value[fields[kind][0]])
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                f"{locator}: invalid numeric {kind!r} operation: {exc}"
            ) from exc
    if kind == "rate_pitch_envelope":
        required = {"type", "interpolation", "time_base", "rate", "pitch"}
        if set(value) != required or name not in {
            "rate_pitch_envelope",
            "prosody",
            "operation",
        }:
            raise ArtifactError(f"{locator}: malformed rate/pitch envelope operation")
        if not isinstance(value["rate"], list) or not isinstance(value["pitch"], list):
            raise ArtifactError(f"{locator}: rate/pitch envelope curves must be arrays")
        if any(
            not isinstance(point, Mapping) or set(point) != {"seconds", "factor"}
            for point in value["rate"]
        ) or any(
            not isinstance(point, Mapping) or set(point) != {"seconds", "semitones"}
            for point in value["pitch"]
        ):
            raise ArtifactError(
                f"{locator}: rate/pitch envelope points have invalid fields"
            )
        try:
            rate = tuple(
                ac.AutomationPoint(point["seconds"], point["factor"])
                for point in value["rate"]
            )
            pitch = tuple(
                ac.AutomationPoint(point["seconds"], point["semitones"])
                for point in value["pitch"]
            )
            return ac.RatePitchEnvelope(
                rate=rate,
                pitch_semitones=pitch,
                interpolation=value["interpolation"],
                time_base=value["time_base"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(
                f"{locator}: invalid rate/pitch envelope: {exc}"
            ) from exc
    raise ArtifactError(
        f"{locator}: unsupported numeric AudioCompose operation {kind!r}; refusing to drop semantics"
    )


def _spans_for_segment(
    segment: RenderSegment,
    timing: Mapping[str, Any],
    locator: str,
    ac: Any,
) -> tuple[Any, ...]:
    spans = []
    cursor = 0
    for index, raw in enumerate(timing["spans"]):
        text = raw.get("text")
        if "text_start" in raw or "text_end" in raw:
            if "text_start" not in raw or "text_end" not in raw:
                raise ArtifactError(
                    f"{locator}: timing span {index} needs both text offsets"
                )
            source_start, source_end = raw["text_start"], raw["text_end"]
            if (
                type(source_start) is not int
                or type(source_end) is not int
                or source_start < 0
                or source_end < source_start
                or source_end > len(segment.text)
            ):
                raise ArtifactError(
                    f"{locator}: timing span {index} text offsets are invalid"
                )
            if text is not None and segment.text[source_start:source_end] != text:
                raise ArtifactError(
                    f"{locator}: timing span {index} text does not match its offsets"
                )
        elif isinstance(text, str):
            found = segment.text.find(text, cursor)
            if found >= 0:
                source_start, source_end = found, found + len(text)
                cursor = source_end
            else:
                # Preserve sample timing even when an engine's surface form does
                # not occur literally in the prepared input text.
                source_start = source_end = 0
        else:
            source_start = source_end = 0
        metadata: dict[str, Any] = {"provenance": "provided"}
        if isinstance(text, str):
            metadata["text"] = text
        if "kind" in raw:
            metadata["kind"] = raw["kind"]
        spans.append(
            ac.AudioSpan(
                source_start,
                source_end,
                raw["start_frame"],
                raw["end_frame"],
                id=f"{locator}:span:{index}",
                metadata=metadata,
            )
        )
    return tuple(spans)


def _validate_lowering(segment: RenderSegment, entry: Mapping[str, Any]) -> None:
    locator = segment.source.locator
    lowering = entry["semantic_lowering"]
    native = lowering["native_directives"]
    composition = lowering["composition_directives"]
    native_names = [item["name"] for item in native]
    composition_names = [item["name"] for item in composition]
    if len(set(native_names)) != len(native_names) or len(
        set(composition_names)
    ) != len(composition_names):
        raise ArtifactError(f"{locator}: duplicate semantic-lowering directive")
    if set(native_names).intersection(composition_names):
        raise ArtifactError(
            f"{locator}: a directive is declared both native and composition-side"
        )
    expected = set(segment.directives).difference({"voice"})
    actual = set(native_names).union(composition_names)
    if expected != actual:
        raise ArtifactError(
            f"{locator}: manifest semantic-lowering names differ from plan; "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for item in native:
        if item["value"] != segment.directives[item["name"]]:
            raise ArtifactError(
                f"{locator}: native directive {item['name']!r} differs from plan"
            )


def _resolve_join(
    plan: UtterPlanV5,
    manifest_path: Path,
    document: dict[str, Any],
    selection: Any,
    pause_policy: PausePolicy,
) -> tuple[tuple[RenderSegment, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        validate_manifest_source(document, plan)
    except ArtifactError as exc:
        raise ArtifactError(
            f"plan_id {plan.plan_id}: render manifest {manifest_path} source mismatch: {exc}"
        ) from exc
    authoritative = _selection_from_metadata(plan, document["selection"])
    chosen = _requested_selection(plan, selection, authoritative)
    entries = document["segments"]
    if len(entries) != len(chosen.segments):
        raise ArtifactError(
            f"plan_id {plan.plan_id}: manifest {manifest_path} selected entry count mismatch"
        )
    indexed: list[tuple[RenderSegment, Mapping[str, Any], Path]] = []
    root = manifest_path.parent.resolve(strict=True)
    for segment, entry in zip(chosen.segments, entries, strict=True):
        locator = segment.source.locator
        source = entry["source"]
        comparisons = {
            "locator": (locator, source["locator"]),
            "flow_index": (segment.source.flow_index, source["flow_index"]),
            "segment_index": (segment.source.segment_index, source["segment_index"]),
            "flow_hash": (segment.source.flow_hash, source["flow_hash"]),
            "semantic_segment_hash": (
                segment.semantic_segment_hash,
                source["semantic_segment_hash"],
            ),
            "language": (segment.language, entry["language"]),
        }
        for field, (expected, actual) in comparisons.items():
            if expected != actual:
                raise ArtifactError(
                    f"plan_id {plan.plan_id} {locator}: render manifest {manifest_path} "
                    f"{field} mismatch: expected {expected!r}, got {actual!r}"
                )
        if (
            segment.logical_voice is not None
            and segment.logical_voice != entry["logical_voice"]
        ):
            raise ArtifactError(
                f"plan_id {plan.plan_id} {locator}: logical_voice mismatch; "
                f"expected {segment.logical_voice!r}, got {entry['logical_voice']!r}"
            )
        if entry["status"] not in {"rendered", "reused"} or not document["complete"]:
            raise ArtifactError(
                f"plan_id {plan.plan_id} {locator}: manifest {manifest_path} is incomplete "
                f"(status={entry['status']!r}, complete={document['complete']!r})"
            )
        if (
            entry["acoustic_cache_key"] == ""
            or entry["synthesis_semantics_sha256"] == ""
        ):
            raise ArtifactError(
                f"{locator}: manifest is missing acoustic identity metadata"
            )
        _validate_lowering(segment, entry)
        audio = entry["audio"]
        resolved = (root / audio["path"]).resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ArtifactError(
                f"plan_id {plan.plan_id} {locator}: audio path escapes manifest bundle {manifest_path}"
            )
        indexed.append((segment, entry, resolved))

    positions = [
        next(
            index
            for index, candidate in enumerate(plan.segments)
            if candidate.source.locator == segment.source.locator
        )
        for segment, _, _ in indexed
    ]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise ArtifactError(
            f"plan_id {plan.plan_id}: AudioJob composition requires a contiguous source selection"
        )

    boundary_records: list[dict[str, Any]] = []
    for index in range(len(indexed) - 1):
        left, left_entry, _ = indexed[index]
        right, right_entry, _ = indexed[index + 1]
        left_seconds = pause_policy.resolve(
            left.pause_after,
            voice=left_entry["logical_voice"],
            where=f"plan_id {plan.plan_id} {left.source.locator} pause_after",
        )
        right_seconds = pause_policy.resolve(
            right.pause_before,
            voice=right_entry["logical_voice"],
            where=f"plan_id {plan.plan_id} {right.source.locator} pause_before",
        )
        seconds = max(left_seconds, right_seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise PausePolicyError(
                f"plan_id {plan.plan_id} {left.source.locator}->{right.source.locator}: "
                "resolved pause must be finite and >= 0"
            )
        boundary_records.append(
            {
                "left": left,
                "right": right,
                "seconds": seconds,
                "left_intent": left.pause_after,
                "right_intent": right.pause_before,
            }
        )
    return (
        chosen.segments,
        [
            {"segment": segment, "entry": entry, "path": path}
            for segment, entry, path in indexed
        ],
        boundary_records,
    )


def build_audio_job(
    plan: UtterPlanV5 | str | Path,
    render_manifest: str | Path,
    pause_policy: PausePolicy | str | Path | Mapping[str, Any],
    *,
    output_policy: Any = None,
    selection: RenderSelection | int | Iterable[str] | str | None = None,
) -> Any:
    """Validate a v5 render bundle and return a normal AudioCompose ``AudioJob``.

    If ``selection`` is omitted, the manifest's selection is authoritative. An
    explicit selection is accepted only when it exactly matches that manifest.
    A non-prefix locator selection is composable only when its source positions
    are contiguous; skipped source segments are never implicitly joined.
    """
    parsed = load_utterplan_v5(plan) if isinstance(plan, (str, Path)) else plan
    if not isinstance(parsed, UtterPlanV5):
        raise PlanError("plan must be a UtterPlanV5 value or path")
    manifest_path = Path(render_manifest).expanduser().absolute()
    try:
        document = read_manifest(manifest_path, verify_audio=True)
    except ArtifactError as exc:
        raise ArtifactError(f"Render manifest {manifest_path}: {exc}") from exc
    policy = load_pause_policy(pause_policy)
    _, clip_records, boundaries = _resolve_join(
        parsed, manifest_path, document, selection, policy
    )

    try:
        import audiocompose as ac
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfileError(
            "AudioJob materialization requires the optional 'audiocompose' package"
        ) from exc

    if output_policy is None:
        output_policy = ac.OutputPolicy()
    elif not isinstance(output_policy, ac.OutputPolicy):
        raise ProfileError("output_policy must be an audiocompose.OutputPolicy value")

    boundary_by_left = {
        record["left"].source.locator: record
        for record in boundaries
        if record["seconds"] > 0
    }
    clip_specs: list[dict[str, Any]] = []
    for record in clip_records:
        segment = record["segment"]
        entry = record["entry"]
        audio = entry["audio"]
        locator = segment.source.locator
        operations = tuple(
            _operation_from_record(item, locator=locator, index=index, ac=ac)
            for index, item in enumerate(
                entry["semantic_lowering"]["composition_directives"]
            )
        )
        spans = _spans_for_segment(segment, entry["timing"], locator, ac)
        metadata = {
            "kind": "speech",
            "utterplan.plan_id": parsed.plan_id,
            "utterplan.flow_index": segment.source.flow_index,
            "utterplan.segment_index": segment.source.segment_index,
            "utterplan.flow_hash": segment.source.flow_hash,
            "voicerender.semantic_segment_hash": segment.semantic_segment_hash,
            "voicerender.acoustic_cache_key": entry["acoustic_cache_key"],
            "voicerender.render_manifest_id": document["manifest_id"],
            "voicerender.logical_voice": entry["logical_voice"],
            "voicerender.timing_provenance": entry["timing"]["provenance"],
        }
        source = ac.AudioFileSource(
            record["path"],
            expected_sha256=audio["sha256"],
            sample_rate=audio["sample_rate"],
            channels=audio["channels"],
            frames=audio["frames"],
        )
        clip_specs.append(
            {
                "id": f"speech:{locator}",
                "source": source,
                "operations": operations,
                "spans": spans,
                "metadata": metadata,
                "locator": locator,
            }
        )

    # No AudioClip is constructed until every selected entry, pause and numeric
    # operation has passed preflight.
    items: list[Any] = []
    for spec in clip_specs:
        locator = spec["locator"]
        items.append(
            ac.AudioClip(
                id=spec["id"],
                source=spec["source"],
                operations=spec["operations"],
                spans=spec["spans"],
                metadata=spec["metadata"],
            )
        )
        boundary = boundary_by_left.get(locator)
        if boundary is not None:
            right = boundary["right"]
            items.append(
                ac.Silence(
                    id=f"pause:{locator}->{right.source.locator}",
                    seconds=boundary["seconds"],
                    metadata={
                        "kind": "pause",
                        "utterplan.plan_id": parsed.plan_id,
                        "utterplan.left_locator": locator,
                        "utterplan.right_locator": right.source.locator,
                        "voicerender.pause_policy_id": policy.policy_id,
                        "voicerender.merge": "max",
                        "voicerender.left_intent": boundary["left_intent"],
                        "voicerender.right_intent": boundary["right_intent"],
                    },
                )
            )
    job_source = {
        "utterplan.plan_id": parsed.plan_id,
        "utterplan.plan_file_sha256": "sha256:" + parsed.file_sha256,
        "voicerender.render_manifest_id": document["manifest_id"],
        "voicerender.pause_policy_id": policy.policy_id,
        "voicerender.selection": dict(document["selection"]),
        "voicerender.materializer": "voicerender-audiocompose.v1",
    }
    producer = {"name": "voicerender", "version": document["producer"]["version"]}
    job = ac.AudioJob(
        items=tuple(items),
        output=output_policy,
        producer=producer,
        source=job_source,
    )
    try:
        # AudioCompose's public serializer supplies the canonical source and
        # operation projection. Hash that payload using its documented JSON
        # canonicalization so in-memory and saved jobs have deterministic IDs.
        payload = job.to_dict(base_dir=str(manifest_path.parent.resolve()))
        payload.pop("job_id", None)
        job_id = "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()
        job = ac.AudioJob(
            items=tuple(items),
            output=output_policy,
            producer=producer,
            job_id=job_id,
            source=job_source,
        )
        job.validate(base_dir=str(manifest_path.parent.resolve()), verify_sources=True)
    except Exception as exc:
        if isinstance(exc, (ArtifactError, PausePolicyError, PlanError, ProfileError)):
            raise
        raise ArtifactError(f"AudioCompose rejected materialized job: {exc}") from exc
    return job


__all__ = ["PausePolicy", "build_audio_job", "load_pause_policy"]
