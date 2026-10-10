"""Direct UtterPlan v5 synthesis into verifiable render-manifest v2 bundles."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import soundfile as sf

from .capacity import CAPACITY_SCHEMA, CapacityContext
from .engines import get_adapter, open_session, resolve_voice, synthesize
from .engines.base import (
    PronunciationSpan,
    SpeechRequest,
    SpeechToken,
    SpeechWordTiming,
)
from .engines.selection import EngineRequest
from .errors import (
    ArtifactError,
    EngineApiIncompatibleError,
    EngineSelectionError,
    EngineUnavailableError,
    PlanError,
    ProfileError,
    RenderEventCallbackError,
    UnsupportedPlanSemanticsError,
)
from .events import RenderEvent, RenderEventCallback, RenderEventKind
from .inputs.utterplan_v5 import (
    RenderSelection,
    SourceOperationKind,
    UtterPlanV5,
    classify_source_operation,
    load_utterplan_v5,
    lower_directives,
    resolve_logical_voice,
    select_segments,
    source_operation_record,
)
from .manifest import (
    EDGE_POLICY,
    ENGINE_IDENTITY_SCHEMA,
    MANIFEST_FILENAME,
    MANIFEST_ID_ALGORITHM,
    MANIFEST_SCHEMA,
    NUMERIC_COMPOSITION_SCHEMA,
    read_manifest,
    resolve_bundle_artifact,
    seal_manifest,
    validate_manifest_document,
    verify_manifest_audio,
)
from .models import RenderResult, VoiceProfile
from .render import _exclusive_run, _identity
from .storage import atomic_json, atomic_wav, digest, file_hash, verified_wav

V5_RENDER_ALGORITHM = "voicerender.utterplan-v5-direct.v1"
CACHE_KEY_SCHEMA = "voicerender.acoustic-cache-key.v3"
SYNTHESIS_SEMANTICS_SCHEMA = "voicerender.synthesis-semantics.v2"


def _emit_event(
    callback: RenderEventCallback | None,
    kind: RenderEventKind,
    **fields: Any,
) -> None:
    if callback is None:
        return
    event = RenderEvent(kind=kind, **fields)
    try:
        callback(event)
    except Exception as exc:
        raise RenderEventCallbackError(kind.value, exc) from exc


try:
    _VERSION = version("voicerender")
except PackageNotFoundError:
    _VERSION = "0+unknown"


def _versioned_hash(value: Any) -> str:
    return "sha256:" + digest(value)


def _directive_items(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": name, "value": values[name]} for name in sorted(values)]


def _selection_data(selection: RenderSelection) -> dict[str, Any]:
    data = selection.as_dict()
    return data


def _valid_transition_duration(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        if isinstance(value, (int, float)):
            seconds = float(value)
        elif isinstance(value, str):
            match = re.fullmatch(
                r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(ms|s)?\s*",
                value,
                re.IGNORECASE,
            )
            if match is None:
                return False
            seconds = float(match.group(1))
            if match.group(2) and match.group(2).lower() == "ms":
                seconds /= 1000.0
        else:
            return False
    except (OverflowError, ValueError):
        return False
    return math.isfinite(seconds) and seconds >= 0


def _preflight_document_semantics(
    plan: UtterPlanV5, profile: VoiceProfile
) -> dict[str, Any]:
    document = (plan.metadata or {}).get("document", {})
    semantics = document.get("semantics", {}) if isinstance(document, Mapping) else {}
    if not isinstance(semantics, Mapping):
        raise PlanError("UtterPlan document.semantics must be a table")

    unknown = set(semantics).difference(
        {"voice_bindings", "prosody_transitions", "requires"}
    )
    if unknown:
        raise UnsupportedPlanSemanticsError(
            "Unsupported UtterPlan document semantics: " + ", ".join(sorted(unknown)),
            details={"semantics": sorted(unknown)},
        )

    voice_bindings = semantics.get("voice_bindings", {})
    if not isinstance(voice_bindings, Mapping):
        raise PlanError("UtterPlan document.semantics.voice_bindings must be a table")
    invalid_bindings = []
    for role, binding in voice_bindings.items():
        if role not in profile.roles or binding != "renderer-owned-choice":
            invalid_bindings.append(str(role))
    if invalid_bindings:
        raise UnsupportedPlanSemanticsError(
            "Unsupported UtterPlan voice binding metadata: "
            + ", ".join(sorted(invalid_bindings)),
            details={"roles": sorted(invalid_bindings)},
        )

    transitions = semantics.get("prosody_transitions", {})
    if not isinstance(transitions, Mapping):
        raise PlanError(
            "UtterPlan document.semantics.prosody_transitions must be a table"
        )
    unknown_transition_fields = set(transitions).difference(
        {"enabled", "same_voice_only", "rate", "pitch", "volume"}
    )
    if unknown_transition_fields:
        raise UnsupportedPlanSemanticsError(
            "Unsupported prosody transition fields: "
            + ", ".join(sorted(unknown_transition_fields)),
            details={"fields": sorted(unknown_transition_fields)},
        )
    for name in ("enabled", "same_voice_only"):
        if name in transitions and not isinstance(transitions[name], bool):
            raise PlanError(
                f"UtterPlan document.semantics.prosody_transitions.{name} must be boolean"
            )
    for name in ("rate", "pitch", "volume"):
        if (
            name in transitions
            and transitions[name] is not None
            and not _valid_transition_duration(transitions[name])
        ):
            raise PlanError(
                f"UtterPlan document.semantics.prosody_transitions.{name} "
                "must be a finite non-negative duration"
            )

    requirements = semantics.get("requires", {})
    if not isinstance(requirements, Mapping):
        raise PlanError("UtterPlan document.semantics.requires must be a table")
    unknown_requirements = set(requirements).difference({"extensions"})
    if unknown_requirements:
        raise UnsupportedPlanSemanticsError(
            "Unsupported UtterPlan document requirements: "
            + ", ".join(sorted(unknown_requirements)),
            details={"requirements": sorted(unknown_requirements)},
        )
    extensions = requirements.get("extensions", [])
    if not isinstance(extensions, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in extensions
    ):
        raise PlanError("UtterPlan required extensions must be non-empty strings")
    if extensions:
        raise UnsupportedPlanSemanticsError(
            "Unsupported required UtterPlan extensions: " + ", ".join(extensions),
            details={"extensions": list(extensions)},
        )
    return dict(semantics)


def _safe_target(bundle: Path, relative_path: str) -> Path:
    target = bundle / relative_path
    current = bundle
    for part in Path(relative_path).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ArtifactError(
                f"Output bundle path contains a symlink: {relative_path}"
            )
        if current.exists():
            if not current.is_dir():
                raise ArtifactError(
                    f"Output bundle parent is not a directory: {current}"
                )
        else:
            current.mkdir()
    if target.is_symlink():
        raise ArtifactError(f"Output bundle artifact is a symlink: {relative_path}")
    if target.exists() and not target.is_file():
        raise ArtifactError(
            f"Output bundle artifact is not a regular file: {relative_path}"
        )
    return target


def _pcm16(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with sf.SoundFile(path) as wav:
            return bool(
                wav.format == "WAV" and wav.channels == 1 and wav.subtype == "PCM_16"
            )
    except (OSError, RuntimeError):
        return False


def _copy_verified(source: Path, target: Path, info: dict[str, Any]) -> None:
    if source.is_symlink() or not source.is_file():
        raise ArtifactError(f"WAV source is not a regular file: {source}")
    fd, name = tempfile.mkstemp(prefix=".tmp-", suffix=".wav", dir=target.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as destination, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, destination)
            destination.flush()
            os.fsync(destination.fileno())
        if not verified_wav(temp, info) or not _pcm16(temp):
            raise ArtifactError(f"WAV copy failed integrity validation: {source}")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _read_cache(cache_dir: Path, key: str) -> tuple[Path, dict[str, Any]] | None:
    wav_path = cache_dir / f"{key}.wav"
    sidecar_path = cache_dir / f"{key}.json"
    if wav_path.is_symlink() or sidecar_path.is_symlink():
        return None
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(data, dict)
        or data.get("key") != key
        or data.get("encoding") != "PCM_16"
        or not isinstance(data.get("wav"), dict)
        or not verified_wav(wav_path, data["wav"])
        or not _pcm16(wav_path)
    ):
        return None
    try:
        data["timing"] = _validated_timing(data.get("timing"), data["wav"])
    except ArtifactError:
        return None
    return wav_path, data


def _timing_spans(
    values: Iterable[Mapping[str, Any] | SpeechWordTiming],
    *,
    sample_rate: int,
    frames: int,
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    previous_end = 0
    for index, value in enumerate(values):
        if isinstance(value, SpeechWordTiming):
            value = {
                "start_frame": value.start_sample,
                "end_frame": value.end_sample,
                "text": value.text,
                "text_start": value.char_start,
                "text_end": value.char_end,
            }
        if not isinstance(value, Mapping):
            raise ArtifactError(f"Engine timing entry {index} must be an object")
        if "start_frame" in value and "end_frame" in value:
            start, end = value["start_frame"], value["end_frame"]
        elif "start" in value and "end" in value:
            start_seconds, end_seconds = value["start"], value["end"]
            if (
                isinstance(start_seconds, bool)
                or isinstance(end_seconds, bool)
                or not isinstance(start_seconds, (int, float))
                or not isinstance(end_seconds, (int, float))
                or not math.isfinite(start_seconds)
                or not math.isfinite(end_seconds)
                or start_seconds < 0
                or end_seconds < 0
            ):
                raise ArtifactError(f"Engine timing entry {index} has invalid seconds")
            start = round(start_seconds * sample_rate)
            end = round(end_seconds * sample_rate)
        else:
            raise ArtifactError(
                f"Engine timing entry {index} needs start_frame/end_frame or start/end seconds"
            )
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < previous_end
            or start >= end
            or end > frames
        ):
            raise ArtifactError(
                f"Engine timing entry {index} is out of order or out of range"
            )
        span: dict[str, Any] = {"start_frame": start, "end_frame": end}
        text = value.get("text", value.get("word"))
        if text is not None:
            if not isinstance(text, str):
                raise ArtifactError(
                    f"Engine timing entry {index} text must be a string"
                )
            span["text"] = text
        kind = value.get("kind")
        if kind is not None:
            if not isinstance(kind, str) or not kind:
                raise ArtifactError(
                    f"Engine timing entry {index} kind must be a non-empty string"
                )
            span["kind"] = kind
        has_text_start = "text_start" in value
        has_text_end = "text_end" in value
        if has_text_start != has_text_end:
            raise ArtifactError(f"Engine timing entry {index} needs both text offsets")
        if has_text_start:
            text_start, text_end = value["text_start"], value["text_end"]
            if (
                isinstance(text_start, bool)
                or isinstance(text_end, bool)
                or not isinstance(text_start, int)
                or not isinstance(text_end, int)
                or text_start < 0
                or text_end < text_start
            ):
                raise ArtifactError(
                    f"Engine timing entry {index} has invalid text offsets"
                )
            span["text_start"] = text_start
            span["text_end"] = text_end
        spans.append(span)
        previous_end = end
    return spans


def _validated_timing(value: Any, info: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "provenance",
        "unit",
        "sample_rate",
        "spans",
    }:
        raise ArtifactError("Cached timing record has an invalid shape")
    if value["unit"] != "frames" or value["provenance"] not in {"absent", "provided"}:
        raise ArtifactError("Cached timing provenance is invalid")
    if value["sample_rate"] != info["sample_rate"] or not isinstance(
        value["spans"], list
    ):
        raise ArtifactError("Cached timing rate or spans do not match the WAV")
    spans = _timing_spans(
        value["spans"], sample_rate=info["sample_rate"], frames=info["frames"]
    )
    if spans != value["spans"] or (value["provenance"] == "provided") != bool(spans):
        raise ArtifactError("Cached timing spans do not match their provenance")
    return {**value, "spans": spans}


def _assert_reference_unchanged(binding: Any, identity: dict[str, Any]) -> None:
    reference = binding.reference_audio
    if reference is None:
        return
    try:
        actual = file_hash(reference)
    except OSError as exc:
        raise ProfileError(f"Reference audio is unavailable: {reference}") from exc
    if actual != identity["reference_sha256"]:
        raise ProfileError(f"Reference audio changed during this render: {reference}")


def _audio_record(relative_path: str, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": relative_path,
        "sha256": info["sha256"],
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "frames": info["frames"],
        "encoding": "PCM_16",
    }


def _write_manifest(path: Path, document: dict[str, Any]) -> None:
    document["complete"] = all(
        entry.get("status") in {"rendered", "reused"}
        for entry in document.get("segments", [])
    )
    seal_manifest(document)
    validate_manifest_document(document)
    atomic_json(path, document)


def _model_label(identity: dict[str, Any]) -> str:
    target_id = identity["target_id"]
    path = Path(target_id).expanduser()
    return path.name if path.is_absolute() else target_id


def _entry_for(
    segment: Any,
    role: str,
    identity: dict[str, Any],
    native: dict[str, Any],
    composition: dict[str, Any],
    key: str,
    semantics_hash: str,
) -> dict[str, Any]:
    relative_path = f"segments/{segment.source.locator.replace('.', '-')}-{key}.wav"
    return {
        "source": {
            "locator": segment.source.locator,
            "flow_index": segment.source.flow_index,
            "segment_index": segment.source.segment_index,
            "flow_hash": segment.source.flow_hash,
            "semantic_segment_hash": segment.semantic_segment_hash,
        },
        "logical_voice": role,
        "language": segment.language,
        "resolved_voice": {
            "engine": identity["engine"],
            "engine_api_version": identity["engine_api_version"],
            "engine_api_expected_version": identity["engine_api_expected_version"],
            "engine_api_contract": identity["engine_api_contract"],
            "engine_identity": {
                "schema": ENGINE_IDENTITY_SCHEMA,
                "sha256": "sha256:" + digest(identity["engine_identity"]),
            },
            "model": _model_label(identity),
            "model_asset_sha256": (
                "sha256:" + identity["model_asset_sha256"]
                if identity["model_asset_sha256"] is not None
                else None
            ),
            "voice": identity["voice"],
            "speaker": identity["speaker"],
            "model_revision": identity["model_revision"],
            "reference_audio_sha256": (
                "sha256:" + identity["reference_sha256"]
                if identity["reference_sha256"] is not None
                else None
            ),
            "effective_binding_sha256": _versioned_hash(identity),
        },
        "synthesis_semantics_sha256": semantics_hash,
        "acoustic_cache_key": "sha256:" + key,
        "semantic_lowering": {
            "composition_schema": NUMERIC_COMPOSITION_SCHEMA,
            "native_directives": _directive_items(native),
            "composition_directives": _directive_items(composition),
            "unsupported_directives": [],
        },
        "edges": {
            "policy": EDGE_POLICY,
            "source_frames": None,
            "trim_start_frames": 0,
            "trim_end_frames": 0,
            "leading_silence_frames": None,
            "trailing_silence_frames": None,
        },
        "timing": {
            "provenance": "absent",
            "unit": "frames",
            "sample_rate": None,
            "spans": [],
        },
        "status": "pending",
        "audio": None,
        "error": None,
        "_relative_path": relative_path,
    }


def _resolve_segment_selection(
    binding: Any,
    profile: VoiceProfile,
    language: str,
) -> Any:
    options = dict(profile.backend_defaults.get(binding.engine, {}))
    options.update(binding.options)
    if binding.reference_audio is not None:
        reference = Path(binding.reference_audio).expanduser().resolve()
        if not reference.is_file():
            raise ProfileError(f"Reference audio is missing: {reference}")
        options["voice_source"] = {
            "kind": "reference",
            "value": str(reference),
            "sha256": file_hash(reference),
        }
    request = EngineRequest(
        engine=binding.engine,
        target_id=binding.target_id,
        language=language,
        voice=binding.voice,
        speaker=binding.speaker,
        options=options,
    )
    selection, diagnostics = resolve_voice(request)
    errors = [
        diagnostic
        for diagnostic in diagnostics
        if getattr(diagnostic, "severity", None) == "error"
    ]
    if errors:
        raise EngineSelectionError(
            "; ".join(f"{item.code}: {item.message}" for item in errors),
            engine=binding.engine,
            target_id=selection.target_id,
            details={"diagnostics": [item.as_dict() for item in errors]},
        )
    if binding.model_revision is not None:
        selection = replace(
            selection,
            metadata={
                **dict(selection.metadata),
                "model_revision": binding.model_revision,
            },
        )
    return selection


def _speech_request(segment: Any, selection: Any) -> SpeechRequest:
    tokens = tuple(
        SpeechToken(
            start=int(token["start"]),
            end=int(token["end"]),
            text=segment.text[int(token["start"]) : int(token["end"])],
            pos=token.get("pos"),
            tag=token.get("tag"),
            lemma=token.get("lemma"),
            morph=token.get("morph"),
            language=segment.language,
        )
        for token in segment.tokens
    )
    pronunciation = segment.directives.get("pronunciation")
    overrides: tuple[PronunciationSpan, ...] = ()
    if isinstance(pronunciation, Mapping):
        overrides = (
            PronunciationSpan(
                start=0,
                end=len(segment.text),
                phonemes=pronunciation["phonemes"],
                language=segment.language,
                alphabet=pronunciation.get("alphabet", "ipa"),
            ),
        )
    return SpeechRequest(
        id=segment.source.locator,
        text=segment.text,
        language=segment.language,
        voice=selection.voice,
        speaker=selection.speaker,
        pronunciation_overrides=overrides,
        tokens=tokens,
        options=selection.options,
    )


def _capacity_context(request: SpeechRequest) -> CapacityContext:
    return CapacityContext(
        linguistic_token_ranges=tuple(
            (token.start, token.end) for token in request.tokens
        ),
        planned_semantics=True,
    )


def validate_utterplan(
    plan: UtterPlanV5 | str | Path,
    *,
    profile: VoiceProfile | str | Path,
) -> dict[str, Any]:
    """Run structural, profile, engine, semantic, and capacity preflight only."""
    parsed = load_utterplan_v5(plan) if isinstance(plan, (str, Path)) else plan
    loaded_profile = (
        VoiceProfile.load(profile) if isinstance(profile, (str, Path)) else profile
    )
    if not isinstance(parsed, UtterPlanV5):
        raise PlanError("plan must be an UtterPlanV5 value or path")
    if not isinstance(loaded_profile, VoiceProfile):
        raise ProfileError("profile must be a VoiceProfile value or path")
    semantics = _preflight_document_semantics(parsed, loaded_profile)
    selection = select_segments(parsed)
    unsupported = [
        segment.source.locator
        for segment in selection.segments
        if classify_source_operation(segment) is SourceOperationKind.UNSUPPORTED
    ]
    if unsupported:
        raise PlanError(
            "Unsupported/non-renderable UtterPlan source operation(s): "
            + ", ".join(unsupported)
        )

    engines: dict[str, dict[str, Any]] = {}
    for segment in selection.speech_segments:
        role = resolve_logical_voice(segment, loaded_profile)
        binding = loaded_profile.roles[role]
        adapter = get_adapter(binding.engine)
        probe = adapter.probe_api()
        if not probe.compatible:
            details: dict[str, object] = {"probe": asdict(probe)}
            if probe.status == "package_missing":
                raise EngineUnavailableError(
                    f"Engine {adapter.id!r} runtime package is unavailable",
                    engine=adapter.id,
                    native_error_type=probe.error_type,
                    details=details,
                )
            raise EngineApiIncompatibleError(
                f"Engine {adapter.id!r} public synthesis API is incompatible",
                engine=adapter.id,
                engine_version=probe.distribution_version,
                native_error_type=probe.error_type,
                details=details,
            )
        engine_selection = _resolve_segment_selection(
            binding, loaded_profile, segment.language
        )
        capabilities = adapter.capabilities()
        lower_directives(segment, capabilities)
        request = _speech_request(segment, engine_selection)
        adapter.canonical_synthesis_identity(engine_selection)
        engines[adapter.id] = {
            "package": adapter.package_name,
            "version": adapter.version() or probe.distribution_version,
            "api_version": probe.api_version,
            "capacity_measurement": capabilities.supports_request_measurement,
        }
        if not request.text.strip():
            raise PlanError(f"Speech operation {segment.source.locator} has empty text")

    return {
        "valid": True,
        "plan_id": parsed.plan_id,
        "source_file_sha256": "sha256:" + parsed.file_sha256,
        "selection": _selection_data(selection),
        "source_operations": [
            source_operation_record(segment) for segment in selection.segments
        ],
        "engines": [{"id": engine, **engines[engine]} for engine in sorted(engines)],
        "document_semantics": semantics,
    }


def render_utterplan_v5(
    plan: UtterPlanV5 | str | Path,
    *,
    profile: VoiceProfile | str | Path,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    selection: RenderSelection | None = None,
    take: int | None = None,
    select: Iterable[str] | str | None = None,
    resume: bool = True,
    force: bool = False,
    fail_fast: bool = True,
    on_event: RenderEventCallback | None = None,
) -> RenderResult:
    """Synthesize source-addressed UtterPlan v5 speech artifacts."""
    parsed = load_utterplan_v5(plan) if isinstance(plan, (str, Path)) else plan
    loaded_profile = (
        VoiceProfile.load(profile) if isinstance(profile, (str, Path)) else profile
    )
    if not isinstance(parsed, UtterPlanV5):
        raise PlanError("plan must be an UtterPlanV5 value or path")
    if not isinstance(loaded_profile, VoiceProfile):
        raise ProfileError("profile must be a VoiceProfile value or path")
    _emit_event(
        on_event,
        RenderEventKind.PREFLIGHT_STARTED,
        plan_id=parsed.plan_id,
    )
    document_semantics = _preflight_document_semantics(parsed, loaded_profile)
    provided_selection = selection
    if provided_selection is None:
        selection = select_segments(parsed, take=take, select=select)
    else:
        if take is not None or select is not None:
            raise PlanError("Use selection or take/select, not both")
        if not isinstance(provided_selection, RenderSelection):
            raise PlanError("selection must be a RenderSelection value")
        if provided_selection.kind == "all":
            expected_selection = select_segments(parsed)
        elif provided_selection.kind == "take":
            expected_selection = select_segments(parsed, take=provided_selection.take)
        elif provided_selection.kind == "locators":
            expected_selection = select_segments(
                parsed, select=provided_selection.locators
            )
        else:
            raise PlanError(
                f"Unknown render selection kind: {provided_selection.kind!r}"
            )
        if provided_selection != expected_selection:
            raise PlanError("selection does not match the supplied UtterPlan v5")
        selection = provided_selection
    unsupported = [
        segment.source.locator
        for segment in selection.segments
        if classify_source_operation(segment) is SourceOperationKind.UNSUPPORTED
    ]
    if unsupported:
        raise PlanError(
            "Unsupported/non-renderable UtterPlan source operation(s): "
            + ", ".join(unsupported)
        )
    out = Path(output_dir).expanduser().resolve()
    cache = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else out.parent / ".voicerender-cache"
    )
    if cache == out or cache.is_relative_to(out) or out.is_relative_to(cache):
        raise ArtifactError("cache_dir must be outside output_dir")

    prepared: list[dict[str, Any]] = []
    identities: dict[str, dict[str, Any]] = {}
    for segment in selection.speech_segments:
        role = resolve_logical_voice(segment, loaded_profile)
        binding = loaded_profile.roles[role]
        adapter = get_adapter(binding.engine)
        probe = adapter.probe_api()
        if not probe.compatible:
            probe_details: dict[str, object] = {"probe": asdict(probe)}
            if probe.status == "package_missing":
                raise EngineUnavailableError(
                    f"Engine {adapter.id!r} runtime package is unavailable",
                    engine=adapter.id,
                    native_error_type=probe.error_type,
                    details=probe_details,
                )
            raise EngineApiIncompatibleError(
                f"Engine {adapter.id!r} public synthesis API is incompatible",
                engine=adapter.id,
                engine_version=probe.distribution_version,
                native_error_type=probe.error_type,
                details=probe_details,
            )
        engine_selection = _resolve_segment_selection(
            binding, loaded_profile, segment.language
        )
        engine_version = adapter.version() or probe.distribution_version
        identity = _identity(
            binding,
            engine_version=engine_version,
            algorithm=V5_RENDER_ALGORITHM,
        )
        identity.update(
            {
                "engine": adapter.id,
                "engine_version": engine_version,
                "target_id": engine_selection.target_id,
                "model_revision": (
                    engine_selection.metadata.get("model_revision")
                    or engine_selection.metadata.get("source_revision")
                    or binding.model_revision
                ),
                "voice": engine_selection.voice,
                "speaker": engine_selection.speaker,
                "engine_api_version": probe.api_version,
                "engine_api_expected_version": probe.expected_api_version,
                "engine_api_contract": probe.contract_source,
                "engine_identity": dict(
                    adapter.canonical_synthesis_identity(engine_selection)
                ),
            }
        )
        identities.setdefault(role, identity)
        capabilities = adapter.capabilities()
        native, composition = lower_directives(segment, capabilities)
        speech_request = _speech_request(segment, engine_selection)
        request_semantics = {
            "tokens": [asdict(token) for token in speech_request.tokens],
            "pronunciation_overrides": [
                asdict(span) for span in speech_request.pronunciation_overrides
            ],
            "whole_request_phonemes": speech_request.whole_request_phonemes,
        }
        semantics_projection = {
            "schema": SYNTHESIS_SEMANTICS_SCHEMA,
            "text": segment.text,
            "language": segment.language,
            "native_directives": native,
            "composition_schema": NUMERIC_COMPOSITION_SCHEMA,
            "composition_directives": composition,
            "request": request_semantics,
        }
        semantics_hash = _versioned_hash(semantics_projection)
        acoustic_projection = {
            "schema": "voicerender.acoustic-request.v1",
            "text": speech_request.text,
            "language": speech_request.language,
            "native_directives": native,
            "request": request_semantics,
        }
        key = digest(
            {
                "schema": CACHE_KEY_SCHEMA,
                "binding": identity,
                "acoustic_request": acoustic_projection,
                "capacity_schema": CAPACITY_SCHEMA,
                "edge_policy": EDGE_POLICY,
                "output": {"format": "WAV", "encoding": "PCM_16"},
            }
        )
        prepared.append(
            {
                "segment": segment,
                "role": role,
                "binding": binding,
                "adapter": adapter,
                "selection": engine_selection,
                "speech_request": speech_request,
                "identity": identity,
                "native": native,
                "composition": composition,
                "key": key,
                "semantics_hash": semantics_hash,
            }
        )

    _emit_event(
        on_event,
        RenderEventKind.PREFLIGHT_FINISHED,
        plan_id=parsed.plan_id,
        selected_count=len(selection.speech_segments),
    )

    profile_identity = _versioned_hash(
        {role: identities[role] for role in sorted(identities)}
    )
    entries = [
        _entry_for(
            item["segment"],
            item["role"],
            item["identity"],
            item["native"],
            item["composition"],
            item["key"],
            item["semantics_hash"],
        )
        for item in prepared
    ]
    selection_data = _selection_data(selection)
    document: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "manifest_id": "sha256:" + "0" * 64,
        "manifest_id_algorithm": MANIFEST_ID_ALGORITHM,
        "source": {
            "kind": "utterplan-v5",
            "plan_id": parsed.plan_id,
            "plan_file_sha256": "sha256:" + parsed.file_sha256,
            "schema_version": 5,
        },
        "producer": {
            "name": "voicerender",
            "version": _VERSION,
            "algorithm": V5_RENDER_ALGORITHM,
        },
        "profile": {
            "name": loaded_profile.name,
            "effective_profile_sha256": profile_identity,
        },
        "selection": selection_data,
        "source_operations": [
            source_operation_record(segment) for segment in selection.segments
        ],
        "edge_policy": EDGE_POLICY,
        "complete": not prepared,
        "segments": [
            {key: value for key, value in entry.items() if key != "_relative_path"}
            for entry in entries
        ],
    }
    manifest_path = out / MANIFEST_FILENAME
    if document_semantics:
        document["document_semantics"] = document_semantics
    rendered = reused = failed = 0
    with _exclusive_run(out):
        if manifest_path.is_symlink():
            raise ArtifactError("Render manifest path must not be a symlink")
        prior: dict[tuple[str, str], dict[str, Any]] = {}
        if resume and not force and manifest_path.is_file():
            try:
                old_document = read_manifest(manifest_path)
                if old_document["source"] == document["source"]:
                    prior = {
                        (old["source"]["locator"], old["acoustic_cache_key"]): old
                        for old in old_document["segments"]
                        if old["status"] in {"rendered", "reused"}
                        and isinstance(old["audio"], dict)
                    }
            except ArtifactError:
                prior = {}
        _write_manifest(manifest_path, document)
        cache.mkdir(parents=True, exist_ok=True)
        with ExitStack() as opened:
            sessions: dict[str, Any] = {}
            for index, item in enumerate(prepared):
                entry = entries[index]
                segment = item["segment"]
                relative_path = entry.pop("_relative_path")
                entry["audio"] = None
                key = item["key"]
                cache_wav = cache / f"{key}.wav"
                _emit_event(
                    on_event,
                    RenderEventKind.SEGMENT_STARTED,
                    plan_id=parsed.plan_id,
                    locator=segment.source.locator,
                    role=item["role"],
                    engine=item["identity"]["engine"],
                    status="pending",
                )
                completion_event: RenderEventKind | None = None
                try:
                    target = _safe_target(out, relative_path)
                    _assert_reference_unchanged(item["binding"], item["identity"])
                    old = prior.get((segment.source.locator, "sha256:" + key))
                    if not force and old is not None:
                        old_path = resolve_bundle_artifact(
                            manifest_path, old["audio"]["path"]
                        )
                        old_info = {
                            key_name: old["audio"][key_name]
                            for key_name in (
                                "sha256",
                                "sample_rate",
                                "channels",
                                "frames",
                            )
                        }
                        if old_path != target:
                            _copy_verified(old_path, target, old_info)
                        entry.update(
                            status="reused",
                            audio=_audio_record(relative_path, old_info),
                            edges=old["edges"],
                            timing=old["timing"],
                        )
                        reused += 1
                        completion_event = RenderEventKind.SEGMENT_REUSED
                    else:
                        cached = None if force else _read_cache(cache, key)
                        if cached is not None:
                            source, sidecar = cached
                            _copy_verified(source, target, sidecar["wav"])
                            info = sidecar["wav"]
                            timing = sidecar["timing"]
                            entry.update(
                                status="reused",
                                audio=_audio_record(relative_path, info),
                                edges={
                                    **entry["edges"],
                                    "source_frames": info["frames"],
                                },
                                timing=timing,
                            )
                            reused += 1
                            completion_event = RenderEventKind.SEGMENT_REUSED
                        else:
                            identity_key = digest(item["identity"])
                            if identity_key not in sessions:
                                _assert_reference_unchanged(
                                    item["binding"], item["identity"]
                                )
                                _emit_event(
                                    on_event,
                                    RenderEventKind.ENGINE_OPEN_STARTED,
                                    plan_id=parsed.plan_id,
                                    role=item["role"],
                                    engine=item["identity"]["engine"],
                                )
                                sessions[identity_key] = opened.enter_context(
                                    open_session(item["selection"])
                                )
                                _emit_event(
                                    on_event,
                                    RenderEventKind.ENGINE_OPEN_FINISHED,
                                    plan_id=parsed.plan_id,
                                    role=item["role"],
                                    engine=item["identity"]["engine"],
                                )
                            request = item["speech_request"]
                            session = sessions[identity_key]
                            result = synthesize(
                                session,
                                request,
                                capacity=_capacity_context(request),
                            )
                            info = atomic_wav(
                                cache_wav, result.audio, result.sample_rate
                            )
                            spans = _timing_spans(
                                result.word_timings,
                                sample_rate=info["sample_rate"],
                                frames=info["frames"],
                            )
                            timing = {
                                "provenance": "provided" if spans else "absent",
                                "unit": "frames",
                                "sample_rate": info["sample_rate"],
                                "spans": spans,
                            }
                            sidecar = {
                                "key": key,
                                "wav": info,
                                "encoding": "PCM_16",
                                "timing": timing,
                            }
                            atomic_json(cache / f"{key}.json", sidecar)
                            _copy_verified(cache_wav, target, info)
                            entry.update(
                                status="rendered",
                                audio=_audio_record(relative_path, info),
                                edges={
                                    **entry["edges"],
                                    "source_frames": info["frames"],
                                },
                                timing=timing,
                            )
                            rendered += 1
                            completion_event = RenderEventKind.SEGMENT_RENDERED
                except RenderEventCallbackError:
                    raise
                except Exception as exc:
                    entry.update(
                        status="failed",
                        audio=None,
                        error={"type": type(exc).__name__, "message": str(exc)},
                    )
                    failed += 1
                    document["segments"][index] = {
                        key_name: value
                        for key_name, value in entry.items()
                        if key_name != "_relative_path"
                    }
                    _write_manifest(manifest_path, document)
                    _emit_event(
                        on_event,
                        RenderEventKind.SEGMENT_FAILED,
                        plan_id=parsed.plan_id,
                        locator=segment.source.locator,
                        role=item["role"],
                        engine=item["identity"]["engine"],
                        status="failed",
                        error_type=type(exc).__name__,
                        error_code=getattr(exc, "code", None),
                    )
                    if fail_fast:
                        raise
                    continue
                document["segments"][index] = {
                    key_name: value
                    for key_name, value in entry.items()
                    if key_name != "_relative_path"
                }
                _write_manifest(manifest_path, document)
                if completion_event is not None:
                    _emit_event(
                        on_event,
                        completion_event,
                        plan_id=parsed.plan_id,
                        locator=segment.source.locator,
                        role=item["role"],
                        engine=item["identity"]["engine"],
                        status=(
                            "reused"
                            if completion_event is RenderEventKind.SEGMENT_REUSED
                            else "rendered"
                        ),
                        rendered_count=rendered,
                        reused_count=reused,
                    )
        verification_failure: ArtifactError | None = None
        for entry in document["segments"]:
            if entry["status"] not in {"rendered", "reused"}:
                continue
            try:
                verify_manifest_audio(manifest_path, {"segments": [entry]})
            except ArtifactError as exc:
                entry.update(
                    status="failed",
                    audio=None,
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                failed += 1
                verification_failure = exc
                _emit_event(
                    on_event,
                    RenderEventKind.SEGMENT_FAILED,
                    plan_id=parsed.plan_id,
                    locator=entry["source"]["locator"],
                    role=entry["logical_voice"],
                    engine=entry["resolved_voice"]["engine"],
                    status="failed",
                    error_type=type(exc).__name__,
                    error_code=getattr(exc, "code", None),
                )
        document["complete"] = failed == 0
        _write_manifest(manifest_path, document)
        _emit_event(
            on_event,
            RenderEventKind.RENDER_COMPLETED,
            plan_id=parsed.plan_id,
            status="completed" if failed == 0 else "failed",
            rendered_count=rendered,
            reused_count=reused,
            failed_count=failed,
            selected_count=len(selection.speech_segments),
        )
        if verification_failure is not None and fail_fast:
            raise verification_failure
    return RenderResult(
        manifest_path=manifest_path,
        rendered_count=rendered,
        reused_count=reused,
        failed_count=failed,
        selected_count=len(selection.speech_segments),
    )


def render_utterplan(
    plan: UtterPlanV5 | str | Path,
    *,
    profile: VoiceProfile | str | Path,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    selection: RenderSelection | None = None,
    resume: bool = True,
    force: bool = False,
    fail_fast: bool = True,
    on_event: RenderEventCallback | None = None,
) -> RenderResult:
    """Render a selected UtterPlan v5 using the frozen standalone API."""
    return render_utterplan_v5(
        plan,
        profile=profile,
        output_dir=output_dir,
        cache_dir=cache_dir,
        selection=selection,
        resume=resume,
        force=force,
        fail_fast=fail_fast,
        on_event=on_event,
    )
