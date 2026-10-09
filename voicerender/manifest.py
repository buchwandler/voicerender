"""Strict v2 render-manifest identities, validation, and safe artifact inspection."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

import soundfile as sf

from .errors import ArtifactError
from .storage import digest, wav_info

MANIFEST_SCHEMA = "voicerender.render-manifest.v2"
MANIFEST_FILENAME = "render-manifest.json"
MANIFEST_ID_ALGORITHM = "voicerender-manifest-id.v1"
EDGE_POLICY = "preserved-unmeasured-v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL = {
    "schema",
    "manifest_id",
    "manifest_id_algorithm",
    "source",
    "producer",
    "profile",
    "selection",
    "edge_policy",
    "complete",
    "segments",
}
_SEGMENT_FIELDS = {
    "source",
    "logical_voice",
    "language",
    "resolved_voice",
    "synthesis_semantics_sha256",
    "acoustic_cache_key",
    "semantic_lowering",
    "edges",
    "timing",
    "status",
    "audio",
    "error",
}


def manifest_id(document: dict[str, Any]) -> str:
    """Return identity for stable semantics and artifacts, excluding progress state."""
    stable = json.loads(json.dumps(document, ensure_ascii=False, allow_nan=False))
    stable.pop("manifest_id", None)
    for entry in stable.get("segments", []):
        if isinstance(entry, dict):
            entry.pop("status", None)
            entry.pop("error", None)
    return "sha256:" + digest(stable)


def seal_manifest(document: dict[str, Any]) -> dict[str, Any]:
    document["manifest_id_algorithm"] = MANIFEST_ID_ALGORITHM
    document["manifest_id"] = manifest_id(document)
    return document


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"Manifest {where} must be an object")
    return value


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"Manifest {where} must be a non-empty string")
    return value


def _sha(value: Any, where: str, *, prefixed: bool = True) -> str:
    pattern = _SHA256 if prefixed else _HEX_SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ArtifactError(f"Manifest {where} must be a lowercase SHA-256 identity")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactError(f"Manifest {where} must be an integer >= {minimum}")
    return value


def _keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    unknown = set(value).difference(expected)
    missing = expected.difference(value)
    if missing or unknown:
        raise ArtifactError(
            f"Manifest {where} fields invalid; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def validate_manifest_document(
    value: Any, *, verify_identity: bool = True
) -> dict[str, Any]:
    """Validate the closed v2 shape and all cross-field invariants."""
    doc = _mapping(value, "root")
    _keys(doc, _TOP_LEVEL, "root")
    if doc["schema"] != MANIFEST_SCHEMA:
        raise ArtifactError(f"Unsupported render-manifest schema: {doc['schema']!r}")
    if doc["manifest_id_algorithm"] != MANIFEST_ID_ALGORITHM:
        raise ArtifactError("Unsupported manifest identity algorithm")
    _sha(doc["manifest_id"], "manifest_id")

    source = _mapping(doc["source"], "source")
    _keys(source, {"kind", "plan_id", "plan_file_sha256", "schema_version"}, "source")
    if source["kind"] != "utterplan-v5" or source["schema_version"] != 5:
        raise ArtifactError("Manifest source must identify an UtterPlan v5 document")
    _sha(source["plan_id"], "source.plan_id")
    _sha(source["plan_file_sha256"], "source.plan_file_sha256")

    producer = _mapping(doc["producer"], "producer")
    _keys(producer, {"name", "version", "algorithm"}, "producer")
    for key in ("name", "version", "algorithm"):
        _nonempty(producer[key], f"producer.{key}")

    profile = _mapping(doc["profile"], "profile")
    _keys(profile, {"name", "effective_profile_sha256"}, "profile")
    _nonempty(profile["name"], "profile.name")
    _sha(profile["effective_profile_sha256"], "profile.effective_profile_sha256")

    selection = _mapping(doc["selection"], "selection")
    required_selection = {"kind", "order", "selected_count", "is_full_plan"}
    kind = selection.get("kind")
    if kind == "take":
        required_selection.add("take")
    elif kind == "locators":
        required_selection.add("locators")
    _keys(selection, required_selection, "selection")
    if kind not in {"all", "take", "locators"} or selection["order"] != "source":
        raise ArtifactError("Manifest selection kind/order is invalid")
    count = _integer(selection["selected_count"], "selection.selected_count", minimum=1)
    if not isinstance(selection["is_full_plan"], bool):
        raise ArtifactError("Manifest selection.is_full_plan must be boolean")
    if kind == "take":
        _integer(selection["take"], "selection.take", minimum=1)
    if kind == "locators":
        locators = selection["locators"]
        if not isinstance(locators, list) or len(locators) != count:
            raise ArtifactError("Manifest selection.locators must match selected_count")
        for locator in locators:
            _nonempty(locator, "selection.locators[]")
        if len(set(locators)) != len(locators):
            raise ArtifactError("Manifest selection.locators must be unique")

    if doc["edge_policy"] != EDGE_POLICY:
        raise ArtifactError("Manifest edge policy is unsupported")
    if not isinstance(doc["complete"], bool):
        raise ArtifactError("Manifest complete must be boolean")
    entries = doc["segments"]
    if not isinstance(entries, list) or len(entries) != count:
        raise ArtifactError(
            "Manifest segments length must match selection.selected_count"
        )

    seen_locators: set[str] = set()
    seen_paths: set[str] = set()
    all_rendered = True
    for index, raw_entry in enumerate(entries):
        where = f"segments[{index}]"
        entry = _mapping(raw_entry, where)
        _keys(entry, _SEGMENT_FIELDS, where)
        source_ref = _mapping(entry["source"], f"{where}.source")
        _keys(
            source_ref,
            {
                "locator",
                "flow_index",
                "segment_index",
                "flow_hash",
                "semantic_segment_hash",
            },
            f"{where}.source",
        )
        locator = _nonempty(source_ref["locator"], f"{where}.source.locator")
        flow_index = _integer(source_ref["flow_index"], f"{where}.source.flow_index")
        segment_index = _integer(
            source_ref["segment_index"], f"{where}.source.segment_index"
        )
        if locator != f"f{flow_index:06d}.s{segment_index:03d}":
            raise ArtifactError(f"Manifest {where} source locator is not canonical")
        if locator in seen_locators:
            raise ArtifactError(f"Manifest contains duplicate source locator {locator}")
        seen_locators.add(locator)
        _sha(source_ref["flow_hash"], f"{where}.source.flow_hash")
        _sha(
            source_ref["semantic_segment_hash"], f"{where}.source.semantic_segment_hash"
        )
        _nonempty(entry["logical_voice"], f"{where}.logical_voice")
        _nonempty(entry["language"], f"{where}.language")

        voice = _mapping(entry["resolved_voice"], f"{where}.resolved_voice")
        _keys(
            voice,
            {
                "engine",
                "model",
                "model_asset_sha256",
                "voice",
                "speaker",
                "model_revision",
                "reference_audio_sha256",
                "effective_binding_sha256",
            },
            f"{where}.resolved_voice",
        )
        _nonempty(voice["engine"], f"{where}.resolved_voice.engine")
        _nonempty(voice["model"], f"{where}.resolved_voice.model")
        for nullable_string in ("voice", "model_revision"):
            if voice[nullable_string] is not None:
                _nonempty(
                    voice[nullable_string], f"{where}.resolved_voice.{nullable_string}"
                )
        if voice["speaker"] is not None:
            _integer(voice["speaker"], f"{where}.resolved_voice.speaker")
        if voice["reference_audio_sha256"] is not None:
            _sha(
                voice["reference_audio_sha256"],
                f"{where}.resolved_voice.reference_audio_sha256",
            )
        if voice["model_asset_sha256"] is not None:
            _sha(
                voice["model_asset_sha256"],
                f"{where}.resolved_voice.model_asset_sha256",
            )
        _sha(
            voice["effective_binding_sha256"],
            f"{where}.resolved_voice.effective_binding_sha256",
        )
        _sha(entry["synthesis_semantics_sha256"], f"{where}.synthesis_semantics_sha256")
        _sha(entry["acoustic_cache_key"], f"{where}.acoustic_cache_key")

        lowering = _mapping(entry["semantic_lowering"], f"{where}.semantic_lowering")
        _keys(
            lowering,
            {"native_directives", "composition_directives", "unsupported_directives"},
            f"{where}.semantic_lowering",
        )
        for field in (
            "native_directives",
            "composition_directives",
            "unsupported_directives",
        ):
            values = lowering[field]
            if not isinstance(values, list):
                raise ArtifactError(
                    f"Manifest {where}.semantic_lowering.{field} must be a list"
                )
            if field == "unsupported_directives" and values:
                raise ArtifactError(f"Manifest {where} contains unsupported directives")
            if field != "unsupported_directives":
                for item in values:
                    item = _mapping(item, f"{where}.semantic_lowering.{field}[]")
                    _keys(
                        item, {"name", "value"}, f"{where}.semantic_lowering.{field}[]"
                    )
                    _nonempty(item["name"], f"{where}.semantic_lowering.{field}[].name")

        edges = _mapping(entry["edges"], f"{where}.edges")
        _keys(
            edges,
            {
                "policy",
                "source_frames",
                "trim_start_frames",
                "trim_end_frames",
                "leading_silence_frames",
                "trailing_silence_frames",
            },
            f"{where}.edges",
        )
        if edges["policy"] != EDGE_POLICY:
            raise ArtifactError(f"Manifest {where}.edges.policy is unsupported")
        source_frames = edges["source_frames"]
        if source_frames is not None:
            source_frames = _integer(
                source_frames, f"{where}.edges.source_frames", minimum=1
            )
        trim_start = _integer(
            edges["trim_start_frames"], f"{where}.edges.trim_start_frames"
        )
        trim_end = _integer(edges["trim_end_frames"], f"{where}.edges.trim_end_frames")
        if source_frames is not None and (trim_start + trim_end > source_frames):
            raise ArtifactError(f"Manifest {where} edge trim exceeds source audio")
        for key in ("leading_silence_frames", "trailing_silence_frames"):
            if edges[key] is not None:
                _integer(edges[key], f"{where}.edges.{key}")
        if trim_start or trim_end:
            raise ArtifactError(
                f"Manifest {where} trims audio under a preserved-unmeasured policy"
            )
        if (
            edges["leading_silence_frames"] is not None
            or edges["trailing_silence_frames"] is not None
        ):
            raise ArtifactError(f"Manifest {where} cannot claim measured edge silence")

        timing = _mapping(entry["timing"], f"{where}.timing")
        _keys(timing, {"provenance", "unit", "sample_rate", "spans"}, f"{where}.timing")
        if (
            timing["provenance"] not in {"absent", "provided"}
            or timing["unit"] != "frames"
        ):
            raise ArtifactError(f"Manifest {where}.timing provenance is invalid")
        sample_rate = timing["sample_rate"]
        if sample_rate is not None:
            _integer(sample_rate, f"{where}.timing.sample_rate", minimum=1)
        spans = timing["spans"]
        if not isinstance(spans, list):
            raise ArtifactError(f"Manifest {where}.timing.spans must be a list")
        if timing["provenance"] == "absent" and spans:
            raise ArtifactError(f"Manifest {where} absent timing cannot contain spans")
        if timing["provenance"] == "provided" and sample_rate is None:
            raise ArtifactError(
                f"Manifest {where} provided timing requires sample_rate"
            )
        previous_end = 0
        for span_index, raw_span in enumerate(spans):
            span = _mapping(raw_span, f"{where}.timing.spans[{span_index}]")
            if set(span).difference(
                {"kind", "start_frame", "end_frame", "text_start", "text_end", "text"}
            ) or not {"start_frame", "end_frame"}.issubset(span):
                raise ArtifactError(f"Manifest {where} timing span fields are invalid")
            start = _integer(span["start_frame"], f"{where}.timing.span.start_frame")
            end = _integer(span["end_frame"], f"{where}.timing.span.end_frame")
            if start >= end or start < previous_end:
                raise ArtifactError(
                    f"Manifest {where} timing spans must be ordered and non-empty"
                )
            if source_frames is not None and end > source_frames:
                raise ArtifactError(
                    f"Manifest {where} timing span exceeds audio frames"
                )
            if "text" in span and not isinstance(span["text"], str):
                raise ArtifactError(
                    f"Manifest {where} timing span text must be a string"
                )
            if "kind" in span:
                _nonempty(span["kind"], f"{where}.timing.span.kind")
            if "text_start" in span or "text_end" in span:
                if "text_start" not in span or "text_end" not in span:
                    raise ArtifactError(
                        f"Manifest {where} timing span must provide both text offsets"
                    )
                text_start = _integer(
                    span["text_start"], f"{where}.timing.span.text_start"
                )
                text_end = _integer(span["text_end"], f"{where}.timing.span.text_end")
                if text_end < text_start:
                    raise ArtifactError(
                        f"Manifest {where} timing span text offsets are reversed"
                    )
            previous_end = end

        status = entry["status"]
        if status not in {"pending", "rendered", "reused", "failed"}:
            raise ArtifactError(f"Manifest {where}.status is invalid")
        audio = entry["audio"]
        error = entry["error"]
        if status in {"rendered", "reused"}:
            all_rendered &= True
            audio = _mapping(audio, f"{where}.audio")
            _keys(
                audio,
                {"path", "sha256", "sample_rate", "channels", "frames", "encoding"},
                f"{where}.audio",
            )
            relpath = _safe_relative_path(audio["path"], f"{where}.audio.path")
            if relpath in seen_paths:
                raise ArtifactError(f"Manifest contains duplicate audio path {relpath}")
            seen_paths.add(relpath)
            _sha(audio["sha256"], f"{where}.audio.sha256", prefixed=False)
            _integer(audio["sample_rate"], f"{where}.audio.sample_rate", minimum=1)
            channels = _integer(audio["channels"], f"{where}.audio.channels", minimum=1)
            if channels != 1:
                raise ArtifactError(f"Manifest {where}.audio must be mono")
            _integer(audio["frames"], f"{where}.audio.frames", minimum=1)
            if sample_rate is not None and sample_rate != audio["sample_rate"]:
                raise ArtifactError(
                    f"Manifest {where} timing sample_rate must match its WAV"
                )
            if audio["encoding"] != "PCM_16":
                raise ArtifactError(f"Manifest {where}.audio.encoding must be PCM_16")
            if source_frames is None or source_frames != audio["frames"]:
                raise ArtifactError(
                    f"Manifest {where} edge frame count must match its WAV"
                )
            if error is not None:
                raise ArtifactError(
                    f"Manifest {where} rendered entry cannot contain an error"
                )
        elif status == "failed":
            all_rendered = False
            if audio is not None:
                raise ArtifactError(
                    f"Manifest {where} failed entry cannot contain audio"
                )
            error = _mapping(error, f"{where}.error")
            _keys(error, {"type", "message"}, f"{where}.error")
            _nonempty(error["type"], f"{where}.error.type")
            _nonempty(error["message"], f"{where}.error.message")
        else:
            all_rendered = False
            if source_frames is not None or sample_rate is not None or spans:
                raise ArtifactError(
                    f"Manifest {where} pending entry has unexpected media metadata"
                )
            if audio is not None or error is not None:
                raise ArtifactError(
                    f"Manifest {where} pending entry cannot contain audio or error"
                )
    if doc["complete"] != all_rendered:
        raise ArtifactError("Manifest complete must agree with final segment statuses")
    if kind == "locators" and selection["locators"] != [
        entry["source"]["locator"] for entry in entries
    ]:
        raise ArtifactError(
            "Manifest selection.locators do not match source-ordered segments"
        )
    if verify_identity and manifest_id(doc) != doc["manifest_id"]:
        raise ArtifactError("Manifest identity mismatch")
    return doc


def _safe_relative_path(value: Any, where: str) -> str:
    path = _nonempty(value, where)
    if "\\" in path or "\x00" in path:
        raise ArtifactError(f"Manifest {where} must use a safe relative POSIX path")
    pure = PurePosixPath(path)
    if (
        not pure.parts
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ArtifactError(f"Manifest {where} must not escape the bundle root")
    if pure.as_posix() != path:
        raise ArtifactError(f"Manifest {where} is not canonical")
    return path


def resolve_bundle_artifact(manifest_path: str | Path, relative_path: str) -> Path:
    """Resolve a bundle-relative file while rejecting symlinks and escapes."""
    rel = _safe_relative_path(relative_path, "audio.path")
    manifest = Path(manifest_path)
    if manifest.is_symlink():
        raise ArtifactError("Render manifest itself must not be a symlink")
    root = manifest.parent.resolve(strict=True)
    candidate = root
    for component in PurePosixPath(rel).parts:
        candidate = candidate / component
        try:
            candidate.lstat()
        except OSError as exc:
            raise ArtifactError(f"Missing bundle artifact {rel}: {exc}") from exc
        if os.path.islink(candidate):
            raise ArtifactError(f"Bundle artifact path contains a symlink: {rel}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ArtifactError(
            f"Bundle artifact is not a regular file inside the bundle: {rel}"
        )
    return resolved


def verify_manifest_audio(manifest_path: str | Path, document: dict[str, Any]) -> None:
    """Verify every rendered WAV's containment, PCM encoding, metadata and bytes."""
    for index, entry in enumerate(document["segments"]):
        if entry["status"] not in {"rendered", "reused"}:
            continue
        audio = entry["audio"]
        path = resolve_bundle_artifact(manifest_path, audio["path"])
        try:
            info = wav_info(path)
            with sf.SoundFile(path) as stream:
                if stream.format != "WAV" or stream.subtype != "PCM_16":
                    raise ArtifactError(
                        f"Manifest segment {index} WAV encoding is not PCM_16"
                    )
        except (OSError, RuntimeError) as exc:
            raise ArtifactError(
                f"Manifest segment {index} WAV is unavailable: {exc}"
            ) from exc
        for key in ("sha256", "sample_rate", "channels", "frames"):
            if info[key] != audio[key]:
                raise ArtifactError(f"Manifest segment {index} WAV {key} mismatch")


def read_manifest(path: str | Path, *, verify_audio: bool = True) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.is_symlink():
        raise ArtifactError("Render manifest itself must not be a symlink")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"Cannot read render manifest {manifest_path}: {exc}"
        ) from exc
    doc = validate_manifest_document(value)
    if verify_audio:
        verify_manifest_audio(manifest_path, doc)
    return doc


def validate_manifest_source(document: dict[str, Any], plan: Any) -> None:
    """Reject a stale manifest join before a consumer materializes its audio."""
    source = document["source"]
    if source["plan_id"] != plan.plan_id:
        raise ArtifactError("Manifest plan_id does not match the supplied UtterPlan")
    if source["plan_file_sha256"] != "sha256:" + plan.file_sha256:
        raise ArtifactError(
            "Manifest exact source-file hash does not match the supplied plan"
        )
