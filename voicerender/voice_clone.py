"""Reusable reference-voice cloning through normal RenderProfile bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from renderplan import RenderProfile

from .engines.registry import discover_targets, get_adapter
from .errors import EngineSelectionError, ProfileError
from .renderprofile import create_render_profile
from .voice_source_codec import load_voice_source
from .voice_sources import (
    VoiceSource,
    create_voice_source,
    normalize_voice_source,
    validate_voice_source,
)


@dataclass(frozen=True, slots=True)
class VoiceCloneRequest:
    engine: str
    target_id: str
    language: str
    reference_wav: Path | None = None
    voice_source: Path | Mapping[str, Any] | None = None
    voicefolio_ref: str | None = None
    transcript: str | None = None
    speaker: str | int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    offline: bool = False
    refresh: bool = False


def _source_for_request(request: VoiceCloneRequest) -> VoiceSource:
    selected = sum(
        value is not None
        for value in (
            request.reference_wav,
            request.voice_source,
            request.voicefolio_ref,
        )
    )
    if selected != 1:
        raise ProfileError(
            "Exactly one of reference_wav, voice_source, or voicefolio_ref must be provided"
        )
    if request.reference_wav is not None:
        return create_voice_source(
            reference_wav=request.reference_wav,
            transcript=request.transcript,
            language=request.language,
        )
    if request.voicefolio_ref is not None:
        return create_voice_source(
            voicefolio_ref=request.voicefolio_ref,
            transcript=request.transcript,
            language=request.language,
        )
    source_value = request.voice_source
    if isinstance(source_value, (Path, str)):
        source_value = load_voice_source(source_value)
    if isinstance(source_value, VoiceSource):
        source_value = source_value.as_mapping()
    if not isinstance(source_value, Mapping):
        raise ProfileError("voice_source must be a source mapping or sidecar path")
    source = normalize_voice_source(source_value, allow_named=False)
    if request.transcript is not None:
        metadata = dict(source.metadata)
        metadata["transcript"] = request.transcript
        source = normalize_voice_source(
            {**source.as_mapping(), "metadata": metadata}, allow_named=False
        )
    return source


def clone_voice_profile(
    request: VoiceCloneRequest,
    *,
    role: str = "narrator",
) -> RenderProfile:
    """Validate a reference binding and return a reusable RenderProfile."""
    if (
        not request.engine.strip()
        or not request.target_id.strip()
        or not request.language.strip()
    ):
        raise ProfileError(
            "Voice clone requires non-empty engine, target_id, and language"
        )
    source = _source_for_request(request)
    if source.kind not in {"reference", "managed_reference"}:
        raise ProfileError(
            "Generic voice clone requires a reference or managed-reference source"
        )
    diagnostics = validate_voice_source(source, offline=request.offline)
    errors = [item for item in diagnostics if item.severity == "error"]
    if errors:
        raise ProfileError(errors[0].message)

    targets = discover_targets(
        request.engine,
        language=request.language,
        offline=request.offline,
        refresh=request.refresh,
    )
    target = next((item for item in targets if item.id == request.target_id), None)
    if target is None:
        raise ProfileError(
            f"Engine {request.engine!r} does not expose target {request.target_id!r}"
        )
    adapter = get_adapter(request.engine)
    engine_supports_reference = adapter.capabilities().supports_reference_voice
    target_supports_reference = "reference_voice" in target.capabilities
    if not engine_supports_reference and not target_supports_reference:
        raise EngineSelectionError(
            f"Engine {adapter.id!r} target {request.target_id!r} does not support reference voices",
            engine=adapter.id,
            target_id=request.target_id,
            code="voice_clone.reference_unsupported",
        )

    options = dict(request.options)
    if "voice_source" in options:
        raise ProfileError(
            "voice_source is selected by the clone request and cannot be overridden"
        )
    source_transcript = source.metadata.get("transcript")
    if source_transcript is None and source.origin is not None:
        source_transcript = source.origin.get("reference_text")
    transcript = (
        request.transcript if request.transcript is not None else source_transcript
    )
    accepted_options = set(adapter.capabilities().option_names) | set(
        adapter.capabilities().options
    )
    transcript_option = next(
        (name for name in ("reference_text", "transcript") if name in accepted_options),
        None,
    )
    if transcript is not None and transcript_option is not None:
        existing = options.get(transcript_option)
        if existing is not None and existing != transcript:
            raise ProfileError(
                f"Conflicting transcript supplied in options.{transcript_option} and clone request"
            )
        options[transcript_option] = transcript

    return create_render_profile(
        engine=request.engine,
        target_id=request.target_id,
        language=request.language,
        role=role,
        speaker=request.speaker,
        voice_source=source.as_mapping(),
        options=options,
        offline=request.offline,
        refresh=request.refresh,
    )


__all__ = ["VoiceCloneRequest", "clone_voice_profile"]
