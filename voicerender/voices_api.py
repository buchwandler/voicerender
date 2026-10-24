"""Normalized voice discovery and canonical selector handling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .engines.registry import discover_targets, list_engines, normalize_engine_id
from .errors import ProfileError
from .jsonutil import json_value

VoiceKind = Literal["builtin", "speaker", "managed-reference", "reference-capable"]


@dataclass(frozen=True, slots=True)
class VoiceDescriptor:
    selector: str
    engine: str
    target_id: str
    voice: str | None
    speaker: str | int | None
    kind: VoiceKind
    languages: tuple[str, ...]
    gender: str = "unknown"
    locale: str = "unknown"
    installed: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedVoiceSelector:
    engine: str
    target_id: str
    voice: str | None = None
    speaker: str | int | None = None


def parse_voice_selector(selector: str) -> ParsedVoiceSelector:
    """Parse the canonical ENGINE:TARGET[/VOICE][#SPEAKER] grammar."""
    if not isinstance(selector, str) or selector.count(":") != 1:
        raise ProfileError("Voice selector must use ENGINE:TARGET[/VOICE][#SPEAKER]")
    engine, value = selector.split(":", 1)
    if not engine or not value:
        raise ProfileError("Voice selector engine and target must be non-empty")
    speaker: str | int | None
    if "#" in value:
        if value.count("#") != 1:
            raise ProfileError(
                "Voice selector contains more than one speaker delimiter"
            )
        value, raw_speaker = value.split("#", 1)
        if not raw_speaker:
            raise ProfileError("Voice selector speaker must be non-empty")
        try:
            numeric_speaker = int(raw_speaker)
        except ValueError:
            speaker = raw_speaker
        else:
            speaker = (
                numeric_speaker if str(numeric_speaker) == raw_speaker else raw_speaker
            )
    else:
        speaker = None
    if "/" in value:
        if value.count("/") != 1:
            raise ProfileError("Voice selector contains more than one voice delimiter")
        target_id, voice = value.split("/", 1)
        if not voice:
            raise ProfileError("Voice selector voice must be non-empty")
    else:
        target_id, voice = value, None
    if not target_id:
        raise ProfileError("Voice selector target must be non-empty")
    return ParsedVoiceSelector(normalize_engine_id(engine), target_id, voice, speaker)


def _selector(
    engine: str, target_id: str, voice: str | None, speaker: str | int | None
) -> str:
    if any(
        delimiter in engine or delimiter in target_id for delimiter in (":", "/", "#")
    ):
        raise ProfileError(
            "Engine and target identifiers cannot contain selector delimiters"
        )
    result = f"{engine}:{target_id}"
    if voice is not None:
        if not voice or any(delimiter in voice for delimiter in ("/", "#")):
            raise ProfileError(
                "Voice identifiers must be non-empty and cannot contain / or #"
            )
        result += f"/{voice}"
    if speaker is not None:
        value = str(speaker)
        if not value or "#" in value:
            raise ProfileError(
                "Speaker identifiers must be non-empty and cannot contain #"
            )
        result += f"#{value}"
    return result


def _descriptor(
    *,
    target: Any,
    voice: str | None = None,
    speaker: str | int | None = None,
    kind: VoiceKind,
    languages: tuple[str, ...] | None = None,
    gender: str = "unknown",
    locale: str = "unknown",
    metadata: Mapping[str, Any] | None = None,
) -> VoiceDescriptor:
    return VoiceDescriptor(
        selector=_selector(target.engine, target.id, voice, speaker),
        engine=target.engine,
        target_id=target.id,
        voice=voice,
        speaker=speaker,
        kind=kind,
        languages=tuple(languages if languages is not None else target.languages),
        gender=gender,
        locale=locale,
        installed=bool(target.runtime_available),
        metadata={"target": dict(target.metadata), **dict(metadata or {})},
    )


def _language_matches(descriptor: VoiceDescriptor, language: str | None) -> bool:
    if language is None:
        return True
    expected = language.casefold()
    return any(item.casefold() == expected for item in descriptor.languages)


def _target_descriptors(target: Any) -> list[VoiceDescriptor]:
    descriptors: list[VoiceDescriptor] = []
    details = {detail.id: detail for detail in target.voice_details}
    voice_ids = tuple(
        dict.fromkeys(
            (*target.voices, *((target.default_voice,) if target.default_voice else ()))
        )
    )
    voice_kinds = target.metadata.get("voice_kinds", {})
    if not isinstance(voice_kinds, Mapping):
        voice_kinds = {}
    voices: list[tuple[str, Any]] = []
    for voice in voice_ids:
        detail = details.get(voice)
        kind: VoiceKind = (
            "managed-reference"
            if voice_kinds.get(voice) == "managed-reference"
            else "builtin"
        )
        descriptors.append(
            _descriptor(
                target=target,
                voice=voice,
                kind=kind,
                languages=tuple(detail.languages)
                if detail and detail.languages
                else None,
                gender=detail.gender if detail else "unknown",
                locale=detail.locale if detail else "unknown",
                metadata={
                    "voice": json_value(detail, label=f"voice {voice!r} metadata")
                }
                if detail
                else {},
            )
        )
        voices.append((voice, detail))
    speakers = tuple(target.speakers)
    for speaker_value in speakers:
        try:
            numeric_speaker = int(speaker_value)
        except (TypeError, ValueError):
            speaker: str | int = str(speaker_value)
        else:
            speaker = (
                numeric_speaker
                if str(numeric_speaker) == str(speaker_value)
                else str(speaker_value)
            )
        descriptors.append(_descriptor(target=target, speaker=speaker, kind="speaker"))
    if voices and speakers:
        for voice, detail in voices:
            for speaker_value in speakers:
                try:
                    numeric_speaker = int(speaker_value)
                except (TypeError, ValueError):
                    speaker = str(speaker_value)
                else:
                    speaker = (
                        numeric_speaker
                        if str(numeric_speaker) == str(speaker_value)
                        else str(speaker_value)
                    )
                descriptors.append(
                    _descriptor(
                        target=target,
                        voice=voice,
                        speaker=speaker,
                        kind="speaker",
                        languages=tuple(detail.languages)
                        if detail and detail.languages
                        else None,
                        gender=detail.gender if detail else "unknown",
                        locale=detail.locale if detail else "unknown",
                        metadata={
                            "voice": json_value(
                                detail, label=f"voice {voice!r} metadata"
                            )
                        }
                        if detail
                        else {},
                    )
                )
    managed = target.metadata.get("managed_references", ())
    if isinstance(managed, (list, tuple)):
        for reference in managed:
            if not isinstance(reference, Mapping):
                continue
            ref_id = reference.get("id", reference.get("ref"))
            if not isinstance(ref_id, str) or not ref_id:
                continue
            languages = reference.get("languages")
            descriptors.append(
                _descriptor(
                    target=target,
                    voice=ref_id,
                    kind="managed-reference",
                    languages=tuple(languages)
                    if isinstance(languages, (list, tuple))
                    else None,
                    metadata={"managed_reference": dict(reference)},
                )
            )
    if "reference_voice" in target.capabilities:
        descriptors.append(_descriptor(target=target, kind="reference-capable"))
    return descriptors


def list_voices(
    *,
    engine: str | None = None,
    target_id: str | None = None,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> tuple[VoiceDescriptor, ...]:
    """List normalized voice choices without opening synthesis sessions."""
    if engine is None:
        engines = [str(item["id"]) for item in list_engines() if item.get("installed")]
    else:
        engines = [normalize_engine_id(engine)]
    results: list[VoiceDescriptor] = []
    for engine_id in engines:
        targets = discover_targets(
            engine_id,
            language=language,
            offline=offline,
            refresh=refresh,
        )
        for target in targets:
            if target_id is not None and target.id != target_id:
                continue
            results.extend(
                descriptor
                for descriptor in _target_descriptors(target)
                if _language_matches(descriptor, language)
            )
    return tuple(
        sorted(results, key=lambda item: (item.engine, item.target_id, item.selector))
    )


def inspect_voice(selector: str, *, language: str | None = None) -> VoiceDescriptor:
    """Resolve one canonical voice selector to its normalized descriptor."""
    parsed = parse_voice_selector(selector)
    canonical = _selector(parsed.engine, parsed.target_id, parsed.voice, parsed.speaker)
    for descriptor in list_voices(
        engine=parsed.engine,
        target_id=parsed.target_id,
        language=language,
    ):
        if descriptor.selector == canonical:
            return descriptor
    raise ProfileError(f"No voice matches selector {canonical!r}")


def voice_descriptor_to_dict(descriptor: VoiceDescriptor) -> dict[str, Any]:
    """Return the stable JSON representation of a voice descriptor."""
    return {
        "selector": descriptor.selector,
        "engine": descriptor.engine,
        "target_id": descriptor.target_id,
        "voice": descriptor.voice,
        "speaker": descriptor.speaker,
        "kind": descriptor.kind,
        "languages": list(descriptor.languages),
        "gender": descriptor.gender,
        "locale": descriptor.locale,
        "installed": descriptor.installed,
        "metadata": json_value(
            descriptor.metadata,
            label=f"voice {descriptor.selector!r} metadata",
        ),
    }


__all__ = [
    "ParsedVoiceSelector",
    "VoiceDescriptor",
    "VoiceKind",
    "inspect_voice",
    "list_voices",
    "parse_voice_selector",
    "voice_descriptor_to_dict",
]
