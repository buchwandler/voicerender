"""Minimal extension point for engine adapters."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import VoiceBinding


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    request_id: str
    text: str
    language: str
    directives: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty string")
        if not isinstance(self.directives, Mapping):
            raise TypeError("directives must be a mapping")


@dataclass(frozen=True)
class EngineCapabilities:
    supports_builtin_voices: bool = True
    supports_reference_audio: bool = False
    supports_speaker_selection: bool = False
    supports_word_timings: bool = False
    options: frozenset[str] = field(default_factory=frozenset)
    native_semantics: frozenset[str] = field(default_factory=frozenset)
    composition_semantics: frozenset[str] = field(default_factory=frozenset)
    supports_synthesis_request: bool = False


@dataclass(frozen=True)
class SynthesisOutput:
    audio: Any
    sample_rate: int
    word_timings: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EngineSession(Protocol):
    def synthesize(
        self, *, request_id: str, text: str, language: str
    ) -> SynthesisOutput: ...

    def synthesize_request(self, request: SynthesisRequest) -> SynthesisOutput: ...


class EngineAdapter(Protocol):
    id: str
    package: str
    capabilities: EngineCapabilities

    def version(self) -> str | None: ...
    def validate(self, binding: VoiceBinding) -> None: ...
    def open(self, binding: VoiceBinding) -> AbstractContextManager[EngineSession]: ...
