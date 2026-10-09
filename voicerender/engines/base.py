"""Minimal extension point for engine adapters."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol
from collections.abc import Mapping

from ..models import VoiceBinding


@dataclass(frozen=True)
class EngineCapabilities:
    supports_builtin_voices: bool = True
    supports_reference_audio: bool = False
    supports_speaker_selection: bool = False
    supports_word_timings: bool = False
    options: frozenset[str] = field(default_factory=frozenset)


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


class EngineAdapter(Protocol):
    id: str
    package: str
    capabilities: EngineCapabilities

    def version(self) -> str | None: ...
    def validate(self, binding: VoiceBinding) -> None: ...
    def open(self, binding: VoiceBinding) -> AbstractContextManager[EngineSession]: ...
