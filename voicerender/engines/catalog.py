"""Neutral catalog types for engine discovery.

This module provides engine-neutral types for representing synthesis targets
(voice bundles, models, etc.) without forcing them into PyKokoro-shaped data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TargetVoice:
    """Typed descriptive metadata and synthesis-language capabilities for a voice.

    Adapters populate canonical VoiceRender values; generic layers only project them.
    ``languages`` describes this voice's synthesis-language capabilities,
    separately from its singular descriptive language and locale.
    """

    id: str
    gender: str = "unknown"
    language: str = "unknown"
    locale: str = "unknown"
    language_label: str = "unknown"
    languages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SynthesisTarget:
    """A neutral representation of a synthesis target.

    For PyKokoro, this represents one model/distribution with its voices and qualities.
    For PiperSynth, this represents one voice bundle.
    """

    engine: str
    id: str
    display_name: str
    languages: tuple[str, ...] = ()
    status: str = "ready"
    runtime_available: bool = True
    sample_rate: int | None = None
    voices: tuple[str, ...] = ()
    speakers: tuple[str, ...] = ()
    voice_details: tuple[TargetVoice, ...] = ()
    default_voice: str | None = None
    qualities: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogRequest:
    """Request for target discovery."""

    engine: str | None = None
    language: str | None = None
    offline: bool = False
    refresh: bool = False
    preference: str = "auto"


@dataclass(frozen=True, slots=True)
class CatalogResult:
    """Result of target discovery."""

    targets: tuple[SynthesisTarget, ...] = ()
    registry_source: str | None = None
    cache_fallback: bool = False
    offline: bool = False
    refreshed: bool = False


__all__ = [
    "CatalogRequest",
    "CatalogResult",
    "SynthesisTarget",
    "TargetVoice",
]
