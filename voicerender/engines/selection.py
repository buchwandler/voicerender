"""Engine selection and resolution for VoiceRender.

This module provides types and utilities for resolving engine selections.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .base import EngineSelection


@dataclass(frozen=True, slots=True)
class EngineRequest:
    """Request for engine selection."""

    engine: str | None = None
    target_id: str | None = None
    language: str | None = None
    voice: str | None = None
    speaker: str | int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    offline: bool = False
    refresh: bool = False
    engine_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EngineResolution:
    """Result of engine selection."""

    selection: EngineSelection
    diagnostics: tuple[Any, ...] = ()


def validate_engine_option(
    engine_id: str,
    option_name: str,
    supported_options: frozenset[str],
) -> bool:
    """Check if an option is supported by the engine."""
    return option_name in supported_options


def reject_incompatible_options(
    engine_id: str,
    options: Mapping[str, Any],
    supported_options: frozenset[str],
) -> list[str]:
    """Return list of unsupported option names."""
    return [name for name in options if name not in supported_options]


__all__ = [
    "EngineRequest",
    "EngineResolution",
    "reject_incompatible_options",
    "validate_engine_option",
]
