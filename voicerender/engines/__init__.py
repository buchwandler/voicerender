"""VoiceRender engine APIs and the legacy batch-render adapter registry."""

from __future__ import annotations

import importlib
from typing import Any

from ..errors import ProfileError
from .registry import (
    discover_targets,
    engine_status,
    get_adapter,
    list_engines,
    open_session,
    register_adapter,
    resolve_voice,
    synthesize,
)
from .selection import EngineRequest

# The original VoicePlan renderer remains temporarily available, but its two
# adapters are separate from the request-centric registry used by public APIs.
_LEGACY_ENGINES = {
    "piper": ("voicerender.engines.piper", "PiperAdapter"),
    "pocket": ("voicerender.engines.pocket", "PocketAdapter"),
}
_legacy_adapters: dict[str, Any] = {}


def register_engine(adapter: Any) -> None:
    """Register an adapter for the transitional VoicePlan renderer."""
    if (
        not adapter.id
        or adapter.id in _legacy_adapters
        or adapter.id in _LEGACY_ENGINES
    ):
        raise ProfileError(f"Invalid or already registered engine: {adapter.id!r}")
    _legacy_adapters[adapter.id] = adapter


def get_engine(name: str) -> Any:
    """Return a legacy renderer adapter, importing its implementation lazily."""
    if name in _legacy_adapters:
        return _legacy_adapters[name]
    try:
        module_name, class_name = _LEGACY_ENGINES[name]
    except KeyError as exc:
        raise ProfileError(
            f"Unknown legacy engine {name!r}; available: {', '.join(sorted(_LEGACY_ENGINES))}"
        ) from exc
    adapter = getattr(importlib.import_module(module_name), class_name)()
    _legacy_adapters[name] = adapter
    return adapter


__all__ = [
    "EngineRequest",
    "discover_targets",
    "engine_status",
    "get_adapter",
    "get_engine",
    "list_engines",
    "open_session",
    "register_adapter",
    "register_engine",
    "resolve_voice",
    "synthesize",
]
