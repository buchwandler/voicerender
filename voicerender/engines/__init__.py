"""Registry of engine adapters; runtime packages are imported lazily."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import ProfileError
from .piper import PiperAdapter
from .pocket import PocketAdapter

if TYPE_CHECKING:
    from .base import EngineAdapter

_ADAPTERS: dict[str, EngineAdapter] = {
    "piper": PiperAdapter(),
    "pocket": PocketAdapter(),
}


def register_engine(adapter: EngineAdapter) -> None:
    """Allow third-party backends without importing Readio."""
    if not adapter.id or adapter.id in _ADAPTERS:
        raise ProfileError(f"Invalid or already registered engine: {adapter.id!r}")
    _ADAPTERS[adapter.id] = adapter


def get_engine(name: str) -> EngineAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise ProfileError(
            f"Unknown engine {name!r}; available: {', '.join(sorted(_ADAPTERS))}"
        ) from exc


def list_engines() -> list[dict[str, object]]:
    return [
        {
            "id": name,
            "package": engine.package,
            "installed_version": engine.version(),
            "reference_audio": engine.capabilities.supports_reference_audio,
        }
        for name, engine in sorted(_ADAPTERS.items())
    ]


__all__ = ["get_engine", "list_engines", "register_engine"]
