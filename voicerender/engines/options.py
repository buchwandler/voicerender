"""Normalized introspection for adapter-accepted engine options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .base import EngineOption
from .registry import get_adapter


def engine_options(
    engine: str, *, target_id: str | None = None
) -> tuple[EngineOption, ...]:
    """Describe every option an adapter currently accepts.

    Adapters may expose richer metadata through an ``option_descriptors`` mapping or
    callable. Missing metadata is represented with ``value_type='unknown'`` so
    introspection remains complete without weakening adapter validation.
    """
    adapter = get_adapter(engine)
    capabilities = adapter.capabilities()
    accepted = set(capabilities.option_names) | set(capabilities.options)
    raw = getattr(adapter, "option_descriptors", None)
    if callable(raw):
        raw = raw(target_id=target_id)
    descriptors: dict[str, EngineOption] = {}
    if isinstance(raw, Mapping):
        entries = raw.items()
    elif raw is not None:
        entries = ((item.name, item) for item in raw)
    else:
        entries = ()
    for name, value in entries:
        if isinstance(value, EngineOption):
            descriptors[name] = value
        elif isinstance(value, Mapping):
            descriptors[name] = EngineOption(name=name, **dict(value))
    for name in accepted:
        descriptors.setdefault(name, EngineOption(name=name, value_type="unknown"))
    return tuple(descriptors[name] for name in sorted(descriptors))


def engine_option_to_dict(option: EngineOption) -> dict[str, Any]:
    """Return the stable JSON form of an option descriptor."""
    value = asdict(option)
    value["choices"] = list(option.choices)
    return value


__all__ = ["engine_option_to_dict", "engine_options"]
