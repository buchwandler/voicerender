"""Safe JSON transport conversion for VoiceRender-owned boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def json_value(value: Any, *, label: str = "value") -> Any:
    """Return a detached JSON-compatible projection of a supported value.

    Dataclasses are walked by field access rather than ``dataclasses.asdict`` so
    engine-owned mappings are never deep-copied before Mapping recursion.
    """
    return _json_value(value, label=label, path=())


def _json_value(value: Any, *, label: str, path: tuple[str, ...]) -> Any:
    if isinstance(value, Enum):
        return _json_value(value.value, label=label, path=path)
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(
                getattr(value, item.name), label=label, path=(*path, item.name)
            )
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, label=label, path=(*path, str(key)))
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _json_value(item, label=label, path=(*path, str(index)))
            for index, item in enumerate(value)
        ]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist(), label=label, path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise ValueError(
            f"{_location(label, path)} contains non-finite float {value!r}"
        )
    raise TypeError(
        f"{_location(label, path)} contains unsupported JSON value "
        f"{type(value).__name__}"
    )


def _location(label: str, path: tuple[str, ...]) -> str:
    if not path:
        return label
    return f"{label} at {'.'.join(path)}"


__all__ = ["json_value"]
