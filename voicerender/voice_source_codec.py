"""Strict TOML codec for portable VoiceRender voice-source sidecars."""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_tomllib = importlib.import_module(
    "tomllib" if sys.version_info >= (3, 11) else "tomli"
)

from .errors import EngineError, InvalidEngineVoiceError
from .voice_sources import normalize_voice_source, validate_voice_source

VOICE_SOURCE_FORMAT = "voicerender-voice-source"
VOICE_SOURCE_SCHEMA_VERSION = 1
_SOURCE_FIELD_ORDER = (
    "kind",
    "format",
    "value",
    "sha256",
    "model_id",
    "model_revision",
    "layout",
    "resolver",
    "source_format",
    "source_revision",
    "source_repository",
    "source_path",
    "license",
    "dataset",
    "variant",
)


def _validated_source(source: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = normalize_voice_source(source)
    except (EngineError, ValueError) as exc:
        raise InvalidEngineVoiceError(
            str(exc),
            code=getattr(exc, "code", "voice_source.sidecar_invalid"),
            details=getattr(exc, "details", {}),
        ) from exc
    if normalized.kind not in {"reference", "managed_reference", "prepared"}:
        raise InvalidEngineVoiceError(
            "Voice-source sidecars require a local, managed, or prepared voice source",
            code="voice_source.sidecar_invalid",
        )
    diagnostics = validate_voice_source(normalized)
    if diagnostics:
        diagnostic = diagnostics[0]
        raise InvalidEngineVoiceError(
            diagnostic.message,
            code=diagnostic.code,
            details=dict(diagnostic.details),
        )
    return normalized.as_mapping()


def _toml_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        raise InvalidEngineVoiceError(
            "Nested voice-source metadata must be encoded as a TOML table",
            code="voice_source.sidecar_invalid",
        )
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    raise InvalidEngineVoiceError(
        f"Voice-source sidecar cannot encode value of type {type(value).__name__}",
        code="voice_source.sidecar_invalid",
    )


def _toml_key(name: str) -> str:
    if name and all(char.isalnum() or char in "_-" for char in name):
        return name
    return json.dumps(name, ensure_ascii=False)


def _serialize_metadata_table(
    lines: list[str], table: tuple[str, ...], metadata: Mapping[str, Any]
) -> None:
    lines.extend(("", "[" + ".".join(_toml_key(item) for item in table) + "]"))
    nested: list[tuple[str, Mapping[str, Any]]] = []
    for name, value in sorted(metadata.items()):
        if not isinstance(name, str):
            raise InvalidEngineVoiceError(
                "Voice-source metadata keys must be strings",
                code="voice_source.sidecar_invalid",
            )
        if isinstance(value, Mapping):
            nested.append((name, value))
        else:
            lines.append(f"{_toml_key(name)} = {_toml_value(value)}")
    for name, value in nested:
        _serialize_metadata_table(lines, (*table, name), value)


def _serialize(source: Mapping[str, Any]) -> str:
    lines = [
        f"format = {_toml_value(VOICE_SOURCE_FORMAT)}",
        f"schema_version = {VOICE_SOURCE_SCHEMA_VERSION}",
        "",
        "[voice_source]",
    ]
    for name in _SOURCE_FIELD_ORDER:
        if name in source:
            lines.append(f"{name} = {_toml_value(source[name])}")
    origin = source.get("origin")
    if origin is not None:
        if not isinstance(origin, Mapping):
            raise InvalidEngineVoiceError(
                "voice_source.origin must be a table",
                code="voice_source.sidecar_invalid",
            )
        lines.extend(("", "[voice_source.origin]"))
        for name, value in origin.items():
            if name != "runtime_identity":
                lines.append(f"{name} = {_toml_value(value)}")
        runtime_identity = origin.get("runtime_identity")
        if runtime_identity is not None:
            if not isinstance(runtime_identity, Mapping):
                raise InvalidEngineVoiceError(
                    "voice_source.origin.runtime_identity must be a table",
                    code="voice_source.sidecar_invalid",
                )
            lines.extend(("", "[voice_source.origin.runtime_identity]"))
            for name, value in runtime_identity.items():
                lines.append(f"{name} = {_toml_value(value)}")
    metadata = source.get("metadata")
    if metadata:
        if not isinstance(metadata, Mapping):
            raise InvalidEngineVoiceError(
                "voice_source.metadata must be a table",
                code="voice_source.sidecar_invalid",
            )
        _serialize_metadata_table(lines, ("voice_source", "metadata"), metadata)
    return "\n".join(lines) + "\n"


def load_voice_source(path: str | Path) -> Mapping[str, Any]:
    """Load, validate, and hash-check a VoiceRender voice-source sidecar."""
    source_path = Path(path)
    try:
        with source_path.open("rb") as stream:
            payload = _tomllib.load(stream)
    except (OSError, ValueError) as exc:
        raise InvalidEngineVoiceError(
            f"Cannot read voice-source sidecar {source_path}: {exc}",
            code="voice_source.sidecar_invalid",
        ) from exc

    if not isinstance(payload, Mapping):
        raise InvalidEngineVoiceError(
            "Voice-source sidecar must be a TOML table",
            code="voice_source.sidecar_invalid",
        )
    unknown = sorted(set(payload) - {"format", "schema_version", "voice_source"})
    if unknown:
        raise InvalidEngineVoiceError(
            "Voice-source sidecar contains unsupported top-level fields",
            code="voice_source.sidecar_invalid",
            details={"fields": unknown},
        )
    if payload.get("format") != VOICE_SOURCE_FORMAT:
        raise InvalidEngineVoiceError(
            "Unsupported voice-source sidecar format",
            code="voice_source.sidecar_invalid",
        )
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != VOICE_SOURCE_SCHEMA_VERSION
    ):
        raise InvalidEngineVoiceError(
            f"Unsupported voice-source sidecar schema version {schema_version!r}",
            code="voice_source.sidecar_invalid",
        )
    raw_source = payload.get("voice_source")
    if not isinstance(raw_source, Mapping):
        raise InvalidEngineVoiceError(
            "Voice-source sidecar requires a [voice_source] table",
            code="voice_source.sidecar_invalid",
        )
    return _validated_source(raw_source)


def save_voice_source(source: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and atomically save a deterministic VoiceRender sidecar."""
    normalized = _validated_source(source)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp-", suffix=".toml", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(_serialize(normalized))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return target


__all__ = [
    "VOICE_SOURCE_FORMAT",
    "VOICE_SOURCE_SCHEMA_VERSION",
    "load_voice_source",
    "save_voice_source",
]
