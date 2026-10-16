"""VoiceRender-owned normalization, validation, resolution, and identity for voices."""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import voicefolio

if TYPE_CHECKING:
    from .engines.base import EngineDiagnostic
from .errors import EngineError, InvalidEngineOptionError, InvalidEngineVoiceError
from .managed_voices import (
    ResolvedManagedVoice,
    resolve_managed_voice,
    validate_managed_voice,
)

VoiceSourceKind = Literal["named", "reference", "managed_reference", "prepared"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_QWEN_ORIGIN_SCHEMA = "voicerender.voice-origin.qwen3-design.v1"
_MANAGED_FIELDS = (
    "resolver",
    "source_revision",
    "source_repository",
    "source_path",
    "license",
    "dataset",
    "variant",
)
_QWEN_ORIGIN_FIELDS = (
    "schema",
    "generator",
    "target_id",
    "model_revision",
    "qwen3synth_version",
    "onnxvoice_version",
    "prompt",
    "reference_text",
    "language",
    "quality",
    "provider",
    "temperature",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
    "seed",
    "model_id",
    "runtime_identity",
)


@dataclass(frozen=True, slots=True)
class VoiceSource:
    """A normalized voice source and its optional provenance."""

    kind: VoiceSourceKind
    value: str
    sha256: str | None = None
    resolver: str | None = None
    source_revision: str | None = None
    source_repository: str | None = None
    source_path: str | None = None
    license: str | None = None
    dataset: str | None = None
    variant: str | None = None
    format: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    layout: str | None = None
    origin: Mapping[str, Any] | None = None

    def as_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "value": self.value}
        for name in (
            "sha256",
            "resolver",
            "source_revision",
            "source_repository",
            "source_path",
            "license",
            "dataset",
            "variant",
            "format",
            "model_id",
            "model_revision",
            "layout",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.origin is not None:
            result["origin"] = dict(self.origin)
        return result


@dataclass(frozen=True, slots=True)
class ResolvedReferenceAudio:
    """A verified local reference WAV and its stable source provenance."""

    kind: Literal["reference", "managed_reference"]
    path: Path
    sha256: str
    resolver: str | None = None
    ref: str | None = None
    source_revision: str | None = None
    source_repository: str | None = None
    source_path: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedPreparedVoice:
    """A verified local prepared-voice artifact and its stable identity."""

    path: Path
    sha256: str
    format: str
    model_id: str | None = None
    model_revision: str | None = None
    layout: str | None = None


def _normalize_qwen_origin(origin: Any) -> dict[str, Any]:
    if not isinstance(origin, Mapping):
        raise _voice_error(
            "voice_source.origin_invalid",
            "Qwen VoiceDesign origin must be a table",
        )
    allowed = set(_QWEN_ORIGIN_FIELDS)
    unknown = sorted(set(origin) - allowed)
    if unknown:
        raise _voice_error(
            "voice_source.origin_invalid",
            "Qwen VoiceDesign origin contains unsupported fields",
            details={"fields": unknown},
        )
    if (
        origin.get("schema") != _QWEN_ORIGIN_SCHEMA
        or origin.get("generator") != "qwen3"
    ):
        raise _voice_error(
            "voice_source.origin_invalid",
            "Qwen VoiceDesign origin schema or generator is invalid",
        )

    required_strings = tuple(
        name
        for name in _QWEN_ORIGIN_FIELDS
        if name
        not in {
            "schema",
            "generator",
            "temperature",
            "top_k",
            "repetition_penalty",
            "max_new_tokens",
            "seed",
            "model_id",
            "runtime_identity",
        }
    )
    normalized: dict[str, Any] = {
        "schema": _QWEN_ORIGIN_SCHEMA,
        "generator": "qwen3",
    }
    for name in required_strings:
        value = origin.get(name)
        if not isinstance(value, str) or not value.strip():
            raise _voice_error(
                "voice_source.origin_invalid",
                f"Qwen VoiceDesign origin requires a non-empty {name}",
                details={"field": name},
            )
        normalized[name] = value

    temperature = origin.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 0 < temperature <= 5
    ):
        raise _voice_error(
            "voice_source.origin_invalid",
            "Qwen VoiceDesign temperature must be finite and in (0, 5]",
            details={"field": "temperature"},
        )
    top_k = origin.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise _voice_error(
            "voice_source.origin_invalid",
            "Qwen VoiceDesign top_k must be a non-negative integer",
            details={"field": "top_k"},
        )
    repetition_penalty = origin.get("repetition_penalty")
    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or not math.isfinite(float(repetition_penalty))
        or repetition_penalty <= 0
    ):
        raise _voice_error(
            "voice_source.origin_invalid",
            "Qwen VoiceDesign repetition_penalty must be finite and positive",
            details={"field": "repetition_penalty"},
        )
    max_new_tokens = origin.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= 8192
    ):
        raise _voice_error(
            "voice_source.origin_invalid",
            "Qwen VoiceDesign max_new_tokens must be between 1 and 8192",
            details={"field": "max_new_tokens"},
        )
    normalized.update(
        {
            "temperature": temperature,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
        }
    )
    seed = origin.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise _voice_error(
                "voice_source.origin_invalid",
                "Qwen VoiceDesign seed must be an integer when present",
                details={"field": "seed"},
            )
        normalized["seed"] = seed

    model_id = origin.get("model_id")
    if model_id is not None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise _voice_error(
                "voice_source.origin_invalid",
                "Qwen VoiceDesign model_id must be a non-empty string",
                details={"field": "model_id"},
            )
        normalized["model_id"] = model_id

    runtime_identity = origin.get("runtime_identity")
    if runtime_identity is not None:
        if not isinstance(runtime_identity, Mapping):
            raise _voice_error(
                "voice_source.origin_invalid",
                "Qwen VoiceDesign runtime_identity must be a table",
                details={"field": "runtime_identity"},
            )
        allowed_runtime_fields = {
            "engine",
            "engine_version",
            "request_api_version",
            "onnxvoice_version",
            "layouts",
            "sample_rate",
        }
        unknown_runtime_fields = sorted(set(runtime_identity) - allowed_runtime_fields)
        if unknown_runtime_fields:
            raise _voice_error(
                "voice_source.origin_invalid",
                "Qwen VoiceDesign runtime_identity contains unsupported fields",
                details={"fields": unknown_runtime_fields},
            )
        runtime_strings: dict[str, str] = {}
        for name in ("engine", "engine_version", "onnxvoice_version"):
            value = runtime_identity.get(name)
            if not isinstance(value, str) or not value.strip():
                raise _voice_error(
                    "voice_source.origin_invalid",
                    f"Qwen VoiceDesign runtime_identity requires a non-empty {name}",
                    details={"field": f"runtime_identity.{name}"},
                )
            runtime_strings[name] = value
        request_api_version = runtime_identity.get("request_api_version")
        if (
            isinstance(request_api_version, bool)
            or not isinstance(request_api_version, int)
            or request_api_version <= 0
        ):
            raise _voice_error(
                "voice_source.origin_invalid",
                "Qwen VoiceDesign runtime request_api_version must be positive",
                details={"field": "runtime_identity.request_api_version"},
            )
        sample_rate = runtime_identity.get("sample_rate")
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise _voice_error(
                "voice_source.origin_invalid",
                "Qwen VoiceDesign runtime sample_rate must be positive",
                details={"field": "runtime_identity.sample_rate"},
            )
        raw_layouts = runtime_identity.get("layouts")
        if (
            not isinstance(raw_layouts, (list, tuple))
            or not raw_layouts
            or any(
                not isinstance(layout, str) or not layout.strip()
                for layout in raw_layouts
            )
        ):
            raise _voice_error(
                "voice_source.origin_invalid",
                "Qwen VoiceDesign runtime layouts must be a non-empty string array",
                details={"field": "runtime_identity.layouts"},
            )
        normalized["runtime_identity"] = {
            **runtime_strings,
            "request_api_version": request_api_version,
            "layouts": tuple(raw_layouts),
            "sample_rate": sample_rate,
        }
    return normalized


def normalize_voice_source(
    source: Mapping[str, Any], *, allow_named: bool = True
) -> VoiceSource:
    """Validate persisted source shape and return its canonical value object."""
    if not isinstance(source, Mapping):
        raise _voice_error("voice_source.invalid", "voice_source must be a table")
    if any(not isinstance(key, str) for key in source):
        raise _voice_error(
            "voice_source.invalid", "voice_source field names must be strings"
        )

    kind = source.get("kind")
    value = source.get("value", source.get("path"))
    if not isinstance(value, str) or not value.strip():
        raise _voice_error(
            "voice_source.invalid", "voice_source requires a non-empty value"
        )
    if not isinstance(kind, str) or kind not in {
        "named",
        "reference",
        "managed_reference",
        "prepared",
    }:
        raise _voice_error(
            "voice_source.invalid",
            "voice_source.kind must be named, reference, managed_reference, or prepared",
        )
    if kind == "named":
        if not allow_named:
            raise _voice_error(
                "voice_source.invalid",
                "This engine does not accept named voice sources",
            )
        return VoiceSource(kind="named", value=value)

    sha256 = source.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise _voice_error(
            "voice_source.invalid",
            "voice_source requires a lowercase 64-character SHA-256 digest",
            details={"field": "sha256"},
        )

    if kind == "reference":
        allowed = {"kind", "value", "path", "sha256", "origin"}
        unknown = sorted(set(source) - allowed - set(_MANAGED_FIELDS))
        if unknown:
            raise _voice_error(
                "voice_source.invalid",
                "Local reference source contains unsupported fields",
                details={"fields": unknown},
            )
        forbidden = [name for name in _MANAGED_FIELDS if name in source]
        if forbidden:
            raise _voice_error(
                "voice_source.invalid",
                "Local reference sources cannot contain managed provenance fields",
                details={"fields": forbidden},
            )
        if value.startswith(("http://", "https://")):
            raise _voice_error(
                "voice_source.invalid",
                "HTTP and HTTPS reference URLs are not supported; use a local WAV or Voicefolio",
            )
        path = Path(value).expanduser()
        if not path.is_absolute() or str(path) != value or ".." in path.parts:
            raise _voice_error(
                "voice_source.invalid",
                "Local reference value must be an absolute normalized path",
                details={"path": value},
            )
        raw_origin = source.get("origin")
        origin = _normalize_qwen_origin(raw_origin) if raw_origin is not None else None
        return VoiceSource(kind="reference", value=value, sha256=sha256, origin=origin)

    if kind == "prepared":
        allowed = {
            "kind",
            "value",
            "path",
            "sha256",
            "format",
            "model_id",
            "model_revision",
            "layout",
        }
        unknown = sorted(set(source) - allowed)
        if unknown:
            raise _voice_error(
                "voice_source.prepared_invalid",
                "Prepared voice source contains unsupported fields",
                details={"fields": unknown},
            )
        path = Path(value).expanduser()
        if not path.is_absolute() or str(path) != value or ".." in path.parts:
            raise _voice_error(
                "voice_source.prepared_invalid",
                "Prepared voice value must be an absolute normalized path",
                details={"path": value},
            )
        format_value = source.get("format")
        if not isinstance(format_value, str) or not format_value.strip():
            raise _voice_error(
                "voice_source.prepared_invalid",
                "Prepared voice source requires a non-empty format",
                details={"field": "format"},
            )
        metadata: dict[str, str | None] = {}
        for name in ("model_id", "model_revision", "layout"):
            item = source.get(name)
            if item is not None and (not isinstance(item, str) or not item.strip()):
                raise _voice_error(
                    "voice_source.prepared_invalid",
                    f"Prepared voice {name} must be a non-empty string",
                    details={"field": name},
                )
            metadata[name] = item
        return VoiceSource(
            kind="prepared",
            value=value,
            sha256=sha256,
            format=format_value,
            model_id=metadata["model_id"],
            model_revision=metadata["model_revision"],
            layout=metadata["layout"],
        )

    allowed = {"kind", "value", "path", "sha256", *_MANAGED_FIELDS}
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise _voice_error(
            "managed_voice.invalid_source",
            "Managed reference source contains unsupported fields",
            details={"fields": unknown},
        )
    raw_resolver = source.get("resolver")
    if raw_resolver is None and value.startswith("kyutai-tts-voices:"):
        warnings.warn(
            "Managed references without resolver are deprecated; persist resolver='voicefolio'.",
            DeprecationWarning,
            stacklevel=2,
        )
        resolver = "voicefolio"
    elif not isinstance(raw_resolver, str) or not raw_resolver.strip():
        raise _voice_error(
            "managed_voice.resolver_missing",
            "Managed reference source requires a non-empty resolver",
            details={"resolver": raw_resolver},
        )
    else:
        resolver = raw_resolver
    if resolver != "voicefolio":
        raise _option_error(
            "voice_source.resolver_unsupported",
            f"Unsupported managed voice resolver {resolver!r}",
            details={"resolver": resolver},
        )

    fields: dict[str, str | None] = {}
    for name in ("source_revision", "source_repository", "source_path"):
        item = source.get(name)
        if not isinstance(item, str) or not item.strip():
            raise _voice_error(
                "managed_voice.invalid_source",
                f"Voicefolio managed-reference source requires non-empty {name}",
                details={"field": name, "ref": value},
            )
        fields[name] = item
    for name in ("license", "dataset", "variant"):
        item = source.get(name)
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise _voice_error(
                "managed_voice.invalid_source",
                f"Voicefolio managed-reference {name} must be a non-empty string",
                details={"field": name, "ref": value},
            )
        fields[name] = item
    return VoiceSource(
        kind="managed_reference",
        value=value,
        sha256=sha256,
        resolver=resolver,
        source_revision=fields["source_revision"],
        source_repository=fields["source_repository"],
        source_path=fields["source_path"],
        license=fields["license"],
        dataset=fields["dataset"],
        variant=fields["variant"],
    )


def local_reference_source(path: str | Path) -> dict[str, str]:
    """Create an absolute local reference binding and compute its content pin."""
    raw = str(path)
    if raw.startswith(("http://", "https://")):
        raise _voice_error(
            "voice_source.invalid",
            "HTTP and HTTPS reference URLs are not supported; use a local WAV or Voicefolio",
        )
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _voice_error(
            "voice_source.reference_missing",
            f"Local reference audio does not exist: {Path(path).expanduser()}",
            details={"path": str(Path(path).expanduser())},
        ) from exc
    if not resolved.is_file():
        raise _voice_error(
            "voice_source.reference_not_file",
            f"Local reference audio is not a regular file: {resolved}",
            details={"path": str(resolved)},
        )
    return {
        "kind": "reference",
        "value": str(resolved),
        "sha256": _hash_file(resolved),
    }


def local_prepared_voice_source(
    path: str | Path,
    *,
    format: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an absolute prepared-voice binding with its complete file hash."""
    if not isinstance(format, str) or not format.strip():
        raise _voice_error(
            "voice_source.prepared_invalid",
            "Prepared voice format must be a non-empty string",
        )
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _voice_error(
            "voice_source.prepared_invalid",
            f"Prepared voice file does not exist: {Path(path).expanduser()}",
            details={"path": str(Path(path).expanduser())},
        ) from exc
    if not resolved.is_file():
        raise _voice_error(
            "voice_source.prepared_invalid",
            f"Prepared voice is not a regular file: {resolved}",
            details={"path": str(resolved)},
        )
    fields = dict(metadata or {})
    unknown = sorted(set(fields) - {"model_id", "model_revision", "layout"})
    if unknown:
        raise _voice_error(
            "voice_source.prepared_invalid",
            "Prepared voice metadata contains unsupported fields",
            details={"fields": unknown},
        )
    source: dict[str, Any] = {
        "kind": "prepared",
        "format": format,
        "value": str(resolved),
        "sha256": _hash_file(resolved),
        **fields,
    }
    return normalize_voice_source(source).as_mapping()


def managed_reference_source(
    ref: str,
    *,
    catalog_paths: Iterable[str | Path] | str | Path | None = None,
) -> dict[str, Any]:
    """Create the fully pinned portable mapping for a Voicefolio reference."""
    if not isinstance(ref, str) or not ref.strip():
        raise _voice_error(
            "managed_voice.invalid_source",
            "Voicefolio reference must be a non-empty string",
            details={"resolver": "voicefolio"},
        )
    paths = _catalog_path_tuple(catalog_paths)
    try:
        catalog = voicefolio.load_catalog(extra_paths=paths)
        result = voicefolio.managed_reference(ref, catalog=catalog)
        if not isinstance(result, Mapping):
            raise _voice_error(
                "managed_voice.invalid_source",
                "Voicefolio returned a managed voice source that is not a table",
                details={"resolver": "voicefolio", "ref": ref},
            )
        return dict(result)
    except (voicefolio.VoicefolioError, ValueError) as exc:
        raise _voice_error(
            "managed_voice.invalid_source",
            f"Cannot create Voicefolio managed reference {ref!r}: {exc}",
            details={"resolver": "voicefolio", "ref": ref},
        ) from exc


def validate_voice_source(
    source: VoiceSource | Mapping[str, Any],
    *,
    offline: bool = False,
    catalog_paths: Iterable[str | Path] | str | Path | None = None,
) -> tuple[EngineDiagnostic, ...]:
    """Validate source shape, local content, or managed metadata without fetching WAVs."""
    del offline  # Metadata validation is identical in online and offline modes.
    from .engines.base import EngineDiagnostic

    try:
        normalized = (
            source
            if isinstance(source, VoiceSource)
            else normalize_voice_source(source)
        )
        if normalized.kind == "reference":
            _validate_local_reference(normalized)
        elif normalized.kind == "prepared":
            _validate_prepared_voice(normalized)
        elif normalized.kind == "managed_reference":
            validate_managed_voice(normalized.as_mapping(), catalog_paths=catalog_paths)
    except (EngineError, OSError, ValueError) as exc:
        return (
            EngineDiagnostic(
                code=getattr(exc, "code", "voice_source.invalid"),
                severity="error",
                message=str(exc),
                field="render.target.voice",
                details=getattr(exc, "details", {}),
            ),
        )
    return ()


def resolve_prepared_voice(
    source: VoiceSource | Mapping[str, Any],
    *,
    expected_format: str | None = None,
) -> ResolvedPreparedVoice:
    """Resolve a prepared voice file after checking its path, format, and hash."""
    normalized = (
        source if isinstance(source, VoiceSource) else normalize_voice_source(source)
    )
    if normalized.kind != "prepared":
        raise _voice_error(
            "voice_source.prepared_invalid",
            "A prepared voice source is required for resolution",
        )
    if expected_format is not None and normalized.format != expected_format:
        raise _voice_error(
            "voice_source.prepared_invalid",
            f"Prepared voice format {normalized.format!r} does not match {expected_format!r}",
            details={
                "expected_format": expected_format,
                "actual_format": normalized.format,
            },
        )
    path, digest = _validate_prepared_voice(normalized)
    return ResolvedPreparedVoice(
        path=path,
        sha256=digest,
        format=str(normalized.format),
        model_id=normalized.model_id,
        model_revision=normalized.model_revision,
        layout=normalized.layout,
    )


def resolve_reference_audio(
    source: VoiceSource | Mapping[str, Any],
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    catalog_paths: Iterable[str | Path] | str | Path | None = None,
) -> ResolvedReferenceAudio:
    """Resolve a local or managed source to a verified local WAV on demand."""
    normalized = (
        source if isinstance(source, VoiceSource) else normalize_voice_source(source)
    )
    if normalized.kind == "reference":
        path, digest = _validate_local_reference(normalized)
        return ResolvedReferenceAudio(kind="reference", path=path, sha256=digest)
    if normalized.kind == "prepared":
        raise _voice_error(
            "voice_source.prepared_invalid",
            "Prepared voices are not reference audio",
        )
    if normalized.kind != "managed_reference":
        raise _voice_error(
            "voice_source.invalid",
            "A reference audio source is required for resolution",
        )
    resolved: ResolvedManagedVoice = resolve_managed_voice(
        normalized.as_mapping(),
        cache_dir=cache_dir,
        offline=offline,
        catalog_paths=catalog_paths,
    )
    return ResolvedReferenceAudio(
        kind="managed_reference",
        path=resolved.path,
        sha256=resolved.sha256,
        resolver=resolved.resolver,
        ref=resolved.ref,
        source_revision=resolved.source_revision,
        source_repository=resolved.source_repository,
        source_path=resolved.source_path,
    )


def voice_source_identity(
    source: VoiceSource | Mapping[str, Any],
) -> Mapping[str, str]:
    """Return canonical synthesis identity fields without machine-local paths."""
    normalized = (
        source if isinstance(source, VoiceSource) else normalize_voice_source(source)
    )
    if normalized.kind == "named":
        return {"kind": "named", "value": normalized.value}
    if normalized.kind == "reference":
        return {"kind": "reference", "sha256": str(normalized.sha256)}
    if normalized.kind == "prepared":
        return {
            "kind": "prepared",
            "format": str(normalized.format),
            "sha256": str(normalized.sha256),
        }
    return {
        "kind": "managed_reference",
        "resolver": str(normalized.resolver),
        "ref": normalized.value,
        "sha256": str(normalized.sha256),
        "source_revision": str(normalized.source_revision),
    }


def _validate_local_reference(source: VoiceSource) -> tuple[Path, str]:
    path = Path(source.value)
    if not path.is_absolute() or str(path) != source.value:
        raise _voice_error(
            "voice_source.invalid",
            "Local reference value must be an absolute normalized path",
            details={"path": source.value},
        )
    if not path.exists():
        raise _voice_error(
            "voice_source.reference_missing",
            f"Local reference audio does not exist: {path}",
            details={"path": str(path)},
        )
    if not path.is_file():
        raise _voice_error(
            "voice_source.reference_not_file",
            f"Local reference audio is not a regular file: {path}",
            details={"path": str(path)},
        )
    canonical_path = path.resolve(strict=True)
    if str(canonical_path) != source.value:
        raise _voice_error(
            "voice_source.invalid",
            "Local reference value must be an absolute normalized path",
            details={"path": source.value},
        )
    digest = _hash_file(path)
    if digest != source.sha256:
        raise _voice_error(
            "voice_source.reference_changed",
            f"Local reference audio changed at {path}: expected SHA-256 {source.sha256}, found {digest}",
            details={
                "path": str(path),
                "expected_sha256": source.sha256,
                "actual_sha256": digest,
            },
        )
    return path, digest


def _validate_prepared_voice(source: VoiceSource) -> tuple[Path, str]:
    path = Path(source.value)
    if not path.is_absolute() or str(path) != source.value:
        raise _voice_error(
            "voice_source.prepared_invalid",
            "Prepared voice value must be an absolute normalized path",
            details={"path": source.value},
        )
    if not path.exists():
        raise _voice_error(
            "voice_source.prepared_invalid",
            f"Prepared voice file does not exist: {path}",
            details={"path": str(path)},
        )
    if not path.is_file():
        raise _voice_error(
            "voice_source.prepared_invalid",
            f"Prepared voice is not a regular file: {path}",
            details={"path": str(path)},
        )
    canonical_path = path.resolve(strict=True)
    if str(canonical_path) != source.value:
        raise _voice_error(
            "voice_source.prepared_invalid",
            "Prepared voice value must be an absolute normalized path",
            details={"path": source.value},
        )
    digest = _hash_file(path)
    if digest != source.sha256:
        raise _voice_error(
            "voice_source.prepared_invalid",
            f"Prepared voice changed at {path}: expected SHA-256 {source.sha256}, found {digest}",
            details={
                "path": str(path),
                "expected_sha256": source.sha256,
                "actual_sha256": digest,
            },
        )
    return path, digest


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_path_tuple(
    catalog_paths: Iterable[str | Path] | str | Path | None,
) -> tuple[str | Path, ...]:
    if catalog_paths is None:
        return ()
    if isinstance(catalog_paths, (str, Path)):
        return (catalog_paths,)
    return tuple(catalog_paths)


def _voice_error(
    code: str, message: str, *, details: dict[str, Any] | None = None
) -> InvalidEngineVoiceError:
    return InvalidEngineVoiceError(message, code=code, details=details)


def _option_error(
    code: str, message: str, *, details: dict[str, Any] | None = None
) -> InvalidEngineOptionError:
    return InvalidEngineOptionError(message, code=code, details=details)


__all__ = [
    "ResolvedPreparedVoice",
    "ResolvedReferenceAudio",
    "VoiceSource",
    "VoiceSourceKind",
    "local_prepared_voice_source",
    "local_reference_source",
    "managed_reference_source",
    "normalize_voice_source",
    "resolve_prepared_voice",
    "resolve_reference_audio",
    "validate_voice_source",
    "voice_source_identity",
]
