"""VoiceRender-owned normalization, validation, resolution, and identity for voices."""

from __future__ import annotations

import hashlib
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

VoiceSourceKind = Literal["named", "reference", "managed_reference"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANAGED_FIELDS = (
    "resolver",
    "source_revision",
    "source_repository",
    "source_path",
    "license",
    "dataset",
    "variant",
)


@dataclass(frozen=True, slots=True)
class VoiceSource:
    """A normalized source description, without engine-prepared voice state."""

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

    def as_mapping(self) -> dict[str, str]:
        result: dict[str, str] = {"kind": self.kind, "value": self.value}
        for name in (
            "sha256",
            "resolver",
            "source_revision",
            "source_repository",
            "source_path",
            "license",
            "dataset",
            "variant",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
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


def normalize_voice_source(
    source: Mapping[str, Any], *, allow_named: bool = True
) -> VoiceSource:
    """Validate persisted source shape and return its canonical value object."""
    if not isinstance(source, Mapping):
        raise _voice_error("voice_source.invalid", "voice_source must be a table")

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
    }:
        raise _voice_error(
            "voice_source.invalid",
            "voice_source.kind must be named, reference, or managed_reference",
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
        if not path.is_absolute() or str(path) != value:
            raise _voice_error(
                "voice_source.invalid",
                "Local reference value must be an absolute normalized path",
                details={"path": value},
            )
        return VoiceSource(kind="reference", value=value, sha256=sha256)

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
        **fields,
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
        return voicefolio.managed_reference(ref, catalog=catalog)
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
    "ResolvedReferenceAudio",
    "VoiceSource",
    "VoiceSourceKind",
    "local_reference_source",
    "managed_reference_source",
    "normalize_voice_source",
    "resolve_reference_audio",
    "validate_voice_source",
    "voice_source_identity",
]
