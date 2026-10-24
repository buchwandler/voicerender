"""Engine-neutral validation and resolution for portable managed voices."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import voicefolio
from voicefolio import (
    Catalog,
    IntegrityError,
    OfflineError,
    ResolvedVoice,
    VoiceEntry,
    VoicefolioError,
    VoiceNotFoundError,
)

from .errors import (
    EngineBackendError,
    InvalidEngineOptionError,
    InvalidEngineVoiceError,
)


@runtime_checkable
class _MaterializedVoice(Protocol):
    source: ResolvedVoice
    path: Path
    sha256: str


VOICEFOLIO_RESOLVER = "voicefolio"
_MANAGED_PIN_FIELDS = (
    "sha256",
    "source_revision",
    "source_repository",
    "source_path",
)


@dataclass(frozen=True, slots=True)
class ResolvedManagedVoice:
    """A verified renderer-ready asset and its independent portable source pin."""

    resolver: str
    ref: str
    path: Path
    source_sha256: str
    resolved_sha256: str
    source_format: str
    source_revision: str
    source_repository: str
    source_path: str
    license: str | None = None
    dataset: str | None = None
    variant: str | None = None

    @property
    def sha256(self) -> str:
        """Compatibility alias for the renderer-ready asset digest."""
        return self.resolved_sha256


def validate_managed_voice(
    source: Mapping[str, Any],
    *,
    catalog_paths: Iterable[str | Path] | str | Path | None = None,
) -> VoiceEntry:
    """Validate persisted pins against Voicefolio metadata without fetching audio."""
    resolver, ref = _source_identity(source)
    catalog = _load_catalog(catalog_paths, resolver=resolver, ref=ref)
    try:
        return voicefolio.validate_voice_source(source, catalog=catalog)
    except VoicefolioError as exc:
        raise InvalidEngineVoiceError(
            f"Invalid Voicefolio managed voice {ref!r}: {exc}",
            code="managed_voice.invalid_source",
            details={"resolver": resolver, "ref": ref},
        ) from exc
    except ValueError as exc:
        raise InvalidEngineVoiceError(
            f"Invalid Voicefolio managed voice {ref!r}: {exc}",
            code="managed_voice.invalid_source",
            details={"resolver": resolver, "ref": ref},
        ) from exc


def resolve_managed_voice(
    source: Mapping[str, Any],
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    catalog_paths: Iterable[str | Path] | str | Path | None = None,
) -> ResolvedManagedVoice:
    """Resolve a pinned Voicefolio reference and return its verified local WAV."""
    resolver, ref = _source_identity(source)
    catalog = _load_catalog(catalog_paths, resolver=resolver, ref=ref)
    try:
        resolved = cast(
            ResolvedVoice | _MaterializedVoice,
            voicefolio.resolve_voice_source(
                source,
                catalog=catalog,
                cache_dir=cache_dir,
                offline=offline,
            ),
        )
    except OfflineError as exc:
        raise EngineBackendError(
            f"Voicefolio has no cached WAV for {ref!r} while offline.",
            code="managed_voice.offline_cache_miss",
            details={"resolver": resolver, "ref": ref},
            native_error_type=type(exc).__name__,
        ) from exc
    except IntegrityError as exc:
        raise EngineBackendError(
            f"Voicefolio integrity verification failed for {ref!r}: {exc}",
            code="managed_voice.integrity_mismatch",
            details={"resolver": resolver, "ref": ref},
            native_error_type=type(exc).__name__,
        ) from exc
    except VoiceNotFoundError as exc:
        raise InvalidEngineVoiceError(
            f"Unknown Voicefolio managed voice {ref!r}: {exc}",
            code="managed_voice.invalid_source",
            details={"resolver": resolver, "ref": ref},
        ) from exc
    except VoicefolioError as exc:
        raise EngineBackendError(
            f"Voicefolio could not resolve managed voice {ref!r}: {exc}",
            code="managed_voice.resolve_failed",
            details={"resolver": resolver, "ref": ref},
            native_error_type=type(exc).__name__,
        ) from exc
    except ValueError as exc:
        raise InvalidEngineVoiceError(
            f"Invalid Voicefolio managed voice {ref!r}: {exc}",
            code="managed_voice.invalid_source",
            details={"resolver": resolver, "ref": ref},
        ) from exc

    if isinstance(resolved, _MaterializedVoice):
        source_voice = resolved.source
        entry = source_voice.entry
        source_sha256 = source_voice.sha256
        resolved_sha256 = resolved.sha256
        renderer_path = Path(resolved.path)
    else:
        entry = resolved.entry
        source_sha256 = resolved.sha256
        resolved_sha256 = resolved.sha256
        renderer_path = Path(resolved.path)
    return ResolvedManagedVoice(
        resolver=resolver,
        ref=str(entry.ref),
        path=renderer_path,
        source_sha256=source_sha256,
        resolved_sha256=resolved_sha256,
        source_format=entry.format,
        source_revision=entry.source_revision,
        source_repository=entry.source_repository,
        source_path=entry.source_path,
        license=entry.license,
        dataset=entry.dataset,
        variant=entry.variant,
    )


def _source_identity(source: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(source, Mapping) or source.get("kind") != "managed_reference":
        raise InvalidEngineVoiceError(
            "Managed voice source must have kind='managed_reference'.",
            code="managed_voice.invalid_source",
        )

    resolver = source.get("resolver")
    if not isinstance(resolver, str) or not resolver.strip():
        raise InvalidEngineVoiceError(
            "Managed voice source requires a non-empty resolver.",
            code="managed_voice.resolver_missing",
            details={"resolver": resolver},
        )
    if resolver != VOICEFOLIO_RESOLVER:
        raise InvalidEngineOptionError(
            f"Unsupported managed voice resolver {resolver!r}.",
            code="managed_voice.resolver_unsupported",
            details={"resolver": resolver},
        )

    ref = source.get("value")
    if not isinstance(ref, str) or not ref.strip():
        raise InvalidEngineVoiceError(
            "Voicefolio managed voice source requires a non-empty value.",
            code="managed_voice.invalid_source",
            details={"resolver": resolver},
        )
    for name in _MANAGED_PIN_FIELDS:
        value = source.get(name)
        if not isinstance(value, str) or not value.strip():
            raise InvalidEngineVoiceError(
                f"Voicefolio managed voice source requires non-empty {name}.",
                code="managed_voice.invalid_source",
                details={"resolver": resolver, "ref": ref, "field": name},
            )
    return resolver, ref


def _load_catalog(
    catalog_paths: Iterable[str | Path] | str | Path | None,
    *,
    resolver: str,
    ref: str,
) -> Catalog:
    if catalog_paths is None:
        paths: tuple[str | Path, ...] = ()
    elif isinstance(catalog_paths, (str, Path)):
        paths = (catalog_paths,)
    else:
        paths = tuple(catalog_paths)
    try:
        return voicefolio.load_catalog(extra_paths=paths)
    except VoicefolioError as exc:
        raise InvalidEngineVoiceError(
            f"Voicefolio catalog is unavailable for {ref!r}: {exc}",
            code="managed_voice.catalog_invalid",
            details={"resolver": resolver, "ref": ref},
        ) from exc


__all__ = [
    "ResolvedManagedVoice",
    "resolve_managed_voice",
    "validate_managed_voice",
]
