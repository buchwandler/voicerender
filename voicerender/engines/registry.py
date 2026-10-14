"""Lazy registry for VoiceRender-owned request-centric engine adapters."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from ..errors import (
    EngineApiIncompatibleError,
    EngineUnavailableError,
    ProfileError,
)
from .base import (
    EngineAdapter,
    EngineSelection,
    EngineSession,
    RenderedSpeech,
    RequestMeasure,
    SpeechRequest,
)
from .catalog import CatalogRequest, SynthesisTarget

if TYPE_CHECKING:
    from ..capacity import CapacityContext


# Keep this table free of adapter imports: listing engines and importing VoiceRender
# must work in installations that contain none of the optional engine SDKs.
@dataclass(frozen=True, slots=True)
class BuiltinEngineSpec:
    package: str
    module: str
    adapter: str
    supports_reference_voice: bool = False


_ENGINE_MODULES: dict[str, BuiltinEngineSpec] = {
    "inflect": BuiltinEngineSpec(
        "inflectsynth", "voicerender.engines.inflectsynth", "InflectSynthEngineAdapter"
    ),
    "kitten": BuiltinEngineSpec(
        "kittensynth", "voicerender.engines.kittensynth", "KittenSynthEngineAdapter"
    ),
    "kokoro": BuiltinEngineSpec(
        "pykokoro", "voicerender.engines.kokoro", "PyKokoroEngineAdapter", True
    ),
    "piper": BuiltinEngineSpec(
        "pipersynth", "voicerender.engines.pipersynth", "PiperSynthEngineAdapter"
    ),
    "pocket": BuiltinEngineSpec(
        "pocketsynth",
        "voicerender.engines.pocketsynth",
        "PocketSynthEngineAdapter",
        True,
    ),
    "sopro": BuiltinEngineSpec(
        "soprosynth", "voicerender.engines.soprosynth", "SoproSynthEngineAdapter", True
    ),
    "supertonic": BuiltinEngineSpec(
        "supertonicsynth",
        "voicerender.engines.supertonicsynth",
        "SupertonicSynthEngineAdapter",
    ),
}
_ADAPTERS: dict[str, EngineAdapter] = {}


def normalize_engine_id(engine: str) -> str:
    """Normalize a public engine identifier without importing its SDK."""
    normalized = engine.strip().casefold()
    aliases = {
        "pykokoro": "kokoro",
        "pipersynth": "piper",
        "pocketsynth": "pocket",
        "soprosynth": "sopro",
    }
    return aliases.get(normalized, normalized)


def register_adapter(adapter: EngineAdapter) -> None:
    """Register a VoiceRender-compatible third-party adapter."""
    engine_id = normalize_engine_id(adapter.id)
    if not engine_id or engine_id in _ENGINE_MODULES or engine_id in _ADAPTERS:
        raise ProfileError(
            f"Invalid or already registered engine adapter: {adapter.id!r}"
        )
    required = ("probe_api", "capabilities", "discover", "resolve", "open", "version")
    missing = [name for name in required if not callable(getattr(adapter, name, None))]
    if missing:
        raise ProfileError(
            f"Engine adapter {adapter.id!r} is missing required methods: {', '.join(missing)}"
        )
    _ADAPTERS[engine_id] = adapter


def get_adapter(engine: str) -> EngineAdapter:
    """Return a request-centric adapter, importing its implementation lazily."""
    engine_id = normalize_engine_id(engine)
    if engine_id in _ADAPTERS:
        return _ADAPTERS[engine_id]
    try:
        spec = _ENGINE_MODULES[engine_id]
    except KeyError as exc:
        available = ", ".join(sorted((*_ENGINE_MODULES, *_ADAPTERS)))
        raise ProfileError(
            f"Unknown engine {engine!r}; available: {available}"
        ) from exc
    module = importlib.import_module(spec.module)
    adapter = cast(EngineAdapter, getattr(module, spec.adapter)())
    _ADAPTERS[engine_id] = adapter
    return adapter


def list_engines() -> list[dict[str, Any]]:
    """List known engines using distribution metadata only (no SDK imports)."""
    result: list[dict[str, Any]] = []
    for engine_id, spec in sorted(_ENGINE_MODULES.items()):
        try:
            installed_version = importlib.metadata.version(spec.package)
        except importlib.metadata.PackageNotFoundError:
            installed_version = None
        except Exception:  # noqa: BLE001 - Metadata failures must not break engine listing.
            installed_version = None
        result.append(
            {
                "id": engine_id,
                "package": spec.package,
                "installed": installed_version is not None,
                "installed_version": installed_version,
                # Compatibility field used by the older CLI output.
                "reference_audio": spec.supports_reference_voice,
                "supports_reference_voice": spec.supports_reference_voice,
            }
        )
    for engine_id, adapter in sorted(_ADAPTERS.items()):
        if engine_id in _ENGINE_MODULES:
            continue
        capabilities = adapter.capabilities()
        result.append(
            {
                "id": engine_id,
                "package": adapter.package_name,
                "installed": adapter.version() is not None,
                "installed_version": adapter.version(),
                "reference_audio": capabilities.supports_reference_voice,
                "supports_reference_voice": capabilities.supports_reference_voice,
            }
        )
    return sorted(result, key=lambda item: str(item["id"]))


def engine_status(engine: str) -> dict[str, Any]:
    """Probe the installed public API without opening a model or runtime."""
    adapter = get_adapter(engine)
    probe = adapter.probe_api()
    return asdict(probe)


def discover_targets(
    engine: str,
    *,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    preference: str = "auto",
) -> tuple[SynthesisTarget, ...]:
    """Discover synthesis targets through the selected registry adapter."""
    adapter = get_adapter(engine)
    try:
        return adapter.discover(
            CatalogRequest(
                engine=adapter.id,
                language=language,
                offline=offline,
                refresh=refresh,
                preference=preference,
            )
        )
    except ImportError as exc:
        raise EngineUnavailableError(
            f"Engine {adapter.id!r} discovery dependencies are unavailable: {exc}",
            engine=adapter.id,
            native_error_type=type(exc).__name__,
            code="engine.unavailable",
        ) from exc


def resolve_voice(request: Any) -> tuple[EngineSelection, tuple[Any, ...]]:
    """Resolve a neutral engine request through the registered adapter."""
    if not isinstance(request.engine, str) or not request.engine.strip():
        raise ProfileError("An engine identifier is required to resolve a voice")
    adapter = get_adapter(request.engine)
    try:
        selection, diagnostics = adapter.resolve(request)
        validate = getattr(adapter, "validate_selection", None)
        if callable(validate):
            diagnostics = (*diagnostics, *validate(selection))
        return selection, diagnostics
    except ImportError as exc:
        raise EngineUnavailableError(
            f"Engine {adapter.id!r} selection dependencies are unavailable: {exc}",
            engine=adapter.id,
            native_error_type=type(exc).__name__,
            code="engine.unavailable",
        ) from exc


class _BoundSynthesisSession:
    def __init__(self, session: EngineSession, synthesis_identity: Mapping[str, Any]):
        self._session = session
        self.synthesis_identity = dict(synthesis_identity)

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        return self._session.measure(request)

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        return self._session.synthesize(request)


@contextmanager
def open_session(selection: EngineSelection) -> Iterator[EngineSession]:
    """Open a probed session bound to immutable engine and API identity."""
    adapter = get_adapter(selection.engine)
    probe = adapter.probe_api()
    if not probe.compatible:
        details: dict[str, object] = {"probe": asdict(probe)}
        if probe.status == "package_missing":
            raise EngineUnavailableError(
                f"Engine {adapter.id!r} runtime package is unavailable",
                engine=adapter.id,
                native_error_type=probe.error_type,
                details=details,
            )

        raise EngineApiIncompatibleError(
            f"Engine {adapter.id!r} public synthesis API is incompatible",
            engine=adapter.id,
            engine_version=probe.distribution_version,
            native_error_type=probe.error_type,
            details=details,
        )
    try:
        identity = {
            "schema": "voicerender.engine-session-identity.v1",
            "engine": adapter.id,
            "engine_version": adapter.version() or probe.distribution_version,
            "api_probe": asdict(probe),
            "engine_identity": dict(adapter.canonical_synthesis_identity(selection)),
        }
        manager = adapter.open(selection)
    except ImportError as exc:
        raise EngineUnavailableError(
            f"Engine {adapter.id!r} runtime dependencies are unavailable: {exc}",
            engine=adapter.id,
            native_error_type=type(exc).__name__,
            code="engine.unavailable",
        ) from exc
    entered = False
    try:
        with manager as session:
            entered = True
            yield _BoundSynthesisSession(session, identity)
    except ImportError as exc:
        if entered:
            raise
        raise EngineUnavailableError(
            f"Engine {adapter.id!r} runtime dependencies are unavailable: {exc}",
            engine=adapter.id,
            native_error_type=type(exc).__name__,
            code="engine.unavailable",
        ) from exc


def synthesize(
    session: EngineSession,
    request: SpeechRequest,
    *,
    capacity: CapacityContext | None = None,
    protected_ranges: tuple[tuple[int, int], ...] = (),
) -> RenderedSpeech:
    """Fit, synthesize, and merge one request through the shared capacity spine."""
    from ..capacity import CAPACITY_SCHEMA, render_atomic_request
    from ..storage import digest

    result = render_atomic_request(
        session,
        request,
        capacity=capacity,
        protected_ranges=protected_ranges,
    ).result
    engine_identity = getattr(session, "synthesis_identity", None)
    if isinstance(engine_identity, Mapping):
        request_identity = {
            "text": request.text,
            "language": request.language,
            "voice": request.voice,
            "speaker": request.speaker,
            "options": dict(request.options),
            "pronunciation_overrides": [
                asdict(item) for item in request.pronunciation_overrides
            ],
            "tokens": [asdict(item) for item in request.tokens],
            "whole_request_phonemes": request.whole_request_phonemes,
        }
        acoustic_identity = {
            "schema": "voicerender.live-acoustic-identity.v1",
            "engine_identity": dict(engine_identity),
            "request": request_identity,
            "capacity_schema": CAPACITY_SCHEMA,
            "output": {"format": "WAV", "encoding": "PCM_16"},
        }
        result.metadata = {
            **dict(result.metadata),
            "engine_identity": dict(engine_identity),
            "acoustic_identity": {
                **acoustic_identity,
                "sha256": "sha256:" + digest(acoustic_identity),
            },
        }
    return result


__all__ = [
    "discover_targets",
    "engine_status",
    "get_adapter",
    "list_engines",
    "normalize_engine_id",
    "open_session",
    "register_adapter",
    "resolve_voice",
    "synthesize",
]
