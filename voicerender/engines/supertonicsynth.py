"""Request-centric adapter for the public SupertonicSynth API."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from ..errors import (
    EmptySpeechTextError,
    EngineBackendError,
    EngineError,
    EngineSynthesisError,
    InvalidEngineLanguageError,
    InvalidEngineModelError,
    InvalidEngineOptionError,
    InvalidEngineVoiceError,
    InvalidSpeechRequestError,
    SpeechRequestTooLongError,
    UnsupportedSynthesisFeatureError,
)
from .api_probe import SUPPORTED_REQUEST_API_VERSION, EngineApiProbe, probe_public_api
from .base import (
    EngineCapabilities,
    EngineSelection,
    RenderedSpeech,
    RenderProfileFragment,
    RequestMeasure,
    SpeechRequest,
    distribution_version,
    validate_rendered_speech,
    voice_level_metadata,
)
from .catalog import CatalogRequest, SynthesisTarget, TargetVoice
from .metadata import (
    language_base,
    normalize_gender,
    normalize_language_key,
    normalize_locale_tag,
)
from .profiles import make_renderprofile_fragment

ENGINE_ID = "supertonic"
PACKAGE_NAME = "supertonicsynth"
SUPERTONIC_OPTION_NAMES = frozenset(
    {
        "steps",
        "seed",
        "cache_dir",
        "providers",
        "provider_options",
        "session_options",
        "voice_level",
        "force_download",
    }
)


class SupertonicSelectionError(ValueError):
    """A selection error with a stable plan diagnostic."""

    def __init__(self, code: str, message: str, field: str) -> None:
        super().__init__(message)
        self.diagnostic_code = code
        self.diagnostic_field = field


def _module() -> Any:
    import supertonicsynth

    return supertonicsynth


def _effective_language(value: str | None) -> str:
    normalized = normalize_locale_tag(value or "en-us")
    return language_base(normalized) or normalize_language_key(value or "en-us")


def _target_voice(voice: Any, target_languages: tuple[str, ...]) -> TargetVoice:
    voice_language = language_base(getattr(voice, "language", None))
    locale = normalize_locale_tag(getattr(voice, "locale", None))
    supported_languages = tuple(
        dict.fromkeys(
            normalized
            for raw in (getattr(voice, "languages", ()) or target_languages)
            if (normalized := normalize_locale_tag(raw))
            and language_base(normalized) not in {"na", "unknown"}
        )
    )
    if voice_language in {"na", "unknown"}:
        voice_language = ""
    if language_base(locale) in {"na", "unknown"}:
        locale = ""
    raw_label = str(getattr(voice, "language_label", "") or "").strip()
    language_label = "" if raw_label.casefold() in {"na", "unknown"} else raw_label
    descriptive_fallback = (
        supported_languages[0] if len(supported_languages) == 1 else "unknown"
    )
    if not language_label and len(supported_languages) > 1:
        language_label = "Multilingual"
    return TargetVoice(
        id=str(voice.id),
        gender=normalize_gender(getattr(voice, "gender", None)),
        language=voice_language or descriptive_fallback,
        locale=locale or descriptive_fallback,
        language_label=language_label or descriptive_fallback,
        languages=supported_languages,
    )


def _target_from_model(model: Any) -> SynthesisTarget:
    languages = tuple(
        dict.fromkeys(
            base
            for language in model.languages
            if (base := language_base(language)) and base != "na"
        )
    )
    voices = tuple(str(voice.id) for voice in model.voices)
    voice_details = tuple(_target_voice(voice, languages) for voice in model.voices)
    metadata = dict(model.metadata)
    metadata.update(
        {
            "source": PACKAGE_NAME,
            "ref": model.ref,
            "version": model.version,
            "source_revision": model.source_revision,
            "max_input_tokens": model.max_input_tokens,
        }
    )
    return SynthesisTarget(
        engine=ENGINE_ID,
        id=model.id,
        display_name=model.display_name,
        languages=languages,
        status="ready" if model.runtime_available else "runtime_unavailable",
        runtime_available=bool(model.runtime_available),
        sample_rate=model.sample_rate,
        voices=voices,
        voice_details=voice_details,
        default_voice=model.default_voice,
        aliases=tuple(model.aliases),
        capabilities=frozenset({"predefined_voice"}),
        metadata=metadata,
    )


def _target_catalog(
    *, language: str | None, offline: bool, refresh: bool, cache_dir: Any = None
) -> tuple[SynthesisTarget, ...]:
    module = _module()
    requested = _effective_language(language) if language is not None else None
    models = module.discover_models(
        language=requested,
        offline=offline,
        refresh=refresh,
        cache_dir=cache_dir,
    )
    return tuple(_target_from_model(model) for model in models)


def _find_target(
    targets: tuple[SynthesisTarget, ...], target_id: str
) -> SynthesisTarget | None:
    return next(
        (
            target
            for target in targets
            if target.id == target_id or target_id in target.aliases
        ),
        None,
    )


def _translate_error(
    error: Exception,
    selection: EngineSelection,
    request: SpeechRequest | None = None,
    *,
    opening: bool = False,
) -> EngineError:
    module = _module()
    error_map: tuple[tuple[str, type[EngineError]], ...] = (
        ("SynthesisInputTooLongError", SpeechRequestTooLongError),
        ("EmptyTextError", EmptySpeechTextError),
        ("InvalidLanguageError", InvalidEngineLanguageError),
        ("InvalidVoiceStyleError", InvalidEngineVoiceError),
        ("BundleNotFoundError", InvalidEngineModelError),
        ("InvalidGenerationConfigError", InvalidEngineOptionError),
        ("InvalidRequestError", InvalidSpeechRequestError),
        ("ClosedRuntimeError", EngineBackendError),
        ("ModelInferenceError", EngineBackendError),
        ("SessionCreationError", EngineBackendError),
        ("RuntimeCapabilityError", EngineBackendError),
        ("OfflineAssetError", EngineBackendError),
        ("AssetDownloadError", EngineBackendError),
        ("AssetCacheError", EngineBackendError),
        ("CatalogUnavailableError", EngineBackendError),
        ("OptionalDependencyError", EngineBackendError),
        ("AssetError", EngineBackendError),
        ("SupertonicSynthError", EngineBackendError),
    )
    error_type: type[EngineError] = EngineBackendError
    for native_name, mapped_type in error_map:
        native_type = getattr(module, native_name, None)
        if isinstance(native_type, type) and isinstance(error, native_type):
            error_type = mapped_type
            break
    else:
        if opening and isinstance(error, (TypeError, ValueError)):
            error_type = InvalidEngineOptionError
        elif isinstance(error, (TypeError, ValueError)):
            error_type = InvalidSpeechRequestError

    context: dict[str, Any] = {
        "engine": ENGINE_ID,
        "engine_version": distribution_version(PACKAGE_NAME),
        "target_id": selection.target_id,
        "language": request.language if request is not None else selection.language,
        "voice": request.voice if request is not None else selection.voice,
        "speaker": request.speaker if request is not None else selection.speaker,
        "request_id": request.id if request is not None else None,
        "native_error_type": type(error).__name__,
    }
    if error_type is SpeechRequestTooLongError:
        context.update(
            amount=getattr(error, "token_count", None),
            maximum=getattr(error, "max_tokens", None),
            unit="model_tokens",
            text_length=getattr(
                error, "text_length", len(request.text) if request else None
            ),
            source="supertonicsynth.atomic",
        )
    return error_type(str(error), **context)


def _native_request(request: SpeechRequest, language: str, module: Any) -> Any:
    return module.SynthesisRequest(id=request.id, text=request.text, language=language)


class SupertonicSynthEngineSession:
    """Adapt one VoiceRender request to one atomic SupertonicSynth inference."""

    def __init__(
        self,
        runtime: Any,
        selection: EngineSelection,
        generation: Any,
        voice_level: Any,
    ) -> None:
        self._runtime = runtime
        self._selection = selection
        self._generation = generation
        self._voice_level = voice_level

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        module = _module()
        language = _effective_language(request.language)
        try:
            measured = self._runtime.measure_request(
                _native_request(request, language, module)
            )
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_error(exc, self._selection, request) from exc
        return RequestMeasure(
            fits=measured.fits,
            amount=measured.amount,
            maximum=measured.maximum,
            unit="model_tokens",
            source="supertonicsynth.measure_request",
            details={"language": language},
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        if request.tokens:
            raise UnsupportedSynthesisFeatureError(
                "SupertonicSynth does not support linguistic tokens",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        if request.pronunciation_overrides:
            raise UnsupportedSynthesisFeatureError(
                "SupertonicSynth does not support pronunciation overrides",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        if request.whole_request_phonemes is not None:
            raise UnsupportedSynthesisFeatureError(
                "SupertonicSynth does not support whole-request phonemes",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        if request.speaker is not None:
            raise UnsupportedSynthesisFeatureError(
                "SupertonicSynth does not support speaker selection",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
            )

        module = _module()
        language = _effective_language(request.language)
        voice = request.voice or self._selection.voice
        try:
            native_request = _native_request(request, language, module)
            result = self._runtime.synthesize(
                native_request,
                voice=voice,
                config=self._generation,
                voice_level=self._voice_level,
            )
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_error(exc, self._selection, request) from exc

        metadata = dict(result.metadata or {})
        application = metadata.get("voice_level")
        metadata.update(
            {
                "engine_language": language,
                "bundle_id": self._selection.target_id,
                "source_revision": self._selection.metadata.get("source_revision"),
                "voice": voice,
                "voice_level": voice_level_metadata(
                    application,
                    self._voice_level.mode,
                    identity_keys=("calibration_key",),
                    revision_keys=("catalog_revision",),
                ),
            }
        )
        rendered = RenderedSpeech(
            id=request.id,
            audio=result.audio,
            sample_rate=result.sample_rate,
            warnings=tuple(result.warnings),
            word_timings=(),
            metadata=metadata,
        )
        return validate_rendered_speech(
            request,
            rendered,
            engine=ENGINE_ID,
            engine_version=distribution_version(PACKAGE_NAME),
            target_id=self._selection.target_id,
        )


class SupertonicSynthEngineAdapter:
    """Supertonic target discovery, selection, and strict request adaptation."""

    id = ENGINE_ID
    package_name = PACKAGE_NAME

    def version(self) -> str | None:
        return distribution_version(self.package_name)

    def probe_api(self) -> EngineApiProbe:
        return probe_public_api(
            engine=self.id,
            package=self.package_name,
            required_symbols=(
                "SupertonicRuntime",
                "SynthesisRequest",
                "GenerationConfig",
                "AtomicSynthesisResult",
                "RequestMeasure",
                "SynthesisInputTooLongError",
                "VoiceLevelConfig",
                "DiscoveredModel",
                "discover_models",
                "runtime_identity",
            ),
            required_methods={
                "SupertonicRuntime": (
                    "from_pretrained",
                    "measure_request",
                    "synthesize",
                    "close",
                ),
                "__module__": ("discover_models", "runtime_identity"),
            },
            expected_api_version=SUPPORTED_REQUEST_API_VERSION,
        )

    def compatible_api(self) -> bool:
        return self.probe_api().compatible

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            voice_binding_namespace=self.id,
            voice_binding_scope="target",
            option_names=SUPERTONIC_OPTION_NAMES,
            supports_named_voices=True,
            supports_reference_voice=False,
            supports_speakers=False,
            supports_pronunciation_overrides=False,
            supports_linguistic_tokens=False,
            supports_whole_request_phonemes=False,
            supports_qualities=False,
            supports_voice_level_calibration=True,
            supports_request_measurement=True,
        )

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            frontend="supertonicsynth",
            pronunciation="unsupported",
            capacity_key="max_input_tokens",
            capacity_metric="model_tokens",
        )

    def discover(self, request: CatalogRequest) -> tuple[SynthesisTarget, ...]:
        return _target_catalog(
            language=request.language,
            offline=request.offline,
            refresh=request.refresh,
        )

    def resolve(self, request: Any) -> tuple[EngineSelection, tuple[Any, ...]]:
        module = _module()
        options = dict(getattr(request, "options", {}) or {})
        engine_options = dict(getattr(request, "engine_options", {}) or {})
        unknown = sorted(set(engine_options) - SUPERTONIC_OPTION_NAMES)
        if unknown:
            raise SupertonicSelectionError(
                "supertonic.invalid_option",
                f"Unsupported SupertonicSynth option(s): {', '.join(unknown)}.",
                "synthesis.engine_options",
            )
        options.update(engine_options)
        target_id = str(getattr(request, "target_id", None) or module.DEFAULT_MODEL)
        language = normalize_locale_tag(getattr(request, "language", None) or "en-us")
        offline = bool(getattr(request, "offline", False))
        refresh = bool(getattr(request, "refresh", False))
        try:
            targets = _target_catalog(
                language=None,
                offline=offline,
                refresh=refresh,
                cache_dir=options.get("cache_dir"),
            )
        except Exception as exc:
            raise SupertonicSelectionError(
                "supertonic.catalog_unavailable",
                f"SupertonicSynth model catalog is unavailable: {exc}",
                "synthesis.engine",
            ) from exc
        target = _find_target(targets, target_id)
        if target is not None:
            target_id = target.id
        voice = (
            getattr(request, "voice", None)
            or options.pop("voice", None)
            or (target.default_voice if target is not None else None)
            or module.DEFAULT_VOICE
        )
        generation_defaults = module.GenerationConfig()
        speed_value = options.get("speed", generation_defaults.speed)
        steps_value = options.get("steps", generation_defaults.steps)
        seed_value = options.get("seed", generation_defaults.seed)
        try:
            generation = module.GenerationConfig(
                steps=steps_value,
                speed=speed_value,
                seed=seed_value,
            )
            voice_level = module.VoiceLevelConfig(
                mode=options.get("voice_level", "off")
            )
        except (TypeError, ValueError) as exc:
            raise SupertonicSelectionError(
                "supertonic.invalid_option",
                f"Invalid SupertonicSynth generation options: {exc}",
                "synthesis.engine_options",
            ) from exc
        selected_options = {
            name: options[name] for name in SUPERTONIC_OPTION_NAMES if name in options
        }
        selected_options.update(
            {
                "steps": generation.steps,
                "speed": generation.speed,
                "seed": generation.seed,
                "voice_level": voice_level.mode,
            }
        )
        selection = EngineSelection(
            engine=self.id,
            target_id=target_id,
            language=language or "en-us",
            voice=str(voice),
            options=selected_options,
            metadata=(
                {
                    "source_revision": target.metadata.get("source_revision"),
                    "backing_ref": target.metadata.get("ref"),
                    "languages": target.languages,
                    "max_input_tokens": target.metadata.get("max_input_tokens"),
                }
                if target is not None
                else {}
            ),
            offline=offline,
            refresh=refresh,
        )
        return selection, ()

    def validate_selection(self, selection: EngineSelection) -> tuple[Any, ...]:
        from .base import EngineDiagnostic

        module = _module()
        try:
            targets = _target_catalog(
                language=None,
                offline=selection.offline,
                refresh=selection.refresh,
                cache_dir=selection.options.get("cache_dir"),
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            module.SupertonicSynthError,
        ) as exc:
            return (
                EngineDiagnostic(
                    code="supertonic.catalog_unavailable",
                    severity="error",
                    message=f"SupertonicSynth model catalog is unavailable: {exc}",
                    field="synthesis.engine",
                ),
            )
        target = _find_target(targets, selection.target_id)
        if target is None:
            return (
                EngineDiagnostic(
                    code="supertonic.model_not_found",
                    severity="error",
                    message=f"SupertonicSynth model {selection.target_id!r} was not found.",
                    field="render.target.id",
                ),
            )
        diagnostics = []
        language = _effective_language(selection.language)
        if language not in target.languages:
            diagnostics.append(
                EngineDiagnostic(
                    code="supertonic.language_unsupported",
                    severity="error",
                    message=f"SupertonicSynth model {target.id!r} does not support {language!r}.",
                    field="render.target.language",
                )
            )
        if selection.voice not in target.voices:
            diagnostics.append(
                EngineDiagnostic(
                    code="supertonic.voice_unavailable",
                    severity="error",
                    message=f"Voice {selection.voice!r} is not available on {target.id!r}.",
                    field="render.target.voice",
                )
            )
        return tuple(diagnostics)

    def target_metadata(self, selection: EngineSelection) -> Mapping[str, Any]:
        module = _module()
        try:
            targets = _target_catalog(
                language=None,
                offline=selection.offline,
                refresh=selection.refresh,
                cache_dir=selection.options.get("cache_dir"),
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            module.SupertonicSynthError,
        ):
            return {}
        target = _find_target(targets, selection.target_id)
        if target is None:
            return {}
        return {
            **dict(target.metadata),
            "languages": target.languages,
            "voices": target.voices,
            "default_voice": target.default_voice,
            "sample_rate": target.sample_rate,
            "aliases": target.aliases,
            "source_revision": target.metadata.get("source_revision"),
            "max_input_tokens": target.metadata.get("max_input_tokens"),
        }

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> Mapping[str, Any]:
        module = _module()
        language = _effective_language(selection.language)
        voice_ref = f"supertonic:{selection.target_id}/{selection.voice}"
        mode = str(selection.options.get("voice_level", "off"))
        calibration_revision = (
            module.default_voice_calibration().revision
            if mode == "calibrated"
            else None
        )
        options = {
            key: selection.options[key]
            for key in (
                "steps",
                "speed",
                "seed",
                "providers",
                "provider_options",
                "session_options",
            )
            if key in selection.options
        }
        runtime_identity = module.runtime_identity()
        return {
            "engine": self.id,
            "engine_version": self.version(),
            "engine_identity": dict(runtime_identity),
            "target_id": selection.target_id,
            "backing_ref": selection.metadata.get("backing_ref"),
            "source_revision": selection.metadata.get("source_revision"),
            "voice": selection.voice,
            "voice_ref": voice_ref,
            "language": language,
            "options": options,
            "voice_level": {
                "mode": mode,
                "calibration_identity": f"{voice_ref}@{language}"
                if mode == "calibrated"
                else None,
                "catalog_revision": calibration_revision,
            },
        }

    def open(
        self, selection: EngineSelection
    ) -> AbstractContextManager[SupertonicSynthEngineSession]:
        module = _module()
        options = dict(selection.options)
        try:
            generation = module.GenerationConfig(
                steps=options.get("steps", module.GenerationConfig().steps),
                speed=options.get("speed", module.GenerationConfig().speed),
                seed=options.get("seed"),
            )
            voice_level = module.VoiceLevelConfig(
                mode=options.get("voice_level", "off")
            )
            runtime = module.SupertonicRuntime.from_pretrained(
                selection.target_id,
                cache_dir=options.get("cache_dir"),
                offline=selection.offline,
                refresh_catalog=selection.refresh,
                force_download=bool(options.get("force_download", False)),
                providers=options.get("providers"),
                provider_options=options.get("provider_options"),
                session_options=options.get("session_options"),
            )
            actual_id = getattr(runtime.bundle, "bundle_id", None)
            actual_revision = getattr(runtime.bundle, "source_revision", None)
            expected_revision = selection.metadata.get("source_revision")
            if actual_id != selection.target_id or (
                expected_revision is not None and actual_revision != expected_revision
            ):
                runtime.close()
                raise EngineBackendError(
                    "Opened SupertonicSynth bundle differs from the planned target revision.",
                    engine=self.id,
                    engine_version=self.version(),
                    target_id=selection.target_id,
                    code="supertonic.target_changed",
                    details={
                        "actual_target_id": actual_id,
                        "expected_source_revision": expected_revision,
                        "actual_source_revision": actual_revision,
                    },
                )
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_error(exc, selection, opening=True) from exc

        @contextmanager
        def session() -> Any:
            try:
                yield SupertonicSynthEngineSession(
                    runtime, selection, generation, voice_level
                )
            finally:
                runtime.close()

        return session()


__all__ = [
    "ENGINE_ID",
    "PACKAGE_NAME",
    "SUPERTONIC_OPTION_NAMES",
    "SupertonicSynthEngineAdapter",
    "SupertonicSynthEngineSession",
]
