"""Request-centric PyKokoro adapter."""

from __future__ import annotations

import importlib.metadata
import logging
import re
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
from ..models import ModelDiscoveryError
from .api_probe import SUPPORTED_REQUEST_API_VERSION, EngineApiProbe, probe_public_api
from .base import (
    EngineCapabilities,
    EngineSelection,
    PronunciationSpan,
    RenderedSpeech,
    RequestMeasure,
    SpeechRequest,
    SpeechToken,
    SpeechWordTiming,
    distribution_version,
    validate_rendered_speech,
    voice_level_metadata,
)
from .catalog import SynthesisTarget, TargetVoice
from .metadata import normalize_language_key
from .registry import normalize_engine_id

logger = logging.getLogger(__name__)


PYKOKORO_REQUIRED = ">=0.10.0,<0.11"
_DISCOVERY_PREFERENCES = {"auto", "github", "huggingface", "upstream"}


def validate_discovery_preference(preference: str) -> str:
    if preference not in _DISCOVERY_PREFERENCES:
        choices = ", ".join(sorted(_DISCOVERY_PREFERENCES))
        raise ModelDiscoveryError(
            f"Unknown discovery preference {preference!r}; choose {choices}.",
            code="pykokoro.invalid_options",
        )
    return preference


def _version_supported(version: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    patch = int(match.group(3) or 0)
    return (major, minor, patch) >= (0, 10, 0) and (major, minor) == (0, 10)


def _package_metadata() -> str | None:
    try:
        return importlib.metadata.version("pykokoro")
    except importlib.metadata.PackageNotFoundError:
        return None


def _pykokoro_module() -> tuple[Any, str, str | None]:
    try:
        import pykokoro
    except Exception as exc:
        raise ModelDiscoveryError(
            f"Unable to import PyKokoro for model discovery: {exc}",
            code="pykokoro.import_failed",
            distribution_version=_package_metadata(),
        ) from exc
    module_version = str(getattr(pykokoro, "__version__", "unknown"))
    return pykokoro, module_version, _package_metadata()


def _pykokoro_discovery() -> Any:
    pykokoro, version, distribution_version = _pykokoro_module()
    module_path = getattr(pykokoro, "__file__", None)
    if not _version_supported(version):
        raise ModelDiscoveryError(
            "Installed PyKokoro does not support VoiceRender's model-discovery contract "
            f"(module version: {version}; distribution version: {distribution_version or 'unknown'}; "
            f"required: {PYKOKORO_REQUIRED}).",
            code="pykokoro.version_unsupported",
            installed_version=version,
            distribution_version=distribution_version,
            module_version=version,
            module_path=str(module_path) if module_path else None,
        )
    try:
        discovery = pykokoro.discover_models
    except AttributeError as exc:
        raise ModelDiscoveryError(
            "Installed PyKokoro does not provide the model-discovery API required by "
            f"VoiceRender 0.1.0 (module version: {version}; distribution version: "
            f"{distribution_version or 'unknown'}; module: {module_path or 'unknown'}; "
            f"required: {PYKOKORO_REQUIRED} with discover_models).",
            code="pykokoro.discovery_api_missing",
            installed_version=version,
            distribution_version=distribution_version,
            module_version=version,
            module_path=str(module_path) if module_path else None,
        ) from exc
    except (ImportError, ModuleNotFoundError) as exc:
        missing = getattr(exc, "name", None)
        detail = f" (missing dependency: {missing})" if missing else ""
        raise ModelDiscoveryError(
            f"PyKokoro's public discovery API could not be imported{detail}: {exc}",
            code="pykokoro.discovery_api_import_failed",
            installed_version=version,
            distribution_version=distribution_version,
            module_version=version,
            module_path=str(module_path) if module_path else None,
            missing_dependency=missing,
        ) from exc
    except Exception as exc:
        raise ModelDiscoveryError(
            f"PyKokoro's public discovery API could not be imported: {exc}",
            code="pykokoro.discovery_api_import_failed",
            installed_version=version,
            distribution_version=distribution_version,
            module_version=version,
            module_path=str(module_path) if module_path else None,
        ) from exc
    if not callable(discovery):
        raise ModelDiscoveryError(
            f"Installed PyKokoro exposes a non-callable discover_models value "
            f"(module version: {version}; module: {module_path or 'unknown'}).",
            code="pykokoro.discovery_api_missing",
            installed_version=version,
            distribution_version=distribution_version,
            module_version=version,
            module_path=str(module_path) if module_path else None,
        )
    return discovery


def _registry_error(exc: Exception) -> ModelDiscoveryError:
    message = str(exc) or exc.__class__.__name__
    name = exc.__class__.__name__.lower()
    code = (
        "pykokoro.registry_invalid"
        if "invalid" in name or "schema" in message.lower()
        else "pykokoro.registry_unavailable"
    )
    return ModelDiscoveryError(message, code=code)


def _target_from_capabilities(capabilities: Any) -> SynthesisTarget:
    languages = tuple(capabilities.languages)
    fallback = languages[0] if languages else "unknown"
    indexed: dict[str, TargetVoice] = {}
    for detail in tuple(getattr(capabilities, "voice_details", ()) or ()):
        name = getattr(detail, "name", None)
        if (
            not isinstance(name, str)
            or name not in capabilities.voices
            or name in indexed
        ):
            raise ValueError(
                f"voice_details contains invalid or duplicate voice {name!r}"
            )
        indexed[name] = TargetVoice(
            id=name,
            gender=str(getattr(detail, "gender", "unknown")),
            language=str(getattr(detail, "language", fallback)),
            locale=str(getattr(detail, "locale", fallback)),
            language_label=str(getattr(detail, "language_label", fallback)),
        )
    voice_details = tuple(
        indexed.get(
            voice,
            TargetVoice(
                id=voice, language=fallback, locale=fallback, language_label=fallback
            ),
        )
        for voice in capabilities.voices
    )
    provider = getattr(capabilities, "provider", None)
    return SynthesisTarget(
        engine=normalize_engine_id(getattr(capabilities, "engine", "kokoro")),
        id=capabilities.model_id,
        display_name=capabilities.model_id,
        languages=languages,
        status=capabilities.status,
        runtime_available=capabilities.runtime_available,
        sample_rate=getattr(capabilities, "sample_rate", None),
        voices=tuple(capabilities.voices),
        voice_details=voice_details,
        default_voice=capabilities.default_voice,
        qualities=tuple(capabilities.qualities),
        aliases=getattr(capabilities, "aliases", ()),
        capabilities=frozenset({"named_voices", "pronunciation_overrides", "lexicons"}),
        metadata={
            "source": capabilities.source,
            "g2p_backend": capabilities.g2p_backend,
            "frontend": capabilities.frontend,
            "lexicons": capabilities.lexicons,
            "experimental": capabilities.experimental,
            "redistribution_allowed": capabilities.redistribution_allowed,
            "distribution_id": getattr(capabilities, "distribution_id", None),
            "provider": provider,
            "distribution_provider": getattr(
                capabilities, "distribution_provider", provider
            ),
            "sample_rate": getattr(capabilities, "sample_rate", None),
            "max_tokens": getattr(capabilities, "max_tokens", None),
        },
    )


def _discover_targets(
    *,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    preference: str = "auto",
) -> tuple[SynthesisTarget, ...]:
    from ..models import language_matches

    if offline and refresh:
        raise ModelDiscoveryError(
            "--offline and --refresh cannot be combined",
            code="pykokoro.invalid_options",
        )
    preference = validate_discovery_preference(preference)
    try:
        result = _pykokoro_discovery()(
            offline=offline, refresh=refresh, preference=preference
        )
    except ModelDiscoveryError:
        raise
    except Exception as exc:
        raise _registry_error(exc) from exc
    try:
        targets = tuple(_target_from_capabilities(item) for item in result.models)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ModelDiscoveryError(
            f"PyKokoro returned invalid model discovery data: {exc}",
            code="pykokoro.registry_invalid",
        ) from exc
    requested = normalize_language_key(language) if language else None
    if requested is not None:
        targets = tuple(
            item for item in targets if language_matches(requested, item.languages)
        )
    return targets


def _translate_pykokoro_error(
    error: Exception,
    selection: EngineSelection,
    request: SpeechRequest | None = None,
    *,
    opening: bool = False,
) -> EngineError:
    import pykokoro

    error_map = (
        ("SynthesisInputTooLongError", SpeechRequestTooLongError),
        ("EmptyTextError", EmptySpeechTextError),
        ("InvalidModelError", InvalidEngineModelError),
        ("InvalidVoiceError", InvalidEngineVoiceError),
        ("InvalidLanguageError", InvalidEngineLanguageError),
        ("InvalidPronunciationError", InvalidSpeechRequestError),
        ("InvalidLinguisticTokensError", InvalidSpeechRequestError),
        ("ConfigurationError", InvalidEngineOptionError),
        ("UnsupportedFeatureError", UnsupportedSynthesisFeatureError),
        ("BackendError", EngineBackendError),
        ("AlignmentError", EngineBackendError),
        ("SynthesisStateError", EngineBackendError),
    )
    error_type: type[EngineError] = EngineBackendError
    for native_name, mapped_type in error_map:
        native_type = getattr(pykokoro, native_name, None)
        if isinstance(native_type, type) and isinstance(error, native_type):
            error_type = mapped_type
            break
    else:
        if opening and isinstance(error, (TypeError, ValueError)):
            error_type = InvalidEngineOptionError
        elif isinstance(error, (TypeError, ValueError)):
            error_type = InvalidSpeechRequestError

    context: dict[str, Any] = {
        "engine": "kokoro",
        "engine_version": distribution_version("pykokoro"),
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
            source="pykokoro.prepare",
        )
    return error_type(str(error), **context)


def _pykokoro_request(request: SpeechRequest, module: Any) -> Any:
    return module.SynthesisRequest(
        id=request.id,
        text=request.text,
        language=request.language,
        voice=request.voice,
        pronunciation_overrides=tuple(
            _pykokoro_override(item, module) for item in request.pronunciation_overrides
        ),
        tokens=tuple(_pykokoro_token(item, module) for item in request.tokens),
        phonemes=request.whole_request_phonemes,
    )


class PyKokoroEngineSession:
    """Adapt neutral requests to one open KokoroSynthesizer."""

    def __init__(
        self, synthesizer: Any, selection: EngineSelection | None = None
    ) -> None:
        self._synthesizer = synthesizer
        self._selection = selection or EngineSelection(
            engine="kokoro", target_id="unknown", language="und"
        )

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        import pykokoro

        try:
            prepared = self._synthesizer.prepare(_pykokoro_request(request, pykokoro))
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_pykokoro_error(exc, self._selection, request) from exc
        token_ids = tuple(getattr(prepared, "token_ids", ()))
        amount = len(token_ids)
        maximum = self._selection.metadata.get("max_tokens")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            maximum = None
        return RequestMeasure(
            fits=amount <= maximum if maximum is not None else None,
            amount=amount,
            maximum=maximum,
            unit="model_tokens",
            source="pykokoro.prepare",
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        import pykokoro

        try:
            result = self._synthesizer.synthesize(_pykokoro_request(request, pykokoro))
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_pykokoro_error(exc, self._selection, request) from exc

        timings = tuple(
            SpeechWordTiming(
                text=item.text,
                char_start=item.char_start,
                char_end=item.char_end,
                start_sample=item.start_sample,
                end_sample=item.end_sample,
            )
            for item in result.word_timings
        )
        applications = tuple(getattr(result, "voice_level_applications", ()) or ())
        metadata: dict[str, Any] = {
            "language": result.language,
            "voice": result.voice,
            "phonemes": result.phonemes,
            "token_ids": tuple(result.token_ids),
            "voice_level": voice_level_metadata(
                applications[0] if applications else None,
                str(self._selection.options.get("voice_level", "off")),
                identity_keys=("key", "calibration_identity"),
                revision_keys=("corpus", "catalog_revision"),
            ),
        }
        if result.trace is not None:
            metadata["trace"] = result.trace
        rendered = RenderedSpeech(
            id=result.id,
            audio=result.audio,
            sample_rate=result.sample_rate,
            warnings=tuple(result.diagnostics),
            word_timings=timings,
            metadata=metadata,
        )
        return validate_rendered_speech(
            request,
            rendered,
            engine="kokoro",
            engine_version=distribution_version("pykokoro"),
            target_id=self._selection.target_id,
        )


def _pykokoro_override(item: PronunciationSpan, module: Any) -> Any:
    return module.PronunciationOverride(
        start=item.start,
        end=item.end,
        phonemes=item.phonemes,
        language=item.language,
    )


def _pykokoro_token(item: SpeechToken, module: Any) -> Any:
    return module.LinguisticToken(
        start=item.start,
        end=item.end,
        text=item.text,
        pos=item.pos,
        tag=item.tag,
        lemma=item.lemma,
        language=item.language,
        morph=item.morph,
    )


class PyKokoroEngineAdapter:
    """PyKokoro implementation of VoiceRender's request-oriented engine contract."""

    id = "kokoro"
    package_name = "pykokoro"

    def version(self) -> str | None:
        return distribution_version(self.package_name)

    def probe_api(self) -> EngineApiProbe:
        return probe_public_api(
            engine=self.id,
            package=self.package_name,
            required_symbols=(
                "KokoroSynthesizer",
                "SynthesisConfig",
                "SynthesisRequest",
                "PronunciationOverride",
                "LinguisticToken",
                "VoiceLevelConfig",
                "SynthesisInputTooLongError",
                "ShortSentenceConfig",
            ),
            required_methods={
                "KokoroSynthesizer": ("prepare", "synthesize", "close"),
            },
            expected_api_version=SUPPORTED_REQUEST_API_VERSION,
        )

    def compatible_api(self) -> bool:
        return self.probe_api().compatible

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            voice_binding_namespace="kokoro",
            option_names=frozenset(
                {
                    "lexicons",
                    "model_source",
                    "quality",
                    "speed",
                    "random_seed",
                    "voice_level",
                    "short_sentence",
                    "g2p_fallback",
                    "lexicon_data_policy",
                    "waveform_validation",
                    "trace",
                    "allow_experimental_frontend",
                }
            ),
            supports_named_voices=True,
            supports_pronunciation_overrides=True,
            pronunciation_alphabets=frozenset({"ipa"}),
            supports_linguistic_tokens=True,
            supports_whole_request_phonemes=True,
            supports_lexicons=True,
            supports_model_sources=True,
            supports_qualities=True,
            supports_live=True,
            supports_timestamps=True,
            supports_voice_level_calibration=True,
            supports_request_measurement=True,
        )

    def discover(self, request: Any) -> tuple[SynthesisTarget, ...]:
        return _discover_targets(
            language=getattr(request, "language", None),
            offline=getattr(request, "offline", False),
            preference=getattr(request, "preference", "auto"),
            refresh=getattr(request, "refresh", False),
        )

    def discover_lexicons(
        self,
        *,
        language: str | None = None,
        model: str | None = None,
        offline: bool = False,
        refresh: bool = False,
        preference: str = "auto",
    ) -> tuple[tuple[Any, ...], Any]:
        from pykokoro import discover_lexicons

        from ..lexicons import LexiconCatalogEntry

        result = discover_lexicons(
            language=language,
            model_variant=model,
            offline=offline,
            refresh=refresh,
            preference=preference,
        )
        entries = tuple(
            LexiconCatalogEntry(
                selector=item.selector,
                engine=self.id,
                language=item.language,
                locale=item.locale,
                asset_id=item.asset_id,
                data_backend=item.data_backend,
                default=item.default,
                installed=item.installed,
                models=item.models,
                model_support=item.model_support,
                display_name=item.display_name,
                phoneme_encoding=item.phoneme_encoding,
                data_version=item.data_version,
            )
            for item in result.lexicons
        )
        return entries, result

    def resolve(self, request: Any) -> tuple[EngineSelection, tuple[Any, ...]]:
        options = dict(getattr(request, "options", {}) or {})
        options.update(dict(getattr(request, "engine_options", {}) or {}))
        return (
            EngineSelection(
                engine=self.id,
                target_id=getattr(request, "target_id", None)
                or options.get("model_variant")
                or "v1.0",
                language=getattr(request, "language", "en-us"),
                voice=getattr(request, "voice", None),
                speaker=getattr(request, "speaker", None),
                options=options,
                offline=bool(getattr(request, "offline", False)),
                refresh=bool(getattr(request, "refresh", False)),
            ),
            (),
        )

    def validate_selection(self, selection: EngineSelection) -> tuple[Any, ...]:
        from ..models import ModelDiscoveryError, get_model_info, language_matches
        from .base import EngineDiagnostic

        try:
            model, _ = get_model_info(
                selection.target_id,
                offline=selection.offline,
                engine=self.id,
                refresh=selection.refresh,
                preference=selection.options.get("model_source", "auto"),
            )
        except (ModelDiscoveryError, ImportError, OSError, ValueError) as exc:
            return (
                EngineDiagnostic(
                    code="pykokoro.model_not_found",
                    severity="error",
                    message=f"PyKokoro target {selection.target_id!r} was not found: {exc}",
                    field="render.target.id",
                ),
            )
        diagnostics: list[Any] = []
        if not language_matches(selection.language, model.languages):
            diagnostics.append(
                EngineDiagnostic(
                    code="pykokoro.language_incompatible",
                    severity="error",
                    message=f"Target {selection.target_id!r} does not support {selection.language!r}.",
                    field="render.target.language",
                )
            )
        if selection.voice is not None and selection.voice not in model.voices:
            diagnostics.append(
                EngineDiagnostic(
                    code="pykokoro.voice_unavailable",
                    severity="error",
                    message=f"Voice {selection.voice!r} is not available on {selection.target_id!r}.",
                    field="render.target.voice",
                )
            )
        quality = selection.options.get("quality")
        if quality is not None and quality not in model.qualities:
            diagnostics.append(
                EngineDiagnostic(
                    code="pykokoro.quality_unavailable",
                    severity="error",
                    message=f"Quality {quality!r} is not available on {selection.target_id!r}.",
                    field="render.options.quality",
                )
            )
        return tuple(diagnostics)

    def target_metadata(self, selection: EngineSelection) -> Mapping[str, Any]:
        from ..models import get_model_info

        try:
            model, _ = get_model_info(
                selection.target_id,
                offline=selection.offline,
                engine=self.id,
                refresh=selection.refresh,
                preference=selection.options.get("model_source", "auto"),
            )
        except (ImportError, OSError, ValueError):
            return {}
        return {
            "languages": tuple(model.languages),
            "voices": tuple(model.voices),
            "sample_rate": model.sample_rate,
            "max_tokens": model.max_tokens,
            "model_source": model.source,
            "model_identity": model.distribution_id or model.id,
            "distribution_id": model.distribution_id,
        }

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> Mapping[str, Any]:
        return {
            "engine": self.id,
            "engine_version": self.version(),
            "target_id": selection.target_id,
            "voice": selection.voice,
            "speaker": selection.speaker,
            "options": dict(selection.options),
            "metadata": dict(selection.metadata),
        }

    def open(
        self, selection: EngineSelection
    ) -> AbstractContextManager[PyKokoroEngineSession]:
        import pykokoro

        options = dict(selection.options)
        try:
            generation = pykokoro.GenerationConfig(
                speed=float(options.get("speed", 1.0)),
                lang=selection.language,
                random_seed=options.get("random_seed"),
            )
            tokenizer = pykokoro.TokenizerConfig(
                lexicons=options.get("lexicons"),
                fallback=options.get("g2p_fallback", "espeak"),
                lexicon_data_policy=options.get("lexicon_data_policy", "auto"),
            )
            short_sentence = options.get("short_sentence")
            short_sentence_config = None
            if short_sentence is not None:
                short_sentence_config = pykokoro.ShortSentenceConfig(
                    enabled=short_sentence != "off",
                    **(
                        {"resolve_mode": short_sentence}
                        if short_sentence != "off"
                        else {}
                    ),
                )
            config = pykokoro.SynthesisConfig(
                voice=selection.voice,
                generation=generation,
                model_quality=options.get("quality"),
                model_source=options.get("model_source"),
                model_variant=selection.target_id,
                tokenizer_config=tokenizer,
                short_sentence_config=short_sentence_config,
                waveform_validation=options.get("waveform_validation", "off"),
                return_trace=bool(options.get("trace", False)),
                allow_experimental_frontend=bool(
                    options.get("allow_experimental_frontend", False)
                ),
                voice_level=pykokoro.VoiceLevelConfig(
                    mode=options.get("voice_level", "off")
                ),
                long_text_split="none",
                long_text_use_spacy=False,
            )
            synthesizer = pykokoro.KokoroSynthesizer(config)
        except Exception as exc:
            raise _translate_pykokoro_error(exc, selection, opening=True) from exc

        @contextmanager
        def session() -> Any:
            try:
                yield PyKokoroEngineSession(synthesizer, selection)
            finally:
                synthesizer.close()

        return session()


__all__ = ["PyKokoroEngineAdapter", "PyKokoroEngineSession"]
