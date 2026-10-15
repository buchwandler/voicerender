"""Request-centric adapter for the public InflectSynth package API."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
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
    EngineDiagnostic,
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

ENGINE_ID = "inflect"
PACKAGE_NAME = "inflectsynth"
SUPPORTED_CAPACITY_API_VERSION = 1
INFLECT_OPTION_NAMES = frozenset(
    {
        "speed",
        "variation",
        "seed",
        "voice_level",
        "cache_dir",
        "force_download",
        "catalog_url",
        "providers",
        "provider_options",
        "session_options",
    }
)


class InflectSelectionError(ValueError):
    """A selection error with a stable plan diagnostic."""

    def __init__(
        self, code: str, message: str, field: str = "synthesis.engine"
    ) -> None:
        super().__init__(message)
        self.diagnostic_code = code
        self.diagnostic_field = field


def _module() -> Any:
    """Import the optional engine only when an Inflect operation is requested."""
    import importlib

    return importlib.import_module(PACKAGE_NAME)


def _effective_language(value: str | None) -> str:
    normalized = normalize_locale_tag(value or "en-us")
    return normalized or "en-us"


def _is_english(value: str | None) -> bool:
    normalized = normalize_language_key(value or "en-us")
    return normalized == "english" or language_base(normalized) == "en"


def _target_voice(voice: Any, target_languages: tuple[str, ...]) -> TargetVoice:
    locale = normalize_locale_tag(getattr(voice, "locale", None))
    language = normalize_locale_tag(getattr(voice, "language", None))
    voice_languages = tuple(
        dict.fromkeys(
            language_base(normalize_locale_tag(raw))
            for raw in (getattr(voice, "languages", ()) or target_languages)
            if normalize_locale_tag(raw)
        )
    )
    voice_languages = tuple(item for item in voice_languages if item)
    return TargetVoice(
        id=str(voice.id),
        gender=normalize_gender(getattr(voice, "gender", None)),
        language=language_base(language)
        or (voice_languages[0] if voice_languages else "unknown"),
        locale=locale or (language or "unknown"),
        language_label=str(
            getattr(voice, "language_label", None) or locale or language or "unknown"
        ),
        languages=voice_languages or target_languages,
    )


def _valid_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _target_from_model(model: Any) -> SynthesisTarget:
    normalized_language = normalize_locale_tag(getattr(model, "language", None))
    base_language = language_base(normalized_language)
    languages = (base_language,) if base_language else ()
    voices = tuple(str(voice.id) for voice in (getattr(model, "voices", ()) or ()))
    voice_details = tuple(
        _target_voice(voice, languages)
        for voice in (getattr(model, "voices", ()) or ())
    )
    metadata = dict(getattr(model, "metadata", {}) or {})
    metadata.update(
        {
            "source": PACKAGE_NAME,
            "version": getattr(model, "version", None),
            "source_revision": getattr(model, "source_revision", None),
        }
    )
    max_input_tokens = metadata.get("max_input_tokens")
    if _valid_positive_int(max_input_tokens):
        metadata["max_input_tokens"] = max_input_tokens
    policy_revision = metadata.get("capacity_policy_revision")
    if not isinstance(policy_revision, str) or not policy_revision.strip():
        policy_revision = metadata.get("capacity_revision")
    if isinstance(policy_revision, str) and policy_revision.strip():
        metadata["capacity_policy_revision"] = policy_revision.strip()
    else:
        metadata.pop("capacity_policy_revision", None)
    sample_rate = getattr(model, "sample_rate", None)
    return SynthesisTarget(
        engine=ENGINE_ID,
        id=str(model.id),
        display_name=str(getattr(model, "display_name", None) or model.id),
        languages=languages,
        status="ready"
        if bool(getattr(model, "runtime_available", False))
        else "runtime_unavailable",
        runtime_available=bool(getattr(model, "runtime_available", False)),
        sample_rate=sample_rate,
        voices=voices,
        voice_details=voice_details,
        default_voice=getattr(model, "default_voice", None),
        aliases=tuple(getattr(model, "aliases", ()) or ()),
        capabilities=frozenset({"predefined_voice"}),
        metadata=metadata,
    )


def _target_catalog(
    *,
    language: str | None,
    offline: bool,
    refresh: bool,
    cache_dir: Any = None,
    catalog_url: str | None = None,
) -> tuple[SynthesisTarget, ...]:
    module = _module()
    models = module.discover_models(
        language=language,
        offline=offline,
        refresh=refresh,
        cache_dir=cache_dir,
        catalog_url=catalog_url,
    )
    return tuple(_target_from_model(model) for model in models)


def _find_target(
    targets: tuple[SynthesisTarget, ...], target_id: str
) -> SynthesisTarget | None:
    requested = target_id.casefold()
    return next(
        (
            target
            for target in targets
            if target.id.casefold() == requested
            or any(alias.casefold() == requested for alias in target.aliases)
        ),
        None,
    )


def _capacity_contract(module: Any) -> Mapping[str, Any] | None:
    function = getattr(module, "capacity_api_contract", None)
    if not callable(function):
        return None
    try:
        contract = function()
    except Exception:  # noqa: BLE001 - Public API probing must classify arbitrary contract failures.
        return None
    return contract if isinstance(contract, Mapping) else None


def _contract_field(contract: Any, name: str, default: Any = None) -> Any:
    if isinstance(contract, Mapping):
        return contract.get(name, default)
    return getattr(contract, name, default)


def _request_measurement_contract(module: Any) -> Any:
    factory = getattr(module, "request_api_contract", None)
    if not callable(factory):
        return None
    try:
        return factory()
    except Exception:  # noqa: BLE001 - Contract inspection must fail closed.
        return None


def _supports_request_measurement(module: Any | None = None) -> bool:
    module = module or _module()
    contract = _request_measurement_contract(module)
    voice_type = getattr(module, "InflectVoice", None)
    return (
        _contract_field(contract, "supports_request_measurement") is True
        and _contract_field(contract, "measurement_entrypoint")
        == "InflectVoice.measure_prepared"
        and _contract_field(contract, "measurement_metric") == "model_tokens"
        and callable(getattr(voice_type, "measure_prepared", None))
    )


def _measurement_value(measured: Any, name: str, default: Any = None) -> Any:
    if isinstance(measured, Mapping):
        return measured.get(name, default)
    return getattr(measured, name, default)


def _first_nonnegative_int(*values: Any) -> int | None:
    return next(
        (
            value
            for value in values
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ),
        None,
    )


def _first_positive_int(*values: Any) -> int | None:
    return next(
        (
            value
            for value in values
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        ),
        None,
    )


def _has_invalid_integer(values: tuple[Any, ...], *, positive: bool) -> bool:
    for value in values:
        if value is None:
            continue
        valid = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and (value > 0 if positive else value >= 0)
        )
        if not valid:
            return True
    return False


def _error_context(
    selection: EngineSelection,
    request: SpeechRequest | None,
    error: Exception,
) -> dict[str, Any]:
    return {
        "engine": ENGINE_ID,
        "engine_version": distribution_version(PACKAGE_NAME),
        "target_id": selection.target_id,
        "language": request.language if request is not None else selection.language,
        "voice": request.voice if request is not None else selection.voice,
        "speaker": request.speaker if request is not None else selection.speaker,
        "request_id": request.id if request is not None else None,
        "native_error_type": type(error).__name__,
    }


def _translate_error(
    error: Exception,
    selection: EngineSelection,
    request: SpeechRequest | None = None,
    *,
    opening: bool = False,
) -> EngineError:
    module = _module()
    context = _error_context(selection, request, error)
    errors: tuple[tuple[str, type[EngineError], str | None], ...] = (
        ("SynthesisInputTooLongError", SpeechRequestTooLongError, None),
        ("EmptyTextError", EmptySpeechTextError, None),
        (
            "TextPreparationError",
            InvalidSpeechRequestError,
            "inflect.text_preparation_failed",
        ),
        ("InvalidSpeedError", InvalidEngineOptionError, "inflect.invalid_speed"),
        (
            "InvalidVariationError",
            InvalidEngineOptionError,
            "inflect.invalid_variation",
        ),
        ("InvalidSeedError", InvalidEngineOptionError, "inflect.invalid_seed"),
        (
            "InvalidSynthesisConfigError",
            InvalidEngineOptionError,
            "inflect.invalid_options",
        ),
        ("InvalidVoiceError", InvalidEngineVoiceError, "inflect.voice_unavailable"),
        ("UnsupportedModelError", InvalidEngineModelError, "inflect.model_unsupported"),
        ("CatalogUnavailableError", EngineBackendError, "inflect.catalog_unavailable"),
        ("CatalogDiscoveryError", EngineBackendError, "inflect.catalog_error"),
        (
            "OnnxVoiceContractError",
            EngineBackendError,
            "inflect.runtime_contract_error",
        ),
        ("ModelInferenceError", EngineBackendError, "inflect.inference_failed"),
        ("InflectSynthError", EngineBackendError, "inflect.engine_error"),
    )
    for name, error_type, code in errors:
        native_type = getattr(module, name, None)
        if isinstance(native_type, type) and isinstance(error, native_type):
            details = None
            if error_type is SpeechRequestTooLongError:
                details = {
                    "amount": getattr(error, "token_count", None),
                    "maximum": getattr(error, "max_tokens", None),
                    "unit": "model_tokens",
                    "text_length": getattr(
                        error,
                        "text_length",
                        len(request.text) if request is not None else None,
                    ),
                    "source": "inflectsynth.synthesize_prepared",
                }
                context.update(
                    {key: value for key, value in details.items() if value is not None}
                )
            return error_type(str(error), **context, details=details, code=code)
    if isinstance(error, (TypeError, ValueError)):
        error_type = InvalidEngineOptionError if opening else InvalidSpeechRequestError
        code = "inflect.invalid_options" if opening else "inflect.invalid_request"
        return error_type(str(error), **context, code=code)
    return EngineBackendError(str(error), **context, code="inflect.engine_error")


def _selection_diagnostics(code: str, message: str, field: str) -> tuple[Any, ...]:
    from .base import EngineDiagnostic

    return (
        EngineDiagnostic(code=code, severity="error", message=message, field=field),
    )


class InflectSynthEngineSession:
    """Open one managed Inflect voice and render independent prepared requests."""

    def __init__(self, voice: Any, selection: EngineSelection, config: Any) -> None:
        self._voice = voice
        self._selection = selection
        self._config = config

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        measure_method = getattr(self._voice, "measure_prepared", None)
        if not callable(measure_method):
            return RequestMeasure(
                fits=None,
                amount=None,
                maximum=None,
                unit="unknown",
                source="inflectsynth.no_capacity_measurement",
            )
        try:
            measured = measure_method(request.text)
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_error(exc, self._selection, request) from exc

        amount_values = (
            _measurement_value(measured, "amount"),
            _measurement_value(measured, "token_count"),
        )
        maximum_values = (
            _measurement_value(measured, "maximum"),
            _measurement_value(measured, "max_tokens"),
        )
        amount = _first_nonnegative_int(*amount_values)
        maximum = _first_positive_int(*maximum_values)
        raw_fits = _measurement_value(measured, "fits")
        fits = raw_fits if isinstance(raw_fits, bool) else None

        invalid_fields: list[str] = []
        if amount is None and _has_invalid_integer(amount_values, positive=False):
            invalid_fields.append("amount")
        if maximum is None and _has_invalid_integer(maximum_values, positive=True):
            invalid_fields.append("maximum")
        if raw_fits is not None and not isinstance(raw_fits, bool):
            invalid_fields.append("fits")

        if invalid_fields:
            return RequestMeasure(
                fits=None,
                amount=amount,
                maximum=maximum,
                unit="model_tokens" if amount is not None else "unknown",
                source="inflectsynth.invalid_measurement",
                details={"invalid_fields": invalid_fields},
            )
        discovered_maximum = self._selection.metadata.get("max_input_tokens")
        if _valid_positive_int(discovered_maximum) and maximum != discovered_maximum:
            return RequestMeasure(
                fits=None,
                amount=amount,
                maximum=maximum,
                unit="model_tokens" if amount is not None else "unknown",
                source="inflectsynth.capacity_metadata_mismatch",
                details={
                    "discovered_max_input_tokens": discovered_maximum,
                    "runtime_maximum": maximum,
                },
            )
        if amount is None:
            return RequestMeasure(
                fits=None,
                amount=None,
                maximum=None,
                unit="unknown",
                source="inflectsynth.no_capacity_measurement",
            )

        details = {
            "model_id": _measurement_value(
                measured, "model_id", self._selection.target_id
            )
        }
        policy_revision = _measurement_value(measured, "capacity_policy_revision")
        if not isinstance(policy_revision, str) or not policy_revision.strip():
            policy_revision = None
            for key in ("capacity_policy_revision", "capacity_revision"):
                candidate = self._selection.metadata.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    policy_revision = candidate
                    break
        if isinstance(policy_revision, str) and policy_revision.strip():
            details["capacity_policy_revision"] = policy_revision.strip()
        return RequestMeasure(
            fits=fits,
            amount=amount,
            maximum=maximum,
            unit="model_tokens",
            source="inflectsynth.measure_prepared",
            details=details,
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        if request.tokens:
            raise UnsupportedSynthesisFeatureError(
                "InflectSynth does not support linguistic tokens",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        if request.pronunciation_overrides:
            raise UnsupportedSynthesisFeatureError(
                "InflectSynth does not support pronunciation overrides",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        if request.whole_request_phonemes is not None:
            raise UnsupportedSynthesisFeatureError(
                "InflectSynth does not support whole-request phonemes",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        if request.speaker is not None:
            raise UnsupportedSynthesisFeatureError(
                "InflectSynth does not support speaker selection",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        if not _is_english(request.language):
            raise InvalidEngineLanguageError(
                f"InflectSynth supports English only, not {request.language!r}.",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                language=request.language,
                request_id=request.id,
                code="inflect.language_unsupported",
            )
        module = _module()
        voice = (
            request.voice
            or self._selection.voice
            or getattr(module, "DEFAULT_VOICE", "default")
        )
        try:
            result = self._voice.synthesize_prepared(
                request.text,
                voice=voice,
                config=self._config,
            )
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_error(exc, self._selection, request) from exc

        metadata = dict(getattr(result, "metadata", {}) or {})
        application = metadata.get("voice_level")
        metadata.update(
            {
                "voice": getattr(result, "voice", voice),
                "model_id": getattr(result, "model_id", self._selection.target_id),
                "model_ref": getattr(result, "model_ref", None),
                "source_revision": self._selection.metadata.get("source_revision"),
                "effective_speed": getattr(result, "speed", self._config.speed),
                "variation": getattr(result, "variation", self._config.variation),
                "seed": getattr(result, "seed", self._config.seed),
                "voice_level": voice_level_metadata(
                    application,
                    self._config.voice_level.mode,
                    identity_keys=("calibration_key",),
                    revision_keys=("catalog_revision",),
                ),
            }
        )
        rendered = RenderedSpeech(
            id=request.id,
            audio=result.audio,
            sample_rate=result.sample_rate,
            warnings=(),
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


class InflectSynthEngineAdapter:
    """InflectSynth discovery, selection, and prepared-text synthesis adapter."""

    id = ENGINE_ID
    package_name = PACKAGE_NAME

    def version(self) -> str | None:
        return distribution_version(self.package_name)

    def probe_api(self) -> EngineApiProbe:
        probe = probe_public_api(
            engine=self.id,
            package=self.package_name,
            required_symbols=(
                "InflectVoice",
                "SynthesisConfig",
                "SynthesisResult",
                "VoiceLevelConfig",
                "DiscoveredModel",
                "DescribedVoice",
                "discover_models",
                "runtime_identity",
                "REQUEST_API_VERSION",
                "request_api_contract",
                "RequestMeasure",
                "SynthesisInputTooLongError",
            ),
            required_methods={
                "InflectVoice": (
                    "from_pretrained",
                    "from_local",
                    "synthesize_prepared",
                    "measure_prepared",
                    "close",
                ),
                "__module__": ("discover_models", "runtime_identity"),
            },
            expected_api_version=SUPPORTED_REQUEST_API_VERSION,
            require_explicit_contract=True,
        )
        if not probe.compatible:
            return probe

        module = _module()
        contract = _request_measurement_contract(module)
        declared = {
            "supports_request_measurement": _contract_field(
                contract, "supports_request_measurement"
            ),
            "measurement_entrypoint": _contract_field(
                contract, "measurement_entrypoint"
            ),
            "measurement_metric": _contract_field(contract, "measurement_metric"),
        }
        if (
            declared["supports_request_measurement"] is not True
            or declared["measurement_entrypoint"] != "InflectVoice.measure_prepared"
            or declared["measurement_metric"] != "model_tokens"
        ):
            return replace(
                probe,
                compatible=False,
                status="api_incompatible",
                failed_stage="request_measurement_contract",
                failed_symbol="request_api_contract",
                warnings=probe.warnings
                + (
                    "InflectSynth does not declare the model-token request measurement API.",
                ),
                details={**dict(probe.details), **declared},
            )

        capacity_version = getattr(module, "CAPACITY_API_VERSION", None)
        capacity_contract = _capacity_contract(module)
        capacity_compatible = (
            isinstance(capacity_version, int)
            and not isinstance(capacity_version, bool)
            and capacity_version == SUPPORTED_CAPACITY_API_VERSION
            and capacity_contract is not None
        )
        details = {
            **dict(probe.details),
            **declared,
            "capacity_api_compatible": capacity_compatible,
            "capacity_api_version": capacity_version,
        }
        if capacity_contract is not None:
            details["capacity_contract"] = dict(capacity_contract)
        return replace(probe, details=details)

    def compatible_api(self) -> bool:
        return self.probe_api().compatible

    def capabilities(self) -> EngineCapabilities:
        try:
            supports_measurement = _supports_request_measurement()
        except (ImportError, ModuleNotFoundError):
            supports_measurement = False
        return EngineCapabilities(
            id=self.id,
            voice_binding_namespace=self.id,
            voice_binding_scope="request",
            option_names=INFLECT_OPTION_NAMES,
            supports_named_voices=True,
            supports_reference_voice=False,
            supports_speakers=False,
            supports_pronunciation_overrides=False,
            supports_linguistic_tokens=False,
            supports_whole_request_phonemes=False,
            supports_lexicons=False,
            supports_model_sources=False,
            supports_qualities=False,
            supports_live=True,
            supports_timestamps=False,
            supports_voice_level_calibration=True,
            supports_request_measurement=supports_measurement,
        )

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            frontend="inflectsynth",
            pronunciation="unsupported",
            capacity_key="max_input_tokens",
            capacity_metric="model_tokens",
        )

    def discover(self, request: CatalogRequest) -> tuple[SynthesisTarget, ...]:
        language = _effective_language(request.language) if request.language else None
        return _target_catalog(
            language=language,
            offline=request.offline,
            refresh=request.refresh,
        )

    def resolve(self, request: Any) -> tuple[EngineSelection, tuple[Any, ...]]:
        module = _module()
        options = dict(getattr(request, "options", {}) or {})
        engine_options = dict(getattr(request, "engine_options", {}) or {})
        unknown = sorted(set(engine_options) - INFLECT_OPTION_NAMES)
        if unknown:
            raise InflectSelectionError(
                "inflect.invalid_option",
                f"Unsupported InflectSynth option(s): {', '.join(unknown)}.",
                "synthesis.engine_options",
            )
        options.update(engine_options)
        default_model = getattr(module, "DEFAULT_MODEL", "nano-v2")
        default_voice = getattr(module, "DEFAULT_VOICE", "default")
        requested_target = str(getattr(request, "target_id", None) or default_model)
        language = _effective_language(getattr(request, "language", None))
        if not _is_english(language):
            raise InflectSelectionError(
                "inflect.language_unsupported",
                f"InflectSynth supports English only, not {language!r}.",
                "render.target.language",
            )
        offline = bool(getattr(request, "offline", False))
        refresh = bool(getattr(request, "refresh", False))
        try:
            targets = _target_catalog(
                language=None,
                offline=offline,
                refresh=refresh,
                cache_dir=options.get("cache_dir"),
                catalog_url=options.get("catalog_url"),
            )
        except Exception as exc:
            raise InflectSelectionError(
                "inflect.catalog_unavailable",
                f"InflectSynth model catalog is unavailable: {exc}",
                "synthesis.engine",
            ) from exc
        target = _find_target(targets, requested_target)
        target_id = target.id if target is not None else requested_target
        voice = (
            getattr(request, "voice", None)
            or options.pop("voice", None)
            or (target.default_voice if target is not None else None)
            or default_voice
        )
        try:
            config = module.SynthesisConfig(
                speed=options.get("speed", 1.0),
                variation=options.get("variation", 0.667),
                seed=options.get("seed", 0),
                voice_level=module.VoiceLevelConfig(
                    mode=options.get("voice_level", "off")
                ),
            ).validated()
        except module.InflectSynthError as exc:
            raise InflectSelectionError(
                "inflect.invalid_option",
                f"Invalid InflectSynth generation options: {exc}",
                "synthesis.engine_options",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise InflectSelectionError(
                "inflect.invalid_option",
                f"Invalid InflectSynth generation options: {exc}",
                "synthesis.engine_options",
            ) from exc
        selected_options = {
            name: options[name] for name in INFLECT_OPTION_NAMES if name in options
        }
        selected_options.update(
            {
                "speed": config.speed,
                "variation": config.variation,
                "seed": config.seed,
                "voice_level": config.voice_level.mode,
            }
        )
        metadata = (
            {
                "source_revision": target.metadata.get("source_revision"),
                "languages": target.languages,
                "voices": target.voices,
                "sample_rate": target.sample_rate,
                "max_input_tokens": target.metadata.get("max_input_tokens"),
                "capacity_policy_revision": (
                    target.metadata.get("capacity_policy_revision")
                    or target.metadata.get("capacity_revision")
                ),
            }
            if target is not None
            else {}
        )
        return (
            EngineSelection(
                engine=self.id,
                target_id=target_id,
                language=language,
                voice=str(voice),
                speaker=None,
                options=selected_options,
                metadata=metadata,
                offline=offline,
                refresh=refresh,
            ),
            (),
        )

    def validate_selection(self, selection: EngineSelection) -> tuple[Any, ...]:
        try:
            targets = _target_catalog(
                language=None,
                offline=selection.offline,
                refresh=selection.refresh,
                cache_dir=selection.options.get("cache_dir"),
                catalog_url=selection.options.get("catalog_url"),
            )
        except Exception as exc:  # noqa: BLE001 - Public discovery errors become selection diagnostics.
            return _selection_diagnostics(
                "inflect.catalog_unavailable",
                f"InflectSynth model catalog is unavailable: {exc}",
                "synthesis.engine",
            )
        target = _find_target(targets, selection.target_id)
        if target is None:
            return _selection_diagnostics(
                "inflect.model_not_found",
                f"InflectSynth model {selection.target_id!r} was not found.",
                "render.target.id",
            )
        diagnostics: list[EngineDiagnostic] = []
        if not target.runtime_available:
            diagnostics.extend(
                _selection_diagnostics(
                    "inflect.runtime_unavailable",
                    f"InflectSynth model {target.id!r} is not available to the runtime.",
                    "render.target.id",
                )
            )
        if (
            isinstance(target.sample_rate, bool)
            or not isinstance(target.sample_rate, int)
            or target.sample_rate <= 0
        ):
            diagnostics.extend(
                _selection_diagnostics(
                    "inflect.invalid_option",
                    f"InflectSynth model {target.id!r} has an invalid sample rate.",
                    "render.target.id",
                )
            )
        if not _is_english(selection.language):
            diagnostics.extend(
                _selection_diagnostics(
                    "inflect.language_unsupported",
                    f"InflectSynth supports English only, not {selection.language!r}.",
                    "render.target.language",
                )
            )
        if selection.voice not in target.voices:
            diagnostics.extend(
                _selection_diagnostics(
                    "inflect.voice_unavailable",
                    f"Voice {selection.voice!r} is not available on {target.id!r}.",
                    "render.target.voice",
                )
            )
        max_input_tokens = target.metadata.get("max_input_tokens")
        if max_input_tokens is not None and not _valid_positive_int(max_input_tokens):
            diagnostics.extend(
                _selection_diagnostics(
                    "inflect.invalid_option",
                    f"InflectSynth model {target.id!r} has invalid capacity metadata.",
                    "render.target.id",
                )
            )
        return tuple(diagnostics)

    def target_metadata(self, selection: EngineSelection) -> Mapping[str, Any]:
        try:
            targets = _target_catalog(
                language=None,
                offline=selection.offline,
                refresh=selection.refresh,
                cache_dir=selection.options.get("cache_dir"),
                catalog_url=selection.options.get("catalog_url"),
            )
        except Exception:  # noqa: BLE001 - Discovery failures make metadata unavailable.
            return {}
        target = _find_target(targets, selection.target_id)
        if target is None:
            return {}
        metadata = {
            **dict(target.metadata),
            "languages": target.languages,
            "voices": target.voices,
            "default_voice": target.default_voice,
            "sample_rate": target.sample_rate,
            "aliases": target.aliases,
            "source_revision": target.metadata.get("source_revision"),
        }
        if _valid_positive_int(target.metadata.get("max_input_tokens")):
            metadata["max_input_tokens"] = target.metadata["max_input_tokens"]
        else:
            metadata.pop("max_input_tokens", None)
        return metadata

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> Mapping[str, Any]:
        module = _module()
        voice = selection.voice or getattr(module, "DEFAULT_VOICE", "default")
        voice_ref = f"{ENGINE_ID}:{selection.target_id}/{voice}"
        mode = str(selection.options.get("voice_level", "off"))
        capacity_contract = _capacity_contract(module) or {}
        capacity_policy_revision = (
            selection.metadata.get("capacity_policy_revision")
            or selection.metadata.get("capacity_revision")
            or capacity_contract.get("policy_revision")
            or capacity_contract.get("revision")
        )
        calibration_revision = None
        calibration_identity = f"{voice_ref}@{_effective_language(selection.language)}"
        if mode == "calibrated":
            calibration_loader = getattr(module, "default_voice_calibration", None)
            if callable(calibration_loader):
                try:
                    calibration = calibration_loader()
                    calibration_revision = getattr(calibration, "revision", None)
                except Exception:  # noqa: BLE001 - Calibration identity is optional metadata.
                    calibration_revision = None
        acoustic_options = {
            name: selection.options[name]
            for name in (
                "speed",
                "variation",
                "seed",
                "providers",
                "provider_options",
                "session_options",
            )
            if name in selection.options
        }
        return {
            "engine": self.id,
            "engine_version": self.version(),
            "engine_identity": dict(module.runtime_identity()),
            "target_id": selection.target_id,
            "source_revision": selection.metadata.get("source_revision"),
            "voice": voice,
            "voice_ref": voice_ref,
            "language": _effective_language(selection.language),
            "options": acoustic_options,
            "voice_level": {
                "mode": mode,
                "calibration_identity": calibration_identity
                if mode == "calibrated"
                else None,
                "catalog_revision": calibration_revision,
            },
            "capacity_policy_revision": capacity_policy_revision,
        }

    def open(
        self, selection: EngineSelection
    ) -> AbstractContextManager[InflectSynthEngineSession]:
        module = _module()
        options = dict(selection.options)
        try:
            config = module.SynthesisConfig(
                speed=options.get("speed", 1.0),
                variation=options.get("variation", 0.667),
                seed=options.get("seed", 0),
                voice_level=module.VoiceLevelConfig(
                    mode=options.get("voice_level", "off")
                ),
            ).validated()
            voice = module.InflectVoice.from_pretrained(
                selection.target_id,
                cache_dir=options.get("cache_dir"),
                offline=selection.offline,
                refresh_catalog=selection.refresh,
                force_download=bool(options.get("force_download", False)),
                catalog_url=options.get("catalog_url"),
                providers=options.get("providers"),
                provider_options=options.get("provider_options"),
                session_options=options.get("session_options"),
            )
            actual_id = getattr(voice, "model_id", None)
            actual_metadata = getattr(voice, "metadata", {}) or {}
            upstream = (
                actual_metadata.get("upstream", {})
                if isinstance(actual_metadata, Mapping)
                else {}
            )
            actual_revision = (
                actual_metadata.get("source_revision")
                or (
                    upstream.get("source_revision")
                    if isinstance(upstream, Mapping)
                    else None
                )
                or (upstream.get("revision") if isinstance(upstream, Mapping) else None)
            )
            expected_revision = selection.metadata.get("source_revision")
            if actual_id != selection.target_id or (
                expected_revision is not None and actual_revision != expected_revision
            ):
                voice.close()
                raise EngineBackendError(
                    "Opened InflectSynth target differs from the planned target revision.",
                    engine=self.id,
                    engine_version=self.version(),
                    target_id=selection.target_id,
                    code="inflect.target_changed",
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
                yield InflectSynthEngineSession(voice, selection, config)
            finally:
                voice.close()

        return session()


__all__ = [
    "ENGINE_ID",
    "INFLECT_OPTION_NAMES",
    "PACKAGE_NAME",
    "InflectSelectionError",
    "InflectSynthEngineAdapter",
    "InflectSynthEngineSession",
]
