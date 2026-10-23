"""Request-centric PyKokoro adapter."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import re
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from typing import Any

from ..errors import (
    EmptySpeechTextError,
    EngineBackendError,
    EngineError,
    EngineSelectionError,
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
from ..voice_sources import (
    normalize_voice_source,
    resolve_reference_audio,
    validate_voice_source,
    voice_source_identity,
)
from .api_probe import SUPPORTED_REQUEST_API_VERSION, EngineApiProbe, probe_public_api
from .base import (
    PCM_RANGE_POLICY,
    EngineCapabilities,
    EngineOption,
    EngineSelection,
    PronunciationSpan,
    RenderedSpeech,
    RenderProfileFragment,
    RequestMeasure,
    SpeechRequest,
    SpeechToken,
    SpeechWordTiming,
    adapt_pcm_full_scale,
    distribution_version,
    validate_rendered_speech,
    voice_level_metadata,
)
from .catalog import SynthesisTarget, TargetVoice
from .metadata import language_base, normalize_language_key
from .profiles import make_renderprofile_fragment
from .registry import normalize_engine_id

logger = logging.getLogger(__name__)


PYKOKORO_REQUIRED = ">=0.10.5,<0.11"
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
    return (major, minor, patch) >= (0, 10, 5) and (major, minor) == (0, 10)


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
    voice_mode = getattr(capabilities, "voice_mode", "static")
    supports_reference = bool(
        getattr(capabilities, "supports_reference_enrollment", False)
    )
    enrollers = tuple(
        {
            "id": str(getattr(item, "id", "")),
            "kind": str(getattr(item, "kind", "")),
            "transcript_required": bool(getattr(item, "transcript_required", False)),
            "min_seconds": getattr(item, "min_seconds", None),
            "max_seconds": getattr(item, "max_seconds", None),
            "recommended_seconds": getattr(item, "recommended_seconds", None),
            "output_format": getattr(item, "output_format", None),
        }
        for item in tuple(getattr(capabilities, "voice_enrollers", ()) or ())
    )
    supports_enrollment = bool(
        getattr(capabilities, "supports_voice_enrollment", False) or enrollers
    )
    target_capabilities = {"pronunciation_overrides", "lexicons"}
    if capabilities.voices:
        target_capabilities.add("named_voices")
    if supports_reference:
        target_capabilities.add("reference_voice")
    if supports_enrollment:
        target_capabilities.add("voice_enrollment")
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
        capabilities=frozenset(target_capabilities),
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
            "voice_mode": voice_mode,
            "supports_reference_enrollment": supports_reference,
            "supports_voice_enrollment": supports_enrollment,
            "voice_enrollers": enrollers,
            "speed_supported": bool(getattr(capabilities, "speed_supported", True)),
        },
    )


def _metadata_enrollers(model: Any) -> tuple[Mapping[str, Any], ...]:
    metadata = getattr(model, "metadata", {})
    if not isinstance(metadata, Mapping):
        return ()
    raw = metadata.get("voice_enrollers", ())
    if not isinstance(raw, (tuple, list)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _supports_inno(model: Any) -> bool:
    return any(
        (item.get("id") == "inno-v0.2" or item.get("kind") == "inno")
        and item.get("transcript_required") is False
        for item in _metadata_enrollers(model)
    )


def _normalize_enrollment(raw: Any, model: Any, language: str) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise EngineSelectionError(
            "voice_enrollment must be a table",
            code="pykokoro.invalid_enrollment_options",
        )
    unknown = set(raw) - {"engine", "fmax", "reference_text"}
    if unknown:
        raise EngineSelectionError(
            f"Unknown voice_enrollment option(s): {', '.join(sorted(unknown))}",
            code="pykokoro.invalid_enrollment_options",
            details={"fields": sorted(unknown)},
        )
    requested = raw.get("engine", "auto")
    if requested not in {"auto", "inno", "akinvox"}:
        raise EngineSelectionError(
            "voice_enrollment.engine must be 'auto', 'inno', or 'akinvox'",
            code="pykokoro.invalid_enrollment_options",
        )
    metadata = getattr(model, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    voice_mode = metadata.get("voice_mode", "static")
    supports_reference = bool(metadata.get("supports_reference_enrollment"))
    if requested == "auto":
        if voice_mode == "reference":
            requested = "akinvox"
        elif voice_mode == "static" and _supports_inno(model):
            requested = "inno"
        else:
            raise EngineSelectionError(
                f"Target {model.id!r} has no supported reference enrollment engine",
                code="pykokoro.enrollment_not_supported",
                target_id=model.id,
            )

    result: dict[str, Any] = {"engine": requested}
    reference_text = raw.get("reference_text")
    fmax = raw.get("fmax")
    if requested == "inno":
        if voice_mode != "static" or not _supports_inno(model):
            raise EngineSelectionError(
                f"Target {model.id!r} does not advertise an Inno voice enroller",
                code="pykokoro.enrollment_not_supported",
                target_id=model.id,
            )
        if reference_text is not None:
            raise EngineSelectionError(
                "Inno enrollment does not accept reference_text",
                code="pykokoro.reference_text_unsupported",
                target_id=model.id,
            )
        if fmax is not None:
            if (
                isinstance(fmax, bool)
                or not isinstance(fmax, (int, float))
                or not math.isfinite(fmax)
                or fmax <= 0
            ):
                raise EngineSelectionError(
                    "Inno fmax must be a finite positive number",
                    code="pykokoro.invalid_enrollment_options",
                    target_id=model.id,
                )
            result["fmax"] = float(fmax)
        return result

    if voice_mode != "reference" or not supports_reference:
        raise EngineSelectionError(
            f"Target {model.id!r} does not support AkinVox reference enrollment",
            code="pykokoro.reference_not_supported",
            target_id=model.id,
        )
    if fmax is not None:
        raise EngineSelectionError(
            "fmax is only supported for Inno enrollment",
            code="pykokoro.invalid_enrollment_options",
            target_id=model.id,
        )
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise EngineSelectionError(
            "AkinVox enrollment requires non-empty reference_text",
            code="pykokoro.reference_text_required",
            target_id=model.id,
        )
    if language_base(language) != "en":
        raise EngineSelectionError(
            "AkinVox reference enrollment currently supports English only",
            code="pykokoro.reference_text_invalid_language",
            target_id=model.id,
            language=language,
        )
    result["reference_text"] = reference_text
    return result


_KOKORO_SHORT_SENTENCE_MODES = frozenset({"off", "wrap", "phrase", "randomized-phrase"})
_SHORT_SENTENCE_FIELDS = frozenset(
    {
        "enabled",
        "resolve_mode",
        "min_phoneme_length",
        "phoneme_pretext",
        "phrase_fallback_tries",
    }
)
_VOICE_LEVEL_FIELDS = frozenset({"mode", "gain_db"})


def _invalid_kokoro_option(message: str) -> InvalidEngineOptionError:
    return InvalidEngineOptionError(message, code="pykokoro.invalid_options")


def _normalize_short_sentence(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in _KOKORO_SHORT_SENTENCE_MODES:
            raise _invalid_kokoro_option(
                "short_sentence must be 'off', 'wrap', 'phrase', 'randomized-phrase', or a table"
            )
        config: dict[str, Any] = {
            "enabled": value != "off",
            "resolve_mode": "randomized-phrase" if value == "off" else value,
        }
    elif isinstance(value, Mapping):
        unknown = set(value) - _SHORT_SENTENCE_FIELDS
        if unknown:
            raise _invalid_kokoro_option(
                f"Unknown short_sentence option(s): {', '.join(sorted(map(str, unknown)))}"
            )
        config = {"enabled": value.get("enabled", True)}
        config["resolve_mode"] = value.get("resolve_mode", "randomized-phrase")
        config["min_phoneme_length"] = value.get("min_phoneme_length", 30)
        config["phoneme_pretext"] = value.get("phoneme_pretext", "—")
        config["phrase_fallback_tries"] = value.get("phrase_fallback_tries", 1)
    else:
        raise _invalid_kokoro_option(
            "short_sentence must be a mode string, table, or None"
        )

    config.setdefault("min_phoneme_length", 30)
    config.setdefault("phoneme_pretext", "—")
    config.setdefault("phrase_fallback_tries", 1)
    if not isinstance(config["enabled"], bool):
        raise _invalid_kokoro_option("short_sentence.enabled must be a boolean")
    mode = config["resolve_mode"]
    if not isinstance(mode, str) or mode not in _KOKORO_SHORT_SENTENCE_MODES - {"off"}:
        raise _invalid_kokoro_option(
            "short_sentence.resolve_mode must be 'wrap', 'phrase', or 'randomized-phrase'"
        )
    threshold = config["min_phoneme_length"]
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise _invalid_kokoro_option(
            "short_sentence.min_phoneme_length must be a non-negative integer"
        )
    pretext = config["phoneme_pretext"]
    if not isinstance(pretext, str) or not pretext.strip():
        raise _invalid_kokoro_option(
            "short_sentence.phoneme_pretext must be non-empty text"
        )
    retries = config["phrase_fallback_tries"]
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise _invalid_kokoro_option(
            "short_sentence.phrase_fallback_tries must be a non-negative integer"
        )
    return config


def _normalize_voice_level(value: Any) -> dict[str, Any]:
    config: dict[str, Any]
    if value is None:
        config = {"mode": "off"}
    elif isinstance(value, str):
        config = {"mode": value}
    elif isinstance(value, Mapping):
        unknown = set(value) - _VOICE_LEVEL_FIELDS
        if unknown:
            raise _invalid_kokoro_option(
                f"Unknown voice_level option(s): {', '.join(sorted(map(str, unknown)))}"
            )
        config = {"mode": value.get("mode", "off")}
        if "gain_db" in value:
            config["gain_db"] = value["gain_db"]
    else:
        raise _invalid_kokoro_option(
            "voice_level must be 'off', 'calibrated', a table, or None"
        )

    mode = config["mode"]
    if not isinstance(mode, str) or mode not in {"off", "calibrated"}:
        raise _invalid_kokoro_option("voice_level.mode must be 'off' or 'calibrated'")
    gain = config.get("gain_db")
    if gain is not None:
        if isinstance(gain, bool) or not isinstance(gain, (int, float)):
            raise _invalid_kokoro_option(
                "voice_level.gain_db must be a finite number or None"
            )
        try:
            normalized_gain = float(gain)
        except OverflowError:
            normalized_gain = math.inf
        if not math.isfinite(normalized_gain):
            raise _invalid_kokoro_option(
                "voice_level.gain_db must be a finite number or None"
            )
        config["gain_db"] = normalized_gain
    else:
        config.pop("gain_db", None)
    return config


def _normalize_kokoro_options(options: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(options)
    short_sentence = _normalize_short_sentence(normalized.get("short_sentence"))
    if short_sentence is None:
        normalized.pop("short_sentence", None)
    else:
        normalized["short_sentence"] = short_sentence
    normalized["voice_level"] = _normalize_voice_level(normalized.get("voice_level"))
    return normalized


def _reference_text_digest(enrollment: Mapping[str, Any]) -> str | None:
    text = enrollment.get("reference_text")
    if not isinstance(text, str):
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reference_source_identity(options: Mapping[str, Any]) -> Mapping[str, str] | None:
    source = options.get("voice_source")
    if not isinstance(source, Mapping):
        return None
    return voice_source_identity(source)


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
    failure_code: str | None = None,
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
            source="pykokoro.measure_request",
        )
    if failure_code is not None and error_type is EngineBackendError:
        context["code"] = failure_code
    details: dict[str, object] = {}
    for attribute, detail_name in (
        ("stage", "native_stage"),
        ("native_error_message", "native_error_message"),
    ):
        try:
            value = getattr(error, attribute, None)
        except Exception:  # noqa: BLE001 - optional diagnostic fields must not block rendering.
            value = None
        if isinstance(value, str):
            details[detail_name] = value

    native_diagnostic = getattr(pykokoro, "error_diagnostic", None)
    if callable(native_diagnostic):
        try:
            diagnostic = native_diagnostic(error)
        except Exception:  # noqa: BLE001 - diagnostics are best-effort.
            diagnostic = None
        if isinstance(diagnostic, Mapping):
            details["native_diagnostic"] = diagnostic
    if details:
        context["details"] = details
    return error_type(str(error), **context)


def _pykokoro_request(
    request: SpeechRequest, module: Any, *, voice: Any | None = None
) -> Any:
    return module.SynthesisRequest(
        id=request.id,
        text=request.text,
        language=request.language,
        voice=voice if voice is not None else request.voice,
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
        self._enrolled_voices: dict[str, Any] = {}

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        import pykokoro

        maximum = self._selection.metadata.get("max_tokens")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            maximum = None
        try:
            measured = self._synthesizer.measure_request(
                _pykokoro_request(request, pykokoro), maximum=maximum
            )
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_pykokoro_error(exc, self._selection, request) from exc
        amount = getattr(measured, "amount", None)
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise EngineBackendError(
                "PyKokoro returned an invalid request measurement",
                engine="kokoro",
                target_id=self._selection.target_id,
                request_id=request.id,
                code="pykokoro.measurement_invalid",
            )
        measured_maximum = getattr(measured, "maximum", maximum)
        if (
            isinstance(measured_maximum, bool)
            or not isinstance(measured_maximum, int)
            or measured_maximum <= 0
        ):
            measured_maximum = maximum
        return RequestMeasure(
            fits=amount <= measured_maximum if measured_maximum is not None else None,
            amount=amount,
            maximum=measured_maximum,
            unit="model_tokens",
            source="pykokoro.measure_request",
        )

    def _reference_voice(
        self, language: str
    ) -> tuple[Any | None, Mapping[str, str] | None, str | None]:
        source = self._selection.options.get("voice_source")
        enrollment = self._selection.options.get("voice_enrollment")
        if not isinstance(source, Mapping) or not isinstance(enrollment, Mapping):
            return None, None, None
        source_identity = voice_source_identity(source)
        identity: dict[str, Any] = {
            "engine": enrollment.get("engine"),
            "target_id": self._selection.target_id,
            "language": normalize_language_key(language),
            "voice_source": dict(source_identity),
        }
        transcript_digest = _reference_text_digest(enrollment)
        if transcript_digest is not None:
            identity["reference_text_sha256"] = transcript_digest
        if "fmax" in enrollment:
            identity["fmax"] = enrollment["fmax"]
        cache_key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        cached = self._enrolled_voices.get(cache_key)
        if cached is not None:
            return cached, source_identity, cache_key

        import pykokoro

        try:
            resolved = resolve_reference_audio(
                source,
                cache_dir=self._selection.options.get("voice_cache_dir"),
                offline=self._selection.offline,
                catalog_paths=self._selection.options.get("voice_catalog_path"),
            )
            engine = enrollment.get("engine")
            enrollment_options = None
            reference_text = None
            if engine == "inno":
                enrollment_options = pykokoro.InnoEnrollmentOptions(
                    fmax=enrollment.get("fmax")
                )
            elif engine == "akinvox":
                reference_text = enrollment.get("reference_text")
            else:
                raise EngineSelectionError(
                    f"Unsupported PyKokoro enrollment engine {engine!r}",
                    code="pykokoro.invalid_enrollment_options",
                    target_id=self._selection.target_id,
                )
            voice = self._synthesizer.enroll_voice(
                resolved.path,
                reference_text,
                engine=engine,
                language=language,
                options=enrollment_options,
            )
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_pykokoro_error(
                exc,
                self._selection,
                failure_code="pykokoro.reference_enrollment_failed",
            ) from exc
        self._enrolled_voices[cache_key] = voice
        return voice, source_identity, cache_key

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        import pykokoro

        voice, source_identity, enrollment_identity = self._reference_voice(
            request.language
        )
        try:
            result = self._synthesizer.synthesize(
                _pykokoro_request(request, pykokoro, voice=voice)
            )
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
        adaptation = adapt_pcm_full_scale(result.audio, engine_label="PyKokoro")
        applications = tuple(getattr(result, "voice_level_applications", ()) or ())
        metadata: dict[str, Any] = {
            "language": result.language,
            "voice": result.voice,
            "phonemes": result.phonemes,
            "token_ids": tuple(result.token_ids),
            "pcm_range": {
                "policy": PCM_RANGE_POLICY,
                "peak_before_clip": adaptation.peak_before_clip,
                "clipped_samples": adaptation.clipped_samples,
            },
            "voice_level": voice_level_metadata(
                applications[0] if applications else None,
                _normalize_voice_level(self._selection.options.get("voice_level"))[
                    "mode"
                ],
                identity_keys=("key", "calibration_identity"),
                revision_keys=("corpus", "catalog_revision"),
            ),
        }
        if source_identity is not None:
            metadata["reference_voice_identity"] = dict(source_identity)
            metadata["reference_enrollment_identity"] = enrollment_identity
        if result.trace is not None:
            metadata["trace"] = result.trace
        rendered = RenderedSpeech(
            id=result.id,
            audio=adaptation.audio,
            sample_rate=result.sample_rate,
            warnings=tuple(result.diagnostics)
            + ((adaptation.warning,) if adaptation.warning else ()),
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
        probe = probe_public_api(
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
                "RequestMeasure",
                "ReferenceVoice",
                "KokoroVoicePack",
                "InnoEnrollmentOptions",
            ),
            required_methods={
                "KokoroSynthesizer": (
                    "prepare",
                    "measure_request",
                    "enroll_voice",
                    "synthesize",
                    "close",
                ),
            },
            expected_api_version=SUPPORTED_REQUEST_API_VERSION,
            require_explicit_contract=True,
            contract_required_symbols=("REQUEST_API_VERSION", "request_api_contract"),
            contract_required_methods={
                "KokoroSynthesizer": ("measure_request", "enroll_voice"),
            },
        )
        if not probe.compatible:
            return probe
        try:
            import pykokoro

            contract = pykokoro.request_api_contract()
            required = {
                "supports_request_measurement": True,
                "supports_reference_voice": True,
                "supports_voice_enrollment": True,
            }
            actual = {
                name: (
                    contract.get(name)
                    if isinstance(contract, Mapping)
                    else getattr(contract, name, None)
                )
                for name in required
            }
            mismatches = {
                name: {"expected": expected, "actual": actual[name]}
                for name, expected in required.items()
                if actual[name] is not expected
            }
        except Exception as exc:  # noqa: BLE001 - API probes must fail closed.
            return replace(
                probe,
                compatible=False,
                status="api_probe_failed",
                failed_stage="request_capability_contract",
                failed_symbol="request_api_contract",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        if mismatches:
            return replace(
                probe,
                compatible=False,
                status="api_incompatible",
                failed_stage="request_capabilities",
                failed_symbol="request_api_contract",
                details={**dict(probe.details), "capability_mismatches": mismatches},
            )
        return probe

    def compatible_api(self) -> bool:
        return self.probe_api().compatible

    def option_descriptors(
        self, *, target_id: str | None = None
    ) -> Mapping[str, EngineOption]:
        """Describe supported Kokoro options, including structured table fields."""
        del target_id
        return {
            "short_sentence": EngineOption(
                name="short_sentence",
                value_type="string|table",
                default=None,
                choices=("off", "wrap", "phrase", "randomized-phrase"),
                description="Mode shorthand or table with enabled, resolve_mode, min_phoneme_length, phoneme_pretext, phrase_fallback_tries.",
            ),
            "short_sentence.enabled": EngineOption(
                name="short_sentence.enabled",
                value_type="boolean",
                default=True,
                description="Enable the short-sentence context quality fix.",
            ),
            "short_sentence.resolve_mode": EngineOption(
                name="short_sentence.resolve_mode",
                value_type="string",
                default="randomized-phrase",
                choices=("wrap", "phrase", "randomized-phrase"),
                description="How short requests are expanded into context.",
            ),
            "short_sentence.min_phoneme_length": EngineOption(
                name="short_sentence.min_phoneme_length",
                value_type="integer",
                default=30,
                minimum=0,
                description="Minimum phoneme count before short-sentence context is added.",
            ),
            "short_sentence.phoneme_pretext": EngineOption(
                name="short_sentence.phoneme_pretext",
                value_type="string",
                default="—",
                description="Text prepended to the short-sentence context.",
            ),
            "short_sentence.phrase_fallback_tries": EngineOption(
                name="short_sentence.phrase_fallback_tries",
                value_type="integer",
                default=1,
                minimum=0,
                description="Number of phrase fallback attempts.",
            ),
            "voice_level": EngineOption(
                name="voice_level",
                value_type="string|table",
                default="off",
                choices=("off", "calibrated"),
                description="Mode shorthand or table with mode and optional gain_db.",
            ),
            "voice_level.mode": EngineOption(
                name="voice_level.mode",
                value_type="string",
                default="off",
                choices=("off", "calibrated"),
                description="Use the static PyKokoro calibration catalog when calibrated.",
            ),
            "voice_level.gain_db": EngineOption(
                name="voice_level.gain_db",
                value_type="number",
                default=None,
                description="Explicit PyKokoro gain override, including for enrolled voices.",
            ),
        }

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
                    "voice_source",
                    "voice_enrollment",
                    "voice_catalog_path",
                    "voice_cache_dir",
                }
            ),
            supports_named_voices=True,
            supports_reference_voice=True,
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

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            token_fields=("span", "pos", "tag"),
            frontend="pykokoro",
            pronunciation="pykokoro.ipa",
            capacity_key="max_tokens",
            capacity_metric="model_tokens",
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

    def bind_reference_voice(
        self,
        *,
        target: SynthesisTarget,
        language: str,
        voice_source: Mapping[str, Any],
        transcript: str | None,
        options: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Bind generic clone input to Kokoro's native enrollment configuration."""
        del (
            voice_source
        )  # The normalized source remains bound by create_render_profile.
        bound = dict(options)
        raw_enrollment = bound.get("voice_enrollment")
        if raw_enrollment is None:
            raw_enrollment = {}
        if not isinstance(raw_enrollment, Mapping):
            raise EngineSelectionError(
                "voice_enrollment must be a table",
                code="pykokoro.invalid_enrollment_options",
                target_id=target.id,
            )
        enrollment = dict(raw_enrollment)
        existing = enrollment.get("reference_text")
        if transcript is not None and existing is not None and existing != transcript:
            raise EngineSelectionError(
                "Conflicting transcript supplied in voice_enrollment.reference_text and clone request",
                code="pykokoro.reference_text_conflict",
                target_id=target.id,
            )
        if transcript is not None:
            enrollment["reference_text"] = transcript
        bound["voice_enrollment"] = _normalize_enrollment(enrollment, target, language)
        return bound

    def resolve(self, request: Any) -> tuple[EngineSelection, tuple[Any, ...]]:
        options = dict(getattr(request, "options", {}) or {})
        options.update(dict(getattr(request, "engine_options", {}) or {}))
        try:
            options = _normalize_kokoro_options(options)
        except InvalidEngineOptionError:
            # Keep invalid input available for structured validation diagnostics.
            pass
        target_id = (
            getattr(request, "target_id", None)
            or options.get("model_variant")
            or "v1.0"
        )
        language = getattr(request, "language", "en-us")
        voice = getattr(request, "voice", None)
        speaker = getattr(request, "speaker", None)
        offline = bool(getattr(request, "offline", False))
        refresh = bool(getattr(request, "refresh", False))
        if "voice_source" in options:
            try:
                if voice is not None or speaker is not None:
                    raise EngineSelectionError(
                        "A reference voice source cannot be combined with a named voice or speaker",
                        code="pykokoro.reference_voice_binding_conflict",
                    )
                source = normalize_voice_source(
                    options["voice_source"], allow_named=False
                )
                options["voice_source"] = source.as_mapping()
                from ..models import ModelDiscoveryError, get_model_info

                try:
                    model, _ = get_model_info(
                        target_id,
                        offline=offline,
                        engine=self.id,
                        refresh=refresh,
                        preference=options.get("model_source", "auto"),
                    )
                except (ModelDiscoveryError, ImportError, OSError, ValueError):
                    pass
                else:
                    options["voice_enrollment"] = _normalize_enrollment(
                        options.get("voice_enrollment"), model, language
                    )
            except EngineError:
                # validate_selection returns structured diagnostics for invalid bindings.
                pass
        return (
            EngineSelection(
                engine=self.id,
                target_id=target_id,
                language=language,
                voice=voice,
                speaker=speaker,
                options=options,
                offline=offline,
                refresh=refresh,
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
        try:
            _normalize_kokoro_options(selection.options)
        except InvalidEngineOptionError as exc:
            diagnostics.append(
                EngineDiagnostic(
                    code=exc.code,
                    severity="error",
                    message=str(exc),
                    field="render.options",
                )
            )
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
        source_value = selection.options.get("voice_source")
        enrollment_value = selection.options.get("voice_enrollment")
        if source_value is not None:
            try:
                source = normalize_voice_source(source_value, allow_named=False)
                diagnostics.extend(
                    validate_voice_source(
                        source,
                        offline=selection.offline,
                        catalog_paths=selection.options.get("voice_catalog_path"),
                    )
                )
            except EngineError as exc:
                diagnostics.append(
                    EngineDiagnostic(
                        code=exc.code,
                        severity="error",
                        message=str(exc),
                        field="render.target.voice",
                        details=exc.details,
                    )
                )
            if selection.voice is not None or selection.speaker is not None:
                diagnostics.append(
                    EngineDiagnostic(
                        code="pykokoro.reference_voice_binding_conflict",
                        severity="error",
                        message="A reference voice source cannot be combined with a named voice or speaker.",
                        field="render.target.voice",
                    )
                )
            try:
                normalized_enrollment = _normalize_enrollment(
                    enrollment_value, model, selection.language
                )
            except EngineError as exc:
                diagnostics.append(
                    EngineDiagnostic(
                        code=exc.code,
                        severity="error",
                        message=str(exc),
                        field="render.target.voice",
                        details=exc.details,
                    )
                )
            else:
                if normalized_enrollment["engine"] == "akinvox":
                    speed = selection.options.get("speed", 1.0)
                    try:
                        supported_speed = (
                            not isinstance(speed, bool)
                            and isinstance(speed, (int, float))
                            and math.isfinite(float(speed))
                            and float(speed) == 1.0
                        )
                    except OverflowError:
                        supported_speed = False
                    if not supported_speed:
                        diagnostics.append(
                            EngineDiagnostic(
                                code="pykokoro.reference_speed_unsupported",
                                severity="error",
                                message="AkinVox reference synthesis requires speed == 1.0.",
                                field="render.options.speed",
                            )
                        )
        elif enrollment_value is not None:
            diagnostics.append(
                EngineDiagnostic(
                    code="pykokoro.reference_source_required",
                    severity="error",
                    message="voice_enrollment requires a voice_source reference audio binding.",
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
        from ..models import ModelDiscoveryError, get_model_info

        try:
            model, _ = get_model_info(
                selection.target_id,
                offline=selection.offline,
                engine=self.id,
                refresh=selection.refresh,
                preference=selection.options.get("model_source", "auto"),
            )
        except (ModelDiscoveryError, ImportError, OSError, ValueError):
            return {}
        metadata = dict(model.metadata)
        return {
            **metadata,
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
        options = _normalize_kokoro_options(selection.options)
        source_identity = _reference_source_identity(options)
        enrollment = options.get("voice_enrollment")
        safe_options = {
            key: value
            for key, value in options.items()
            if key
            not in {
                "voice_source",
                "voice_enrollment",
                "voice_catalog_path",
                "voice_cache_dir",
            }
        }
        enrollment_identity: dict[str, Any] | None = None
        if isinstance(enrollment, Mapping):
            enrollment_identity = {"engine": enrollment.get("engine")}
            enrollment_identity["language"] = normalize_language_key(selection.language)
            if "fmax" in enrollment:
                enrollment_identity["fmax"] = enrollment["fmax"]
            transcript_digest = _reference_text_digest(enrollment)
            if transcript_digest is not None:
                enrollment_identity["reference_text_sha256"] = transcript_digest
        identity: dict[str, Any] = {
            "engine": self.id,
            "pcm_range_policy": PCM_RANGE_POLICY,
            "engine_version": self.version(),
            "target_id": selection.target_id,
            "voice": None if source_identity is not None else selection.voice,
            "speaker": selection.speaker,
            "options": safe_options,
            "metadata": dict(selection.metadata),
        }
        if source_identity is not None:
            identity["voice_source"] = dict(source_identity)
            identity["voice_enrollment"] = enrollment_identity
        return identity

    def open(
        self, selection: EngineSelection
    ) -> AbstractContextManager[PyKokoroEngineSession]:
        import pykokoro

        options = _normalize_kokoro_options(selection.options)
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
            short_sentence_config = (
                pykokoro.ShortSentenceConfig(**short_sentence)
                if short_sentence is not None
                else None
            )
            config = pykokoro.SynthesisConfig(
                voice=None if "voice_source" in options else selection.voice,
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
                voice_level=pykokoro.VoiceLevelConfig(**options["voice_level"]),
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
