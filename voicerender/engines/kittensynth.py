"""Request-centric adapter for the public KittenSynth API."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

from ..error_diagnostics import sanitize_error_message
from ..errors import (
    EmptySpeechTextError,
    EngineBackendError,
    EngineError,
    EngineSynthesisError,
    InvalidEngineModelError,
    InvalidEngineOptionError,
    InvalidEngineVoiceError,
    SpeechRequestTooLongError,
    UnsupportedSynthesisFeatureError,
)
from .api_probe import EngineApiProbe, probe_public_api
from .base import (
    PCM_RANGE_POLICY,
    EngineCapabilities,
    EngineOption,
    EngineSelection,
    RenderedSpeech,
    RenderProfileFragment,
    RequestMeasure,
    SpeechRequest,
    adapt_pcm_full_scale,
    distribution_version,
    validate_rendered_speech,
)
from .catalog import CatalogRequest, SynthesisTarget, TargetVoice
from .metadata import (
    language_base,
    normalize_gender,
    normalize_language_key,
    normalize_locale_tag,
)
from .profiles import make_renderprofile_fragment

DEFAULT_KITTEN_MODEL = "nano-0.8-int8"
DEFAULT_KITTEN_VOICE = "Jasper"
KITTEN_OPTION_NAMES = frozenset(
    {
        "speed",
        "quality",
        "cache_dir",
        "force_download",
        "providers",
        "provider_options",
        "session_options",
        "lexicons",
        "g2lex_files",
        "lexicon_data_policy",
        "heteronym_policy",
    }
)
KITTEN_REQUEST_API_VERSION = 2

KITTEN_PCM_RANGE_POLICY = PCM_RANGE_POLICY


class KittenSelectionError(ValueError):
    """A Kitten selection error with a stable plan diagnostic."""

    def __init__(self, code: str, message: str, field: str) -> None:
        super().__init__(message)
        self.diagnostic_code = code

        self.diagnostic_field = field


def _kitten_request_inputs(
    request: SpeechRequest, module: Any
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    tokens = tuple(
        module.LinguisticToken(
            start=item.start,
            end=item.end,
            text=item.text,
            pos=item.pos,
            tag=item.tag,
            lemma=item.lemma,
            morph=item.morph,
            language=item.language,
        )
        for item in request.tokens
    )
    overrides: list[Any] = []
    for item in request.pronunciation_overrides:
        alphabet = item.alphabet or "ipa"
        if alphabet.casefold() != "ipa":
            raise UnsupportedSynthesisFeatureError(
                f"KittenSynth supports IPA pronunciation overrides, not {alphabet!r}",
                engine="kitten",
                language=item.language or request.language,
            )
        if not isinstance(item.phonemes, str) or not item.phonemes.strip():
            raise InvalidEngineOptionError(
                "Kitten pronunciation spans require non-empty IPA phonemes",
                engine="kitten",
                language=item.language or request.language,
            )
        overrides.append(
            module.PronunciationOverride(
                start=item.start,
                end=item.end,
                phonemes=item.phonemes,
                language=item.language,
                alphabet="ipa",
            )
        )
    return tokens, tuple(overrides)


def _source_entries(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if (
        isinstance(value, (Mapping, str, Path))
        or getattr(value, "path", None) is not None
    ):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _source_field(source: Any, name: str) -> Any:
    return (
        source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
    )


def _requested_selectors(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        raise TypeError("lexicons must be a selector or sequence of selector strings")
    values = (value,) if isinstance(value, str) else tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError("lexicons must contain non-empty selector strings")
    return tuple(item.strip() for item in values)


def _sha256_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold().removeprefix("sha256:")
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        return None
    return normalized


def _is_english(language: str) -> bool:
    normalized = normalize_language_key(language)
    return normalized == "english" or language_base(normalized) == "en"


def _language_values(value: Any) -> tuple[str, str, str]:
    raw = str(value or "en")
    normalized = normalize_locale_tag(raw)
    base = language_base(normalized) or normalize_language_key(raw)
    if base == "english":
        base = "en"
        normalized = "en"
    return base or "unknown", normalized or base or "unknown", raw


def _target_from_model(model: Any) -> SynthesisTarget:
    language, locale, raw_language = _language_values(model.language)
    quality = model.quality
    voices = tuple(str(voice) for voice in model.voice_ids)
    display_name = str(model.display_name)
    if model.version:
        display_name = f"{display_name} {model.version}"
    normalized_metadata = {
        **dict(model.metadata),
        "language": language,
        "locale": locale,
        "language_label": raw_language
        if raw_language.casefold() != "en"
        else "English",
        "quality": quality,
        "source": "kittensynth",
    }
    missing = object()
    maximum = getattr(model, "max_tokens", missing)
    if maximum is missing:
        maximum = normalized_metadata.get("max_tokens")
    if _valid_measurement_integer(maximum, positive=True):
        normalized_metadata["max_tokens"] = maximum
    else:
        normalized_metadata.pop("max_tokens", None)

    policy_revision = normalized_metadata.get("capacity_policy_revision")
    if not isinstance(policy_revision, str) or not policy_revision.strip():
        policy_revision = normalized_metadata.get("capacity_revision")
    if isinstance(policy_revision, str) and policy_revision.strip():
        normalized_metadata["capacity_policy_revision"] = policy_revision.strip()
    else:
        normalized_metadata.pop("capacity_policy_revision", None)
    return SynthesisTarget(
        engine="kitten",
        id=str(model.id),
        display_name=display_name,
        languages=(locale,) if locale and locale != "unknown" else (),
        status="ready",
        runtime_available=bool(model.runtime_available),
        sample_rate=model.sample_rate,
        voices=voices,
        voice_details=tuple(
            TargetVoice(
                id=str(voice.id),
                gender=normalize_gender(voice.gender),
                language=language_base(voice.language) or language,
                locale=normalize_locale_tag(voice.locale) or locale,
                language_label=str(voice.language_label or raw_language),
            )
            for voice in model.voices
        ),
        default_voice=model.default_voice,
        qualities=(str(quality),) if isinstance(quality, str) and quality else (),
        aliases=tuple(str(alias) for alias in model.aliases),
        capabilities=frozenset({"named_voices"}),
        metadata=normalized_metadata,
    )


def _catalog_targets(request: CatalogRequest) -> tuple[SynthesisTarget, ...]:
    import kittensynth

    models = kittensynth.discover_models(
        language=request.language,
        offline=request.offline,
        refresh=request.refresh,
    )
    return tuple(_target_from_model(model) for model in models)


def _kitten_model_download_details(error: Exception) -> dict[str, Any]:
    """Extract bounded transport diagnostics without copying arbitrary metadata."""
    details: dict[str, Any] = {"stage": "model_install"}
    cause = error.__cause__
    if cause is None and not error.__suppress_context__:
        cause = error.__context__
    if cause is None:
        return details

    details["transport_error_type"] = type(cause).__name__
    native_stage = getattr(cause, "stage", None)
    if isinstance(native_stage, str) and native_stage.strip():
        details["native_stage"] = sanitize_error_message(native_stage, limit=128)

    native_error_type = getattr(cause, "native_error_type", None)
    if isinstance(native_error_type, str) and native_error_type.strip():
        details["transport_native_error_type"] = sanitize_error_message(
            native_error_type, limit=128
        )

    native_message = getattr(error, "native_error_message", None)
    if not isinstance(native_message, str):
        native_message = getattr(cause, "native_error_message", None)
    if (
        not isinstance(native_message, str)
        and type(cause).__name__ == "AssetTransportError"
    ):
        native_message = str(cause)
    if isinstance(native_message, str) and native_message.strip():
        details["native_error_message"] = sanitize_error_message(
            native_message, limit=1024
        )

    for name in (
        "fallback_attempted",
        "fallback_failed",
        "fallback_outcome",
        "retry_attempted",
        "retry_failed",
        "retry_outcome",
        "runtime_switch_activated",
    ):
        value = getattr(cause, name, None)
        if isinstance(value, bool):
            details[name] = value
        elif isinstance(value, str) and value.strip():
            details[name] = sanitize_error_message(value, limit=64)
    return details


def _translate_kitten_error(
    error: Exception,
    selection: EngineSelection,
    request: SpeechRequest | None = None,
    *,
    opening: bool = False,
) -> EngineError:
    try:
        import kittensynth
    except ImportError:
        kittensynth = None

    error_map: tuple[tuple[str, type[EngineError], str | None], ...] = (
        ("EmptyTextError", EmptySpeechTextError, None),
        ("SynthesisInputTooLongError", SpeechRequestTooLongError, None),
        ("InvalidSpeedError", InvalidEngineOptionError, "kitten.invalid_speed"),
        ("InvalidVoiceError", InvalidEngineVoiceError, "kitten.voice_unavailable"),
        ("UnsupportedModelError", InvalidEngineModelError, "kitten.model_unsupported"),
        (
            "ManagedModelDependencyError",
            EngineBackendError,
            "kitten.runtime_dependency_error",
        ),
        ("OnnxVoiceContractError", EngineBackendError, "kitten.runtime_contract_error"),
        ("ModelDownloadError", EngineBackendError, "kitten.model_download_error"),
    )
    error_type: type[EngineError] = EngineBackendError
    code: str | None = "kitten.engine_error"
    if kittensynth is not None:
        for native_name, mapped_type, mapped_code in error_map:
            native_type = getattr(kittensynth, native_name, None)
            if isinstance(native_type, type) and isinstance(error, native_type):
                error_type = mapped_type
                code = mapped_code
                break
        else:
            if opening and isinstance(error, (TypeError, ValueError)):
                error_type = InvalidEngineOptionError
                code = "kitten.invalid_options"
    context: dict[str, Any] = {
        "engine": "kitten",
        "engine_version": distribution_version("kittensynth"),
        "target_id": selection.target_id,
        "language": request.language if request is not None else selection.language,
        "voice": request.voice if request is not None else selection.voice,
        "request_id": request.id if request is not None else None,
        "native_error_type": type(error).__name__,
    }
    if error_type is SpeechRequestTooLongError:
        context.update(
            amount=getattr(error, "amount", getattr(error, "token_count", None)),
            maximum=getattr(error, "maximum", getattr(error, "max_tokens", None)),
            unit=getattr(error, "unit", "model_tokens"),
            text_length=len(request.text) if request is not None else None,
            source="kittensynth.synthesize_prepared",
        )
    details = (
        _kitten_model_download_details(error)
        if code == "kitten.model_download_error"
        else None
    )
    message = (
        sanitize_error_message(str(error), limit=2048)
        if code == "kitten.model_download_error"
        else str(error)
    )
    if code == "kitten.model_download_error" and not message:
        message = "Kitten model download failed."
    return error_type(message, **context, details=details, code=code)


def _measurement_field(measured: Any, name: str, default: Any = None) -> Any:
    if isinstance(measured, Mapping):
        return measured.get(name, default)
    return getattr(measured, name, default)


def _valid_measurement_integer(value: Any, *, positive: bool) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (value > 0 if positive else value >= 0)
    )


class KittenSynthEngineSession:
    """Adapt prepared VoiceRender requests to one open ``KittenVoice``."""

    def __init__(self, voice: Any, selection: EngineSelection) -> None:
        self._voice = voice
        self._selection = selection

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        measure_method = getattr(self._voice, "measure_prepared", None)
        if not callable(measure_method):
            return RequestMeasure(
                fits=None,
                amount=None,
                maximum=None,
                unit="unknown",
                source="kittensynth.no_capacity_measurement",
            )
        try:
            import kittensynth

            tokens, overrides = _kitten_request_inputs(request, kittensynth)
            request_inputs: dict[str, Any] = {}
            if request.tokens:
                request_inputs["tokens"] = tokens
            if request.pronunciation_overrides:
                request_inputs["pronunciation_overrides"] = overrides
            measured = measure_method(request.text, **request_inputs)
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_kitten_error(exc, self._selection, request) from exc

        amount_value = _measurement_field(measured, "token_count")
        maximum_value = _measurement_field(measured, "max_tokens")
        raw_fits = _measurement_field(measured, "fits")
        amount = (
            amount_value
            if _valid_measurement_integer(amount_value, positive=False)
            else None
        )
        maximum = (
            maximum_value
            if _valid_measurement_integer(maximum_value, positive=True)
            else None
        )
        fits = raw_fits if isinstance(raw_fits, bool) else None

        invalid_fields: list[str] = []
        if amount_value is not None and amount is None:
            invalid_fields.append("amount")
        if maximum_value is not None and maximum is None:
            invalid_fields.append("maximum")
        if raw_fits is not None and not isinstance(raw_fits, bool):
            invalid_fields.append("fits")
        unit: Literal["model_tokens", "unknown"] = (
            "model_tokens" if amount is not None else "unknown"
        )
        if invalid_fields:
            return RequestMeasure(
                fits=None,
                amount=amount,
                maximum=maximum,
                unit=unit,
                source="kittensynth.invalid_measurement",
                details={"invalid_fields": invalid_fields},
            )
        if amount is None:
            return RequestMeasure(
                fits=None,
                amount=None,
                maximum=None,
                unit="unknown",
                source="kittensynth.no_capacity_measurement",
            )

        details: dict[str, Any] = {
            "model_id": _measurement_field(
                measured, "model_id", self._selection.target_id
            )
        }
        native_metadata = _measurement_field(measured, "metadata")
        if isinstance(native_metadata, Mapping):
            details["native_metadata"] = dict(native_metadata)
            policy_revision = native_metadata.get("capacity_policy_revision")
            if isinstance(policy_revision, str) and policy_revision:
                details["capacity_policy_revision"] = policy_revision
        return RequestMeasure(
            fits=fits,
            amount=amount,
            maximum=maximum,
            unit="model_tokens",
            source="kittensynth.measure_prepared",
            details=details,
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        speed = float(self._selection.options.get("speed", 1.0))
        try:
            import kittensynth

            tokens, overrides = _kitten_request_inputs(request, kittensynth)
            request_inputs: dict[str, Any] = {}
            if request.tokens:
                request_inputs["tokens"] = tokens
            if request.pronunciation_overrides:
                request_inputs["pronunciation_overrides"] = overrides
            result = self._voice.synthesize_prepared(
                request.text,
                voice=request.voice or self._selection.voice or DEFAULT_KITTEN_VOICE,
                speed=speed,
                **request_inputs,
            )
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_kitten_error(exc, self._selection, request) from exc
        adaptation = adapt_pcm_full_scale(result.audio, engine_label="KittenSynth")

        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "voice": result.voice,
                "model_ref": result.model_ref,
                "effective_speed": result.speed,
                "pcm_range": {
                    "policy": KITTEN_PCM_RANGE_POLICY,
                    "peak_before_clip": adaptation.peak_before_clip,
                    "clipped_samples": adaptation.clipped_samples,
                },
            }
        )
        warnings = (adaptation.warning,) if adaptation.warning else ()
        rendered = RenderedSpeech(
            id=request.id,
            audio=adaptation.audio,
            sample_rate=result.sample_rate,
            warnings=warnings,
            word_timings=(),
            metadata=metadata,
        )
        return validate_rendered_speech(
            request,
            rendered,
            engine="kitten",
            engine_version=distribution_version("kittensynth"),
            target_id=self._selection.target_id,
        )


class KittenSynthEngineAdapter:
    """Kitten model discovery, selection, runtime and request adaptation."""

    id = "kitten"
    package_name = "kittensynth"

    def version(self) -> str | None:
        return distribution_version(self.package_name)

    def probe_api(self) -> EngineApiProbe:
        probe = probe_public_api(
            engine=self.id,
            package=self.package_name,
            required_symbols=(
                "KittenVoice",
                "SynthesisMeasure",
                "SynthesisConfig",
                "SynthesisResult",
                "discover_models",
                "KittenG2PConfig",
                "G2LexFileRef",
                "LinguisticToken",
                "PronunciationOverride",
                "PronunciationTrace",
                "discover_lexicons",
                "inspect_g2lex_file",
                "runtime_identity",
            ),
            required_methods={
                "KittenVoice": (
                    "synthesize_prepared",
                    "measure_prepared",
                    "from_pretrained",
                    "from_local",
                    "close",
                ),
                "__module__": (
                    "discover_models",
                    "runtime_identity",
                    "discover_lexicons",
                    "inspect_g2lex_file",
                ),
            },
            expected_api_version=KITTEN_REQUEST_API_VERSION,
        )
        if not probe.compatible:
            diagnostic = (
                "KittenSynth does not support request measurement. "
                "Kitten lexicon-aware G2P requires kittensynth>=0.1.8 and "
                "kitteng2p>=0.1.2."
                if probe.failed_stage == "request_measurement_contract"
                else "Kitten lexicon-aware G2P requires kittensynth>=0.1.8 and "
                "kitteng2p>=0.1.2."
            )
            if (
                probe.failed_stage == "contract_inspection"
                and probe.failed_symbol == "request_api_contract"
            ):
                return replace(
                    probe,
                    failed_stage="pronunciation_capability_contract",
                    warnings=probe.warnings + (diagnostic,),
                )
            return replace(probe, warnings=probe.warnings + (diagnostic,))

        try:
            kittensynth = importlib.import_module(self.package_name)
            contract_factory = getattr(kittensynth, "request_api_contract", None)
            if not callable(contract_factory):
                raise TypeError("request_api_contract is not callable")
            contract = contract_factory()
        except Exception as exc:  # noqa: BLE001 - Contract inspection must fail closed.
            return replace(
                probe,
                compatible=False,
                status="api_probe_failed",
                failed_stage="pronunciation_capability_contract",
                failed_symbol="request_api_contract",
                error_type=type(exc).__name__,
                error_message=str(exc),
                warnings=probe.warnings
                + (
                    (
                        "Kitten lexicon-aware G2P requires kittensynth>=0.1.8 and "
                        "kitteng2p>=0.1.2."
                    ),
                ),
            )

        required_capabilities = (
            ("supports_request_measurement", "request_measurement_contract"),
            ("supports_linguistic_tokens", "pronunciation_capability_contract"),
            ("supports_pronunciation_overrides", "pronunciation_capability_contract"),
            ("supports_lexicons", "pronunciation_capability_contract"),
        )
        for name, stage in required_capabilities:
            value = (
                contract.get(name)
                if isinstance(contract, Mapping)
                else getattr(contract, name, None)
            )
            if value is True:
                continue
            shown_value = "missing" if value is None else value
            diagnostic = (
                "KittenSynth does not support request measurement. Install "
                "kittensynth>=0.1.8 with kitteng2p>=0.1.2."
                if name == "supports_request_measurement"
                else "KittenSynth does not advertise the required pronunciation "
                "capabilities. Install kittensynth>=0.1.8 with kitteng2p>=0.1.2."
            )
            return replace(
                probe,
                compatible=False,
                status="api_incompatible",
                failed_stage=stage,
                failed_symbol=name,
                warnings=probe.warnings + (diagnostic,),
                details={name: shown_value, "required": True},
            )
        return probe

    def compatible_api(self) -> bool:
        return self.probe_api().compatible

    def option_descriptors(
        self, *, target_id: str | None = None
    ) -> Mapping[str, EngineOption]:
        del target_id
        return {
            "lexicons": EngineOption(
                name="lexicons",
                value_type="string|array[string]",
                default=None,
                description="Ordered named Kitten-compatible IPA pronunciation sources.",
            ),
            "g2lex_files": EngineOption(
                name="g2lex_files",
                value_type="array[table]",
                default=None,
                description="Ordered local G2Lex files pinned by content and logical SHA-256.",
            ),
            "lexicon_data_policy": EngineOption(
                name="lexicon_data_policy",
                value_type="string",
                default="auto",
                choices=("auto", "installed-only"),
                description="Provision selected named sources automatically or require installation.",
            ),
            "heteronym_policy": EngineOption(
                name="heteronym_policy",
                value_type="string",
                default="auto",
                choices=("auto", "off"),
                description="Enable or disable deterministic English contextual heteronym rules.",
            ),
        }

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            voice_binding_namespace="kitten",
            voice_binding_scope="request",
            option_names=KITTEN_OPTION_NAMES,
            supports_named_voices=True,
            supports_reference_voice=False,
            supports_speakers=False,
            supports_pronunciation_overrides=True,
            pronunciation_alphabets=frozenset({"ipa"}),
            supports_linguistic_tokens=True,
            supports_whole_request_phonemes=False,
            supports_lexicons=True,
            supports_model_sources=False,
            supports_qualities=True,
            supports_live=True,
            supports_timestamps=False,
            supports_voice_level_calibration=False,
            supports_request_measurement=True,
        )

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            token_fields=("span", "pos", "tag", "lemma", "morph"),
            frontend="kittensynth",
            pronunciation="kitteng2p.ipa",
            capacity_key="max_tokens",
            capacity_metric="model_tokens",
        )

    def discover(self, request: CatalogRequest) -> tuple[SynthesisTarget, ...]:
        try:
            return _catalog_targets(request)
        except ImportError:
            return ()

    def discover_lexicons(
        self,
        *,
        language: str | None = None,
        model: str | None = None,
        offline: bool = False,
        refresh: bool = False,
        preference: str = "auto",
    ) -> tuple[tuple[Any, ...], Any]:
        import kittensynth

        from ..lexicons import LexiconCatalogEntry

        del model  # Kitteng2p's catalog exposes encoding-based Kitten compatibility.
        result = kittensynth.discover_lexicons(
            language=language or "en-us",
            offline=offline,
            refresh=refresh,
            preference=preference,
        )
        entries = tuple(
            LexiconCatalogEntry(
                selector=item.selector,
                engine=self.id,
                language=item.language,
                locale=normalize_locale_tag(item.language) or item.language,
                asset_id=item.asset_id,
                data_backend=item.backend,
                default=item.default,
                installed=item.installed,
                models=(),
                model_support="known" if item.model_support else "unsupported",
                display_name=item.display_name,
                phoneme_encoding=item.phoneme_encoding,
                data_version=item.data_version,
                content_sha256=item.content_sha256,
                logical_sha256=item.logical_sha256,
                compatibility_reason=item.compatibility_reason,
            )
            for item in result.entries
        )
        return entries, result

    def inspect_g2lex_file(self, path: str) -> Mapping[str, Any]:
        import kittensynth

        inspection = kittensynth.inspect_g2lex_file(path)
        if is_dataclass(inspection) and not isinstance(inspection, type):
            value = asdict(inspection)
        elif isinstance(inspection, Mapping):
            value = dict(inspection)
        else:
            raise TypeError("Kitteng2p returned an invalid G2Lex inspection record")
        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping):
            value["metadata"] = {}
        else:
            value["metadata"] = dict(metadata)
        return value

    def pronunciation_stack(
        self,
        *,
        language: str,
        target_id: str,
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        selected_options = dict(options or {})
        policy = selected_options.get("lexicon_data_policy", "auto")
        heteronym_policy = selected_options.get("heteronym_policy", "auto")
        selectors = _requested_selectors(selected_options.get("lexicons"))
        local_records: list[dict[str, Any]] = []
        for source in _source_entries(selected_options.get("g2lex_files")):
            raw_path = (
                source
                if isinstance(source, (str, Path))
                else _source_field(source, "path")
            )
            record: dict[str, Any] = {
                "path": str(raw_path) if raw_path is not None else None,
                "sha256": _source_field(source, "sha256"),
                "logical_sha256": _source_field(source, "logical_sha256"),
                "status": "invalid",
            }
            if isinstance(raw_path, (str, Path)):
                try:
                    inspection = self.inspect_g2lex_file(str(raw_path))
                except (ImportError, OSError, TypeError, ValueError) as exc:
                    record["error"] = str(exc)
                else:
                    record["path"] = inspection.get("path", str(raw_path))
                    record["current_sha256"] = inspection.get("sha256")
                    record["current_logical_sha256"] = inspection.get("logical_sha256")
                    record["language"] = inspection.get("language")
                    record["phoneme_encoding"] = inspection.get("phoneme_encoding")
                    if inspection.get("valid") is not True:
                        record["status"] = "invalid"
                        record["error"] = inspection.get("reason")
                    elif inspection.get("compatible") is not True:
                        record["status"] = "incompatible"
                        record["error"] = inspection.get("reason")
                    else:
                        expected = _source_field(source, "sha256")
                        expected_logical = _source_field(source, "logical_sha256")
                        actual = inspection.get("sha256")
                        actual_logical = inspection.get("logical_sha256")
                        invalid_pin = (
                            expected is not None and _sha256_key(expected) is None
                        ) or (
                            expected_logical is not None
                            and _sha256_key(expected_logical) is None
                        )
                        changed = (
                            _sha256_key(expected) is not None
                            and _sha256_key(expected) != _sha256_key(actual)
                        ) or (
                            _sha256_key(expected_logical) is not None
                            and _sha256_key(expected_logical)
                            != _sha256_key(actual_logical)
                        )
                        source_language = inspection.get("language")
                        try:
                            if not isinstance(source_language, str):
                                raise TypeError("G2Lex language must be a string")
                            source_key = normalize_language_key(source_language)
                            request_key = normalize_language_key(language)
                            language_matches = language_base(
                                source_key
                            ) == language_base(request_key)
                        except (TypeError, ValueError):
                            language_matches = False
                        record["status"] = (
                            "invalid-pin"
                            if invalid_pin
                            else "changed"
                            if changed
                            else "incompatible"
                            if not language_matches
                            else "valid"
                        )
                        if not language_matches:
                            record["error"] = (
                                "G2Lex language does not match the synthesis request."
                            )
            local_records.append(record)

        catalog_available = True
        catalog_error: str | None = None
        named_info: dict[str, Any] = {}
        if selectors:
            try:
                _entries, discovery = self.discover_lexicons(
                    language=language,
                    offline=True,
                    preference="auto",
                )
                named_info = {
                    item.selector.casefold(): item for item in discovery.entries
                }
            except (
                EngineError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                catalog_available = False
                catalog_error = str(exc)

        named_records: list[dict[str, Any]] = []
        for selector in selectors:
            item = named_info.get(selector.casefold())
            named_records.append(
                {
                    "selector": selector,
                    "available": item is not None,
                    "installed": None if item is None else item.installed,
                    "model_support": False if item is None else item.model_support,
                    "phoneme_encoding": None if item is None else item.phoneme_encoding,
                    "asset_id": None if item is None else item.asset_id,
                    "data_version": None if item is None else item.data_version,
                    "content_sha256": None if item is None else item.content_sha256,
                    "logical_sha256": None if item is None else item.logical_sha256,
                    "compatibility_reason": (
                        None if item is None else item.compatibility_reason
                    ),
                }
            )

        precedence = ["request-local IPA spans"]
        precedence.extend(
            f"local G2Lex file {index + 1}" for index in range(len(local_records))
        )
        precedence.extend(selectors)
        if heteronym_policy == "auto":
            precedence.append("contextual heteronym rules")
        precedence.append("eSpeak fallback")
        return {
            "engine": self.id,
            "language": language,
            "target_id": target_id,
            "precedence": precedence,
            "local_g2lex": local_records,
            "named_lexicons": named_records,
            "lexicon_catalog_available": catalog_available,
            "lexicon_catalog_error": catalog_error,
            "lexicon_data_policy": policy,
            "heteronym_policy": heteronym_policy,
            "g2p_fallback": "espeak",
        }

    def resolve(self, request: Any) -> tuple[EngineSelection, tuple[Any, ...]]:
        language = normalize_language_key(getattr(request, "language", None) or "en-us")
        if not _is_english(language):
            raise KittenSelectionError(
                "kitten.language_incompatible",
                f"KittenSynth supports English only, not {language!r}.",
                "synthesis.language",
            )
        options = dict(getattr(request, "options", {}) or {})
        options.update(dict(getattr(request, "engine_options", {}) or {}))
        speed = options.get("speed", 1.0)
        try:
            speed = float(speed)
        except (TypeError, ValueError) as exc:
            raise KittenSelectionError(
                "kitten.invalid_speed",
                "Kitten speed must be finite and greater than zero.",
                "synthesis.speed",
            ) from exc
        if not math.isfinite(speed) or speed <= 0:
            raise KittenSelectionError(
                "kitten.invalid_speed",
                "Kitten speed must be finite and greater than zero.",
                "synthesis.speed",
            )
        options["speed"] = speed
        target_id = (
            getattr(request, "target_id", None)
            or options.pop("model", None)
            or DEFAULT_KITTEN_MODEL
        )
        voice = (
            getattr(request, "voice", None)
            or options.pop("voice", None)
            or DEFAULT_KITTEN_VOICE
        )
        return (
            EngineSelection(
                engine=self.id,
                target_id=str(target_id),
                language=language,
                voice=str(voice),
                speaker=getattr(request, "speaker", None),
                options={
                    key: value
                    for key, value in options.items()
                    if key in KITTEN_OPTION_NAMES
                },
                offline=bool(getattr(request, "offline", False)),
                refresh=bool(getattr(request, "refresh", False)),
            ),
            (),
        )

    def validate_selection(self, selection: EngineSelection) -> tuple[Any, ...]:
        from .base import EngineDiagnostic

        if not _is_english(selection.language):
            return (
                EngineDiagnostic(
                    code="kitten.language_incompatible",
                    severity="error",
                    message=f"KittenSynth supports English only, not {selection.language!r}.",
                    field="synthesis.language",
                ),
            )
        try:
            targets = self.discover(
                CatalogRequest(
                    engine=self.id,
                    language=selection.language,
                    offline=selection.offline,
                    refresh=selection.refresh,
                )
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return (
                EngineDiagnostic(
                    code="kitten.engine_unavailable",
                    severity="error",
                    message=f"Kitten model catalog is unavailable: {exc}",
                    field="synthesis.engine",
                ),
            )
        target = next(
            (
                item
                for item in targets
                if item.id == selection.target_id or selection.target_id in item.aliases
            ),
            None,
        )
        if target is None:
            return (
                EngineDiagnostic(
                    code="kitten.model_not_found",
                    severity="error",
                    message=f"Kitten model {selection.target_id!r} was not found in the Kitten catalog.",
                    field="render.target.id",
                ),
            )
        if target.engine != "kitten":
            return (
                EngineDiagnostic(
                    code="kitten.model_not_found",
                    severity="error",
                    message=f"Catalog target {target.id!r} does not belong to the Kitten system.",
                    field="render.target.id",
                ),
            )
        if not target.sample_rate or target.sample_rate <= 0:
            return (
                EngineDiagnostic(
                    code="kitten.model_not_found",
                    severity="error",
                    message=f"Kitten model {target.id!r} has no valid sample rate.",
                    field="render.target.id",
                ),
            )
        if selection.voice not in target.voices:
            return (
                EngineDiagnostic(
                    code="kitten.voice_unavailable",
                    severity="error",
                    message=f"Voice {selection.voice!r} is not available for Kitten model {target.id!r}.",
                    field="render.target.voice",
                ),
            )
        quality = selection.options.get("quality")
        if quality is not None and quality not in target.qualities:
            return (
                EngineDiagnostic(
                    code="kitten.quality_unavailable",
                    severity="error",
                    message=f"Quality {quality!r} is not available for Kitten model {target.id!r}.",
                    field="render.options.quality",
                ),
            )
        return self._validate_pronunciation_sources(selection)

    def _validate_pronunciation_sources(
        self, selection: EngineSelection
    ) -> tuple[Any, ...]:
        from .base import EngineDiagnostic

        options = selection.options
        diagnostics: list[EngineDiagnostic] = []

        def add(code: str, message: str, field: str) -> None:
            diagnostics.append(
                EngineDiagnostic(
                    code=code,
                    severity="error",
                    message=message,
                    field=field,
                )
            )

        policy = options.get("lexicon_data_policy", "auto")
        if not isinstance(policy, str) or policy not in {"auto", "installed-only"}:
            add(
                "kitten.lexicon_policy_invalid",
                "lexicon_data_policy must be 'auto' or 'installed-only'.",
                "render.options.lexicon_data_policy",
            )
        heteronym_policy = options.get("heteronym_policy", "auto")
        if not isinstance(heteronym_policy, str) or heteronym_policy not in {
            "auto",
            "off",
        }:
            add(
                "kitten.heteronym_policy_invalid",
                "heteronym_policy must be 'auto' or 'off'.",
                "render.options.heteronym_policy",
            )
        try:
            selectors = _requested_selectors(options.get("lexicons"))
        except (TypeError, ValueError) as exc:
            add(
                "kitten.lexicon_selector_invalid",
                str(exc),
                "render.options.lexicons",
            )
            selectors = ()
        if len({selector.casefold() for selector in selectors}) != len(selectors):
            add(
                "kitten.lexicon_selector_duplicate",
                "Named lexicon selectors must be unique, ignoring case.",
                "render.options.lexicons",
            )

        local_sources = _source_entries(options.get("g2lex_files"))
        seen_paths: set[str] = set()
        for index, source in enumerate(local_sources):
            field = f"render.options.g2lex_files[{index}]"
            raw_path = (
                source
                if isinstance(source, (str, Path))
                else _source_field(source, "path")
            )
            if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
                add(
                    "kitten.g2lex_file_invalid",
                    "G2Lex file reference requires a path.",
                    field,
                )
                continue
            path = Path(raw_path).expanduser()
            path_key = str(path.resolve()).casefold()
            if path_key in seen_paths:
                add(
                    "kitten.g2lex_file_duplicate",
                    "G2Lex file paths must be unique.",
                    field,
                )
                continue
            seen_paths.add(path_key)
            try:
                inspection = self.inspect_g2lex_file(str(path))
            except (ImportError, OSError, TypeError, ValueError) as exc:
                add(
                    "kitten.g2lex_file_invalid",
                    f"Cannot inspect G2Lex file: {exc}",
                    field,
                )
                continue
            if inspection.get("valid") is not True:
                add(
                    "kitten.g2lex_file_invalid",
                    str(
                        inspection.get("reason") or "File is not a valid G2Lex lexicon."
                    ),
                    field,
                )
                continue
            source_language = inspection.get("language")
            try:
                if not isinstance(source_language, str):
                    raise TypeError("G2Lex language must be a string")
                source_key = normalize_language_key(source_language)
                request_key = normalize_language_key(selection.language)
                language_matches = language_base(source_key) == language_base(
                    request_key
                )
            except (TypeError, ValueError):
                language_matches = False
            if not language_matches:
                add(
                    "kitten.g2lex_language_incompatible",
                    f"G2Lex file language {source_language!r} does not match "
                    f"{selection.language!r}.",
                    field,
                )
            encoding = inspection.get("phoneme_encoding")
            if not isinstance(encoding, str) or encoding.casefold() != "ipa":
                add(
                    "kitten.g2lex_encoding_incompatible",
                    f"Kitten requires IPA G2Lex values, not {encoding!r}.",
                    field,
                )
            if inspection.get("compatible") is not True:
                reason = inspection.get("reason")
                add(
                    "kitten.g2lex_incompatible",
                    str(reason)
                    if isinstance(reason, str) and reason
                    else "G2Lex file is not compatible with KittenSynth.",
                    field,
                )
            for pin_name, actual_name in (
                ("sha256", "sha256"),
                ("logical_sha256", "logical_sha256"),
            ):
                expected = _source_field(source, pin_name)
                if expected is None:
                    continue
                expected_hash = _sha256_key(expected)
                actual_hash = _sha256_key(inspection.get(actual_name))
                if expected_hash is None:
                    add(
                        "kitten.g2lex_pin_invalid",
                        f"G2Lex {pin_name} pin must be a SHA-256 digest.",
                        field,
                    )
                elif expected_hash != actual_hash:
                    add(
                        "kitten.g2lex_pin_changed",
                        f"G2Lex file no longer matches its pinned {pin_name}.",
                        field,
                    )

        if selectors:
            try:
                _entries, discovery = self.discover_lexicons(
                    language=selection.language,
                    offline=selection.offline,
                    refresh=selection.refresh,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                add(
                    "kitten.lexicon_catalog_unavailable",
                    f"Cannot verify selected Kitten lexicons: {exc}",
                    "render.options.lexicons",
                )
                return tuple(diagnostics)
            by_selector = {item.selector.casefold(): item for item in discovery.entries}
            for index, selector in enumerate(selectors):
                field = f"render.options.lexicons[{index}]"
                entry = by_selector.get(selector.casefold())
                if entry is None:
                    add(
                        "kitten.lexicon_not_found",
                        f"Named lexicon {selector!r} is unavailable for {selection.language!r}.",
                        field,
                    )
                    continue
                try:
                    source_key = normalize_language_key(entry.language)
                    request_key = normalize_language_key(selection.language)
                    language_matches = language_base(source_key) == language_base(
                        request_key
                    )
                except (TypeError, ValueError):
                    language_matches = False
                if not language_matches:
                    add(
                        "kitten.lexicon_language_incompatible",
                        f"Named lexicon {selector!r} language {entry.language!r} does not "
                        f"match {selection.language!r}.",
                        field,
                    )
                if (
                    entry.model_support not in {"known", "supported"}
                    or not isinstance(entry.phoneme_encoding, str)
                    or entry.phoneme_encoding.casefold() != "ipa"
                ):
                    add(
                        "kitten.lexicon_incompatible",
                        entry.compatibility_reason
                        or f"Named lexicon {selector!r} is not Kitten-compatible IPA.",
                        field,
                    )
                if entry.installed is not True and (
                    policy == "installed-only" or selection.offline
                ):
                    add(
                        "kitten.lexicon_not_installed",
                        f"Named lexicon {selector!r} is not installed and cannot be "
                        "provisioned under the current policy/offline setting.",
                        field,
                    )
        return tuple(diagnostics)

    def target_metadata(self, selection: EngineSelection) -> Mapping[str, Any]:
        targets = self.discover(
            CatalogRequest(
                engine=self.id,
                language=selection.language,
                offline=selection.offline,
                refresh=selection.refresh,
            )
        )
        target = next(
            (
                item
                for item in targets
                if item.id == selection.target_id or selection.target_id in item.aliases
            ),
            None,
        )
        if target is None:
            return {}
        return {
            **dict(target.metadata),
            "sample_rate": target.sample_rate,
            "voices": target.voices,
            "qualities": target.qualities,
            "source_revision": target.metadata.get("source_revision"),
        }

    def _pronunciation_identity(self, selection: EngineSelection) -> Mapping[str, Any]:
        import kittensynth

        options = selection.options
        selectors = _requested_selectors(options.get("lexicons"))
        named_identity: list[dict[str, Any]] = []
        if selectors:
            try:
                _entries, discovery = self.discover_lexicons(
                    language=selection.language,
                    offline=selection.offline,
                    refresh=selection.refresh,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                raise InvalidEngineOptionError(
                    f"Cannot identify selected Kitten lexicons: {exc}",
                    engine="kitten",
                    language=selection.language,
                    code="kitten.lexicon_identity_unavailable",
                ) from exc
            by_selector = {item.selector.casefold(): item for item in discovery.entries}
            for index, selector in enumerate(selectors):
                item = by_selector.get(selector.casefold())
                if item is None:
                    raise InvalidEngineOptionError(
                        f"Selected Kitten lexicon {selector!r} is unavailable.",
                        engine="kitten",
                        language=selection.language,
                        code="kitten.lexicon_identity_unavailable",
                    )
                named_identity.append(
                    {
                        "order": index,
                        "selector": item.selector,
                        "asset_id": item.asset_id or item.id,
                        "data_version": item.data_version,
                        "sha256": item.content_sha256,
                        "logical_sha256": item.logical_sha256,
                    }
                )

        local_identity: list[dict[str, Any]] = []
        for index, source in enumerate(_source_entries(options.get("g2lex_files"))):
            raw_path = (
                source
                if isinstance(source, (str, Path))
                else _source_field(source, "path")
            )
            if not isinstance(raw_path, (str, Path)):
                raise InvalidEngineOptionError(
                    "Each local G2Lex source must include a file path.",
                    engine="kitten",
                    language=selection.language,
                    code="kitten.g2lex_reference_invalid",
                )
            inspection = self.inspect_g2lex_file(str(raw_path))
            if inspection.get("valid") is not True:
                raise InvalidEngineOptionError(
                    str(inspection.get("reason") or "Local G2Lex source is invalid."),
                    engine="kitten",
                    language=selection.language,
                    code="kitten.g2lex_file_invalid",
                )
            local_identity.append(
                {
                    "order": index,
                    "sha256": inspection.get("sha256"),
                    "logical_sha256": inspection.get("logical_sha256"),
                }
            )

        runtime = dict(kittensynth.runtime_identity())
        return {
            "named_lexicons": named_identity,
            "local_g2lex": local_identity,
            "heteronym_policy": options.get("heteronym_policy", "auto"),
            "frontend_identity": {
                "kitteng2p_revision": runtime.get("g2p_revision"),
                "request_api_version": runtime.get("request_api_version"),
            },
        }

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> Mapping[str, Any]:
        import kittensynth

        pcm_options = {
            key: selection.options[key]
            for key in (
                "quality",
                "speed",
                "providers",
                "provider_options",
                "session_options",
            )
            if key in selection.options
        }
        return {
            "engine": self.id,
            "engine_version": self.version(),
            "engine_identity": dict(kittensynth.runtime_identity()),
            "target_id": selection.target_id,
            "voice": selection.voice,
            "language": selection.language,
            "options": pcm_options,
            "pronunciation_identity": self._pronunciation_identity(selection),
            "model_revision": selection.metadata.get("source_revision"),
            "pcm_range_policy": KITTEN_PCM_RANGE_POLICY,
        }

    def open(
        self, selection: EngineSelection
    ) -> AbstractContextManager[KittenSynthEngineSession]:
        import kittensynth

        options = dict(selection.options)
        try:
            lexicon_policy = options.get("lexicon_data_policy", "auto")
            if selection.offline and lexicon_policy == "auto":
                lexicon_policy = "installed-only"
            g2p_config = kittensynth.KittenG2PConfig(
                language=selection.language,
                lexicons=options.get("lexicons"),
                g2lex_files=options.get("g2lex_files"),
                lexicon_data_policy=lexicon_policy,
                heteronym_policy=options.get("heteronym_policy", "auto"),
            )
            voice = kittensynth.KittenVoice.from_pretrained(
                selection.target_id,
                quality=options.get("quality"),
                cache_dir=options.get("cache_dir"),
                offline=selection.offline,
                refresh_catalog=selection.refresh,
                force_download=bool(options.get("force_download", False)),
                providers=options.get("providers"),
                provider_options=options.get("provider_options"),
                session_options=options.get("session_options"),
                g2p_config=g2p_config,
            )
        except Exception as exc:
            raise _translate_kitten_error(exc, selection, opening=True) from exc

        @contextmanager
        def session() -> Any:
            try:
                yield KittenSynthEngineSession(voice, selection)
            finally:
                voice.close()

        return session()


__all__ = [
    "DEFAULT_KITTEN_MODEL",
    "DEFAULT_KITTEN_VOICE",
    "KittenSynthEngineAdapter",
    "KittenSynthEngineSession",
]
