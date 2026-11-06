"""Request-centric adapter for the public PhononSynth prepared-text API."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from importlib import import_module
from typing import Any

from ..errors import (
    EmptySpeechTextError,
    EngineBackendError,
    EngineError,
    EngineSynthesisError,
    InvalidEngineModelError,
    InvalidEngineOptionError,
    InvalidEngineVoiceError,
    UnsupportedSynthesisFeatureError,
)
from .api_probe import EngineApiProbe, probe_public_api
from .base import (
    PCM_RANGE_POLICY,
    EngineCapabilities,
    EngineDiagnostic,
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
    language_tags_match,
    normalize_gender,
    normalize_locale_tag,
)
from .profiles import make_renderprofile_fragment

ENGINE_ID = "phonon"
PACKAGE_NAME = "phononsynth"
PHONONSYNTH_REQUEST_API_VERSION = 2
_PHONON_API_COMPATIBILITY_MESSAGE = (
    "PhononSynth request API is incompatible. VoiceRender requires the "
    "authoritative request-measurement contract. Install "
    "phononsynth>=0.1.2,<0.2."
)
PHONON_OPTION_NAMES = frozenset(
    {
        "quality",
        "duration_delta",
        "temperature",
        "seed",
        "eos_threshold",
        "frames_after_eos",
        "min_frames_before_eos",
        "max_frames",
        "cache_dir",
        "catalog_url",
        "providers",
        "provider_options",
        "session_options",
        "force_download",
    }
)
_GENERATION_OPTION_NAMES = (
    "duration_delta",
    "temperature",
    "seed",
    "eos_threshold",
    "frames_after_eos",
    "min_frames_before_eos",
    "max_frames",
)
_UNSUPPORTED_OPTION_FIELDS = frozenset(
    {
        "voice_source",
        "reference_audio",
        "reference_voice",
        "model_source",
        "model_path",
        "local_model",
        "local_model_path",
        "bundle_dir",
    }
)


def _module() -> Any:
    """Import PhononSynth only for an operation that needs its public API."""
    return import_module(PACKAGE_NAME)


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _contract_field(contract: Any, name: str, default: Any = None) -> Any:
    return _value(contract, name, default)


def _target_voice(voice: Any, target_languages: tuple[str, ...]) -> TargetVoice:
    raw_language = normalize_locale_tag(_value(voice, "language"))
    locale = normalize_locale_tag(_value(voice, "locale"))
    languages = tuple(
        dict.fromkeys(
            base
            for raw in (_value(voice, "languages", ()) or target_languages)
            if (base := language_base(raw))
        )
    )
    language = language_base(raw_language) or (languages[0] if languages else "unknown")
    language_label = str(
        _value(voice, "language_label")
        or _value(voice, "display_name")
        or locale
        or language
    )
    return TargetVoice(
        id=str(_value(voice, "id", "")),
        gender=normalize_gender(_value(voice, "gender")),
        language=language,
        locale=locale or raw_language or language,
        language_label=language_label or "unknown",
        languages=languages or target_languages,
    )


def _target_from_model(model: Any) -> SynthesisTarget:
    languages = tuple(
        dict.fromkeys(
            base
            for raw in (_value(model, "languages", ()) or ())
            if (base := language_base(raw))
        )
    )
    raw_voices = tuple(_value(model, "voices", ()) or ())
    voice_details = tuple(_target_voice(voice, languages) for voice in raw_voices)
    voices = tuple(detail.id for detail in voice_details)
    upstream_metadata = _value(model, "metadata", {})
    metadata = dict(upstream_metadata) if isinstance(upstream_metadata, Mapping) else {}
    release_tags = _value(model, "release_tags", {})
    source_revisions = _value(model, "source_revisions", {})
    release_tags = dict(release_tags) if isinstance(release_tags, Mapping) else {}
    source_revisions = (
        dict(source_revisions) if isinstance(source_revisions, Mapping) else {}
    )
    metadata.update(
        {
            "source": PACKAGE_NAME,
            "source_revision": _value(model, "source_revision"),
            "release_version": _value(model, "release_version"),
            "release_tags": release_tags,
            "source_revisions": source_revisions,
            "license_spdx": _value(model, "license_spdx"),
            "commercial_use": _value(model, "commercial_use"),
            "languages": languages,
            "qualities": tuple(_value(model, "quality_choices", ()) or ()),
            "voice_display_names": {
                detail.id: str(_value(voice, "display_name") or detail.id)
                for voice, detail in zip(raw_voices, voice_details, strict=True)
            },
        }
    )
    default_voice = _value(model, "default_voice")
    return SynthesisTarget(
        engine=ENGINE_ID,
        id=str(_value(model, "id", "")),
        display_name=str(_value(model, "display_name") or _value(model, "id", "")),
        languages=languages,
        status="ready",
        runtime_available=True,
        sample_rate=_value(model, "sample_rate"),
        voices=voices,
        voice_details=voice_details,
        default_voice=str(default_voice) if isinstance(default_voice, str) else None,
        qualities=tuple(
            str(value) for value in (_value(model, "quality_choices", ()) or ())
        ),
        capabilities=frozenset({"predefined_voice"}),
        metadata=metadata,
    )


def _target_catalog(
    *,
    language: str | None,
    offline: bool,
    refresh: bool,
    options: Mapping[str, Any] | None = None,
) -> tuple[SynthesisTarget, ...]:
    module = _module()
    values = options or {}
    requested_language = language_base(language) if language else None
    models = module.discover_models(
        language=requested_language,
        offline=offline,
        refresh=refresh,
        cache_dir=values.get("cache_dir"),
        catalog_url=values.get("catalog_url"),
    )
    targets = tuple(_target_from_model(model) for model in models)
    if not language:
        return targets
    normalized = normalize_locale_tag(language)
    return tuple(
        target
        for target in targets
        if any(language_tags_match(normalized, item) for item in target.languages)
    )


def _find_target(
    targets: tuple[SynthesisTarget, ...], target_id: str
) -> SynthesisTarget | None:
    requested = target_id.casefold()
    return next(
        (target for target in targets if target.id.casefold() == requested), None
    )


def _selection_diagnostic(code: str, message: str, field: str) -> EngineDiagnostic:
    return EngineDiagnostic(code=code, severity="error", message=message, field=field)


class PhononSelectionError(ValueError):
    """Selection failure with a stable machine-readable diagnostic code."""

    def __init__(
        self, code: str, message: str, field: str = "synthesis.engine"
    ) -> None:
        super().__init__(message)
        self.diagnostic_code = code
        self.diagnostic_field = field


def _error_context(
    error: Exception,
    selection: EngineSelection,
    request: SpeechRequest | None,
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


def _validated_measurement(measured: Any, selection: EngineSelection) -> RequestMeasure:
    amount = _value(measured, "token_count")
    maximum = _value(measured, "max_tokens")
    fits = _value(measured, "fits")

    valid_amount = (
        isinstance(amount, int) and not isinstance(amount, bool) and amount >= 0
    )
    valid_maximum = maximum is None or (
        isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0
    )
    if not valid_amount or not isinstance(fits, bool) or not valid_maximum:
        raise EngineBackendError(
            "PhononSynth returned an invalid request measurement.",
            engine=ENGINE_ID,
            target_id=selection.target_id,
            code="phonon.measurement_invalid",
            details={"token_count": amount, "max_tokens": maximum, "fits": fits},
        )

    if maximum is not None and fits is not (amount <= maximum):
        raise EngineBackendError(
            "PhononSynth request measurement contradicts its maximum.",
            engine=ENGINE_ID,
            target_id=selection.target_id,
            code="phonon.measurement_invalid",
            details={"token_count": amount, "max_tokens": maximum, "fits": fits},
        )

    return RequestMeasure(
        fits=fits,
        amount=amount,
        maximum=maximum,
        unit="model_tokens",
        source="phononsynth.measure_prepared",
        details={"model_id": _value(measured, "model_id") or selection.target_id},
    )


def _translate_phonon_error(
    error: Exception,
    selection: EngineSelection,
    request: SpeechRequest | None = None,
    *,
    opening: bool = False,
) -> EngineError:
    """Translate public PhononSynth failures without inventing capacity semantics."""
    try:
        module = _module()
    except ImportError:
        module = None
    error_map: tuple[tuple[str, type[EngineError], str], ...] = (
        ("EmptyTextError", EmptySpeechTextError, "phonon.empty_text"),
        (
            "InvalidSynthesisConfigError",
            InvalidEngineOptionError,
            "phonon.invalid_options",
        ),
        ("UnsupportedModelError", InvalidEngineModelError, "phonon.model_unsupported"),
        ("CatalogUnavailableError", EngineBackendError, "phonon.catalog_unavailable"),
        ("CatalogDiscoveryError", EngineBackendError, "phonon.catalog_error"),
        ("TokenizerError", EngineBackendError, "phonon.tokenizer_error"),
        (
            "ManagedModelDependencyError",
            EngineBackendError,
            "phonon.runtime_dependency_error",
        ),
        ("OnnxVoiceContractError", EngineBackendError, "phonon.runtime_contract_error"),
        ("ModelInferenceError", EngineBackendError, "phonon.inference_failed"),
    )
    mapped: tuple[type[EngineError], str] | None = None
    if module is not None:
        for native_name, error_type, code in error_map:
            native_type = getattr(module, native_name, None)
            if isinstance(native_type, type) and isinstance(error, native_type):
                mapped = error_type, code
                break
    if mapped is None and isinstance(error, ValueError):
        message = str(error).casefold()
        if "unknown phonon voice" in message or "no default phonon voice" in message:
            mapped = InvalidEngineVoiceError, "phonon.voice_not_found"
        elif opening:
            mapped = InvalidEngineOptionError, "phonon.invalid_options"
    error_type, code = mapped or (EngineBackendError, "phonon.backend_error")
    return error_type(
        str(error),
        **_error_context(error, selection, request),
        code=code,
    )


def _serializable_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    from ..jsonutil import json_value

    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        try:
            result[key] = json_value(item, label=f"PhononSynth metadata.{key}")
        except (TypeError, ValueError):
            continue
    return result


class PhononSynthEngineSession:
    """Adapt exact prepared-text requests to one managed PhononSynth voice."""

    def __init__(self, voice: Any, selection: EngineSelection, config: Any) -> None:
        self._voice = voice
        self._selection = selection
        self._config = config

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        try:
            measured = self._voice.measure_prepared(request.text)
        except EngineSynthesisError:
            raise
        except Exception as exc:  # Backend errors are normalized at this boundary.
            raise _translate_phonon_error(exc, self._selection, request) from exc
        return _validated_measurement(measured, self._selection)

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        unsupported = (
            (request.tokens, "linguistic tokens"),
            (request.pronunciation_overrides, "pronunciation overrides"),
            (request.say_as, "say_as spans"),
            (request.whole_request_phonemes is not None, "whole-request phonemes"),
            (request.speaker is not None, "speaker selection"),
        )
        for present, label in unsupported:
            if present:
                raise UnsupportedSynthesisFeatureError(
                    f"PhononSynth does not support {label}",
                    engine=ENGINE_ID,
                    target_id=self._selection.target_id,
                    request_id=request.id,
                )
        voice = request.voice or self._selection.voice
        try:
            result = self._voice.synthesize_prepared(
                request.text,
                voice=voice,
                config=self._config,
            )
        except EngineSynthesisError:
            raise
        except Exception as exc:  # Backend errors are normalized at this boundary.
            raise _translate_phonon_error(exc, self._selection, request) from exc

        result_metadata = _serializable_mapping(_value(result, "metadata", {}))
        quality = self._selection.options.get("quality")
        source_revision = self._selection.metadata.get("source_revision")
        release_version = self._selection.metadata.get("release_version")
        metadata: dict[str, Any] = {
            "engine": ENGINE_ID,
            "target_id": self._selection.target_id,
            "model_ref": _value(result, "model_ref"),
            "voice": _value(result, "voice") or voice,
            "quality": quality,
            "source_revision": source_revision,
            "release_version": release_version,
            "release_tags": _serializable_mapping(
                self._selection.metadata.get("release_tags", {})
            ),
            "source_revisions": _serializable_mapping(
                self._selection.metadata.get("source_revisions", {})
            ),
            "generation": {
                name: getattr(self._config, name, None)
                for name in _GENERATION_OPTION_NAMES
            },
            "phononsynth": result_metadata,
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        adaptation = adapt_pcm_full_scale(
            _value(result, "audio"), engine_label="PhononSynth"
        )
        metadata["pcm_range"] = {
            "policy": PCM_RANGE_POLICY,
            "peak_before_clip": adaptation.peak_before_clip,
            "clipped_samples": adaptation.clipped_samples,
        }
        raw_warnings = _value(result, "warnings", ()) or ()
        warnings = tuple(item for item in raw_warnings if isinstance(item, str))
        if adaptation.warning:
            warnings += (adaptation.warning,)
        rendered = RenderedSpeech(
            id=request.id,
            audio=adaptation.audio,
            sample_rate=_value(result, "sample_rate"),
            warnings=warnings,
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


class PhononSynthEngineAdapter:
    """Phonon model discovery, named-voice selection, and prepared-text synthesis."""

    id = ENGINE_ID
    package_name = PACKAGE_NAME

    def version(self) -> str | None:
        return distribution_version(self.package_name)

    def probe_api(self) -> EngineApiProbe:
        probe = probe_public_api(
            engine=self.id,
            package=self.package_name,
            required_symbols=(
                "PhononVoice",
                "SynthesisConfig",
                "SynthesisResult",
                "SynthesisMeasure",
                "DiscoveredModel",
                "DescribedVoice",
                "discover_models",
                "runtime_identity",
                "REQUEST_API_VERSION",
                "request_api_contract",
            ),
            required_methods={
                "PhononVoice": (
                    "from_pretrained",
                    "measure_prepared",
                    "synthesize_prepared",
                    "close",
                ),
                "__module__": ("discover_models", "runtime_identity"),
            },
            expected_api_version=PHONONSYNTH_REQUEST_API_VERSION,
            require_explicit_contract=True,
        )
        if not probe.compatible:
            if probe.status == "api_version_incompatible":
                return replace(
                    probe,
                    error_message=_PHONON_API_COMPATIBILITY_MESSAGE,
                    warnings=probe.warnings + (_PHONON_API_COMPATIBILITY_MESSAGE,),
                )
            return probe
        try:
            contract = _module().request_api_contract()
            features = _contract_field(contract, "features", {})
            declared = {
                "engine": _contract_field(contract, "engine"),
                "entrypoint": _contract_field(contract, "entrypoint"),
                "measure_entrypoint": _contract_field(contract, "measure_entrypoint"),
                "input": _contract_field(contract, "input"),
                "measurement_unit": _contract_field(contract, "measurement_unit"),
                "measurement_fit_is_authoritative": _contract_field(
                    contract, "measurement_fit_is_authoritative"
                ),
                "named_voices": _contract_field(features, "named_voices"),
                "voice_cloning": _contract_field(features, "voice_cloning"),
                "word_timings": _contract_field(features, "word_timings"),
                "automatic_chunking": _contract_field(features, "automatic_chunking"),
            }
        except Exception as exc:  # noqa: BLE001 - Contract inspection must fail closed.
            return replace(
                probe,
                compatible=False,
                status="api_probe_failed",
                failed_stage="request_api_contract",
                failed_symbol="request_api_contract",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        expected = {
            "engine": ENGINE_ID,
            "entrypoint": "PhononVoice.synthesize_prepared",
            "measure_entrypoint": "PhononVoice.measure_prepared",
            "input": "prepared_text",
            "measurement_unit": "model_tokens",
            "measurement_fit_is_authoritative": True,
            "named_voices": True,
            "voice_cloning": False,
            "word_timings": False,
            "automatic_chunking": False,
        }
        contradictions = {
            key: {"expected": expected[key], "actual": value}
            for key, value in declared.items()
            if value != expected[key]
        }
        if contradictions:
            return replace(
                probe,
                compatible=False,
                status="api_incompatible",
                failed_stage="request_api_contract",
                failed_symbol="request_api_contract",
                error_message=_PHONON_API_COMPATIBILITY_MESSAGE,
                warnings=probe.warnings + (_PHONON_API_COMPATIBILITY_MESSAGE,),
                details={**dict(probe.details), "contract_mismatches": contradictions},
            )
        return replace(
            probe, details={**dict(probe.details), "request_api_contract": declared}
        )

    def compatible_api(self) -> bool:
        return self.probe_api().compatible

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            voice_binding_namespace=ENGINE_ID,
            voice_binding_scope="target",
            option_names=PHONON_OPTION_NAMES,
            supports_named_voices=True,
            supports_reference_voice=False,
            supports_speakers=False,
            supports_pronunciation_overrides=False,
            supports_say_as=False,
            supports_linguistic_tokens=False,
            supports_whole_request_phonemes=False,
            supports_lexicons=False,
            supports_model_sources=False,
            supports_qualities=True,
            supports_live=False,
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
            frontend=PACKAGE_NAME,
            pronunciation="unsupported",
        )

    def discover(self, request: CatalogRequest) -> tuple[SynthesisTarget, ...]:
        try:
            return _target_catalog(
                language=request.language,
                offline=request.offline,
                refresh=request.refresh,
            )
        except Exception as exc:  # Discovery failures have a stable public error.
            raise EngineBackendError(
                f"PhononSynth model catalog is unavailable: {exc}",
                engine=ENGINE_ID,
                native_error_type=type(exc).__name__,
                code="phonon.catalog_unavailable",
            ) from exc

    def resolve(self, request: Any) -> tuple[EngineSelection, tuple[Any, ...]]:
        options = dict(getattr(request, "options", {}) or {})
        engine_options = dict(getattr(request, "engine_options", {}) or {})
        for name in _UNSUPPORTED_OPTION_FIELDS:
            if options.get(name) is not None or engine_options.get(name) is not None:
                raise PhononSelectionError(
                    "phonon.model_source_unsupported"
                    if name
                    in _UNSUPPORTED_OPTION_FIELDS
                    - {
                        "voice_source",
                        "reference_audio",
                        "reference_voice",
                    }
                    else "phonon.reference_voice_unsupported",
                    f"PhononSynth does not support selection field {name!r}.",
                    "synthesis.engine_options",
                )
        unknown = sorted(set(engine_options) - PHONON_OPTION_NAMES)
        if unknown:
            raise PhononSelectionError(
                "phonon.option_unsupported",
                f"Unsupported PhononSynth option(s): {', '.join(unknown)}.",
                "synthesis.engine_options",
            )
        options.update(engine_options)
        speaker = getattr(request, "speaker", None)
        if speaker is not None:
            raise PhononSelectionError(
                "phonon.speaker_unsupported",
                "PhononSynth does not support speaker selection.",
                "render.target.speaker",
            )
        offline = bool(getattr(request, "offline", False))
        refresh = bool(getattr(request, "refresh", False))
        language = normalize_locale_tag(getattr(request, "language", None))
        requested_target_id = getattr(request, "target_id", None)
        try:
            targets = _target_catalog(
                language=None,
                offline=offline,
                refresh=refresh,
                options=options,
            )
        except Exception as exc:  # Catalog failures become stable selection errors.
            raise PhononSelectionError(
                "phonon.catalog_unavailable",
                f"PhononSynth model catalog is unavailable: {exc}",
                "synthesis.engine",
            ) from exc
        if requested_target_id:
            target = _find_target(targets, str(requested_target_id))
            if target is None:
                raise PhononSelectionError(
                    "phonon.model_not_found",
                    f"PhononSynth model {requested_target_id!r} was not found.",
                    "render.target.id",
                )
        elif len(targets) == 1:
            target = targets[0]
        else:
            raise PhononSelectionError(
                "phonon.model_required",
                "Select a PhononSynth model explicitly; no single unambiguous default exists.",
                "render.target.id",
            )
        if (
            language
            and target.languages
            and not any(
                language_tags_match(language, available)
                for available in target.languages
            )
        ):
            raise PhononSelectionError(
                "engine_language_incompatible",
                f"PhononSynth model {target.id!r} supports {', '.join(target.languages)}, "
                f"not {language!r}.",
                "render.target.language",
            )
        language = language or (target.languages[0] if target.languages else "en-us")
        quality = options.get("quality")
        if quality is not None and quality not in target.qualities:
            raise PhononSelectionError(
                "phonon.quality_unavailable",
                f"Quality {quality!r} is not available on PhononSynth model {target.id!r}.",
                "render.options.quality",
            )
        voice = getattr(request, "voice", None) or target.default_voice
        if not target.voices:
            raise PhononSelectionError(
                "phonon.voice_required",
                f"PhononSynth model {target.id!r} does not publish a named voice.",
                "render.target.voice",
            )
        if not voice:
            raise PhononSelectionError(
                "phonon.voice_required",
                f"Select a named voice for PhononSynth model {target.id!r}.",
                "render.target.voice",
            )
        if voice not in target.voices:
            raise PhononSelectionError(
                "phonon.voice_not_found",
                f"Voice {voice!r} is not available on PhononSynth model {target.id!r}.",
                "render.target.voice",
            )
        selected_options = {
            name: options[name] for name in PHONON_OPTION_NAMES if name in options
        }
        metadata = dict(target.metadata)
        metadata.update(
            {
                "languages": target.languages,
                "voices": target.voices,
                "default_voice": target.default_voice,
                "sample_rate": target.sample_rate,
                "qualities": target.qualities,
            }
        )
        selection = EngineSelection(
            engine=ENGINE_ID,
            target_id=target.id,
            language=language,
            voice=str(voice),
            options=selected_options,
            metadata=metadata,
            offline=offline,
            refresh=refresh,
        )
        return selection, ()

    def validate_selection(
        self, selection: EngineSelection
    ) -> tuple[EngineDiagnostic, ...]:
        try:
            targets = _target_catalog(
                language=None,
                offline=selection.offline,
                refresh=selection.refresh,
                options=selection.options,
            )
        except Exception as exc:  # noqa: BLE001 - Catalog failures are selection diagnostics.
            return (
                _selection_diagnostic(
                    "phonon.catalog_unavailable",
                    f"PhononSynth model catalog is unavailable: {exc}",
                    "synthesis.engine",
                ),
            )
        target = _find_target(targets, selection.target_id)
        if target is None:
            return (
                _selection_diagnostic(
                    "phonon.model_not_found",
                    f"PhononSynth model {selection.target_id!r} was not found.",
                    "render.target.id",
                ),
            )
        diagnostics: list[EngineDiagnostic] = []
        if selection.speaker is not None:
            diagnostics.append(
                _selection_diagnostic(
                    "phonon.speaker_unsupported",
                    "PhononSynth does not support speaker selection.",
                    "render.target.speaker",
                )
            )
        if target.languages and not any(
            language_tags_match(selection.language, available)
            for available in target.languages
        ):
            diagnostics.append(
                _selection_diagnostic(
                    "engine_language_incompatible",
                    f"PhononSynth model {target.id!r} supports {', '.join(target.languages)}, "
                    f"not {selection.language!r}.",
                    "render.target.language",
                )
            )
        quality = selection.options.get("quality")
        if quality is not None and quality not in target.qualities:
            diagnostics.append(
                _selection_diagnostic(
                    "phonon.quality_unavailable",
                    f"Quality {quality!r} is not available on PhononSynth model {target.id!r}.",
                    "render.options.quality",
                )
            )
        voice = selection.voice or target.default_voice
        if not target.voices or not voice:
            diagnostics.append(
                _selection_diagnostic(
                    "phonon.voice_required",
                    f"PhononSynth model {target.id!r} requires a published named voice.",
                    "render.target.voice",
                )
            )
        elif voice not in target.voices:
            diagnostics.append(
                _selection_diagnostic(
                    "phonon.voice_not_found",
                    f"Voice {voice!r} is not available on PhononSynth model {target.id!r}.",
                    "render.target.voice",
                )
            )
        if (
            isinstance(target.sample_rate, bool)
            or not isinstance(target.sample_rate, int)
            or target.sample_rate <= 0
        ):
            diagnostics.append(
                _selection_diagnostic(
                    "phonon.invalid_model_metadata",
                    f"PhononSynth model {target.id!r} has an invalid sample rate.",
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
                options=selection.options,
            )
        except Exception:  # noqa: BLE001 - Metadata is optional profile enrichment.
            return dict(selection.metadata)
        target = _find_target(targets, selection.target_id)
        if target is None:
            return dict(selection.metadata)
        return {
            **dict(target.metadata),
            "languages": target.languages,
            "voices": target.voices,
            "default_voice": target.default_voice,
            "sample_rate": target.sample_rate,
            "qualities": target.qualities,
        }

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> Mapping[str, Any]:
        module = _module()
        options = selection.options
        source_revisions = selection.metadata.get("source_revisions", {})
        release_tags = selection.metadata.get("release_tags", {})
        quality = options.get("quality")
        selected_revision = (
            source_revisions.get(quality)
            if isinstance(source_revisions, Mapping) and quality is not None
            else None
        ) or selection.metadata.get("source_revision")
        selected_release_tag = (
            release_tags.get(quality)
            if isinstance(release_tags, Mapping) and quality is not None
            else None
        )
        acoustic_options = {
            key: options[key]
            for key in ("providers", "provider_options", "session_options")
            if key in options
        }
        return {
            "schema": "voicerender.phonon-synthesis-identity.v1",
            "engine": ENGINE_ID,
            "engine_version": self.version(),
            "target_id": selection.target_id,
            "language": normalize_locale_tag(selection.language),
            "voice": selection.voice,
            "quality": quality,
            "source_revision": selected_revision,
            "release_version": selection.metadata.get("release_version"),
            "release_tag": selected_release_tag,
            "release_tags": dict(release_tags)
            if isinstance(release_tags, Mapping)
            else {},
            "source_revisions": (
                dict(source_revisions) if isinstance(source_revisions, Mapping) else {}
            ),
            "generation": {
                name: options.get(name) for name in _GENERATION_OPTION_NAMES
            },
            "providers": acoustic_options,
            "pcm_range_policy": PCM_RANGE_POLICY,
            "engine_identity": dict(module.runtime_identity()),
        }

    def open(
        self, selection: EngineSelection
    ) -> AbstractContextManager[PhononSynthEngineSession]:
        module = _module()
        options = dict(selection.options)
        try:
            config = module.SynthesisConfig(
                **{name: options.get(name) for name in _GENERATION_OPTION_NAMES}
            ).validated()
            voice = module.PhononVoice.from_pretrained(
                selection.target_id,
                quality=options.get("quality"),
                cache_dir=options.get("cache_dir"),
                offline=selection.offline,
                refresh_catalog=selection.refresh,
                force_download=bool(options.get("force_download", False)),
                catalog_url=options.get("catalog_url"),
                providers=options.get("providers"),
                provider_options=options.get("provider_options"),
                session_options=options.get("session_options"),
            )
        except EngineSynthesisError:
            raise
        except Exception as exc:  # Opening errors are classified centrally.
            raise _translate_phonon_error(exc, selection, opening=True) from exc

        @contextmanager
        def session() -> Any:
            try:
                yield PhononSynthEngineSession(voice, selection, config)
            finally:
                voice.close()

        return session()


__all__ = [
    "ENGINE_ID",
    "PACKAGE_NAME",
    "PHONONSYNTH_REQUEST_API_VERSION",
    "PHONON_OPTION_NAMES",
    "PhononSelectionError",
    "PhononSynthEngineAdapter",
    "PhononSynthEngineSession",
]
