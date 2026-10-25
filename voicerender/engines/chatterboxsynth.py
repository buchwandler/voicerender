"""Lazy ChatterboxSynth adapter for VoiceRender's request-centric engine API."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from typing import Any

from ..errors import (
    EmptySpeechTextError,
    EngineBackendError,
    EngineError,
    EngineSelectionError,
    EngineSynthesisError,
    EngineUnavailableError,
    InvalidEngineLanguageError,
    InvalidEngineModelError,
    InvalidEngineOptionError,
    InvalidEngineVoiceError,
    UnsupportedSynthesisFeatureError,
)
from ..voice_sources import (
    ResolvedReferenceAudio,
    normalize_voice_source,
    resolve_reference_audio,
    validate_voice_source,
    voice_source_identity,
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
)
from .catalog import CatalogRequest, SynthesisTarget
from .metadata import language_base, normalize_locale_tag
from .profiles import make_renderprofile_fragment
from .selection import EngineRequest

ENGINE_ID = "chatterbox"
PACKAGE_NAME = "chatterboxsynth"
SUPPORTED_LANGUAGES = frozenset({"en"})
SAMPLE_RATE = 24_000
CHATTERBOX_OPTION_NAMES = frozenset(
    {
        "voice_source",
        "voice_catalog_path",
        "voice_cache_dir",
        "quality",
        "provider",
        "provider_options",
        "session_options",
        "cache_dir",
        "max_new_tokens",
        "repetition_penalty",
    }
)

_REQUIRED_SYMBOLS = (
    "ChatterboxRuntime",
    "ChatterboxSynthError",
    "SynthesisRequest",
    "SynthesisResult",
    "GenerationConfig",
    "PreparedVoice",
    "RequestMeasure",
    "DiscoveredBundle",
    "discover_bundles",
    "runtime_identity",
    "REQUEST_API_VERSION",
    "request_api_contract",
)
_REQUIRED_METHODS = {
    "ChatterboxRuntime": (
        "from_installed",
        "prepare_voice",
        "measure_request",
        "synthesize",
        "close",
    ),
    "__module__": ("discover_bundles", "runtime_identity", "request_api_contract"),
}


def _module() -> Any:
    """Import ChatterboxSynth only when a Chatterbox operation is requested."""
    return importlib.import_module(PACKAGE_NAME)


def _value(owner: Any, name: str, default: Any = None) -> Any:
    if isinstance(owner, Mapping):
        return owner.get(name, default)
    return getattr(owner, name, default)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(
        dict.fromkeys(item for item in value if isinstance(item, str) and item)
    )


def _target_from_bundle(bundle: Any) -> SynthesisTarget:
    bundle_id = _value(bundle, "id")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise InvalidEngineModelError(
            "ChatterboxSynth discovery returned a bundle without an id.",
            engine=ENGINE_ID,
            code="chatterbox.catalog_invalid",
        )

    profiles = _string_tuple(_value(bundle, "profiles", ()))
    aliases = _string_tuple(_value(bundle, "aliases", ()))
    source_revision = _value(bundle, "source_revision")
    ref = _value(bundle, "ref")
    default_profile = _value(bundle, "default_profile")
    installed_profiles = _string_tuple(_value(bundle, "installed_profiles", ()))
    runtime_available = _value(bundle, "runtime_available", False) is True
    raw_sample_rate = _value(bundle, "sample_rate")
    sample_rate = (
        raw_sample_rate
        if isinstance(raw_sample_rate, int)
        and not isinstance(raw_sample_rate, bool)
        and raw_sample_rate > 0
        else SAMPLE_RATE
    )
    metadata = {
        "ref": ref,
        "source_revision": source_revision,
        "default_profile": default_profile,
        "profiles": profiles,
        "installed_profiles": installed_profiles,
        "sample_rate": sample_rate,
        "runtime_available": runtime_available,
    }
    return SynthesisTarget(
        engine=ENGINE_ID,
        id=bundle_id,
        display_name=str(_value(bundle, "display_name") or bundle_id),
        languages=("en",),
        status="ready",
        runtime_available=runtime_available,
        sample_rate=SAMPLE_RATE,
        qualities=profiles,
        voices=(),
        default_voice=None,
        aliases=aliases,
        capabilities=frozenset({"reference_voice", "request_measurement"}),
        metadata=metadata,
    )


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


def _engine_language(value: str | None) -> str:
    normalized = normalize_locale_tag(value or "en")
    language = language_base(normalized)
    if language not in SUPPORTED_LANGUAGES:
        raise InvalidEngineLanguageError(
            f"ChatterboxSynth supports English only, not {value!r}.",
            engine=ENGINE_ID,
            language=value,
            code="chatterbox.language_incompatible",
        )
    return "en"


def _error_context(
    selection: EngineSelection,
    error: Exception,
    request: SpeechRequest | None,
    *,
    stage: str,
) -> dict[str, Any]:
    module = _module()
    version = getattr(module, "__version__", None) or selection.metadata.get(
        "engine_version"
    )
    return {
        "engine": ENGINE_ID,
        "engine_version": version if isinstance(version, str) else None,
        "target_id": selection.target_id,
        "language": request.language if request is not None else selection.language,
        "request_id": request.id if request is not None else None,
        "native_error_type": type(error).__name__,
        "details": {"operation": stage, "native_error_type": type(error).__name__},
    }


def _translate_error(
    error: Exception,
    selection: EngineSelection,
    request: SpeechRequest | None = None,
    *,
    stage: str,
) -> EngineError:
    if isinstance(error, EngineError):
        return error
    module = _module()
    context = _error_context(selection, error, request, stage=stage)

    def is_native(name: str) -> bool:
        native_type = getattr(module, name, None)
        return isinstance(native_type, type) and isinstance(error, native_type)

    if is_native("OptionalDependencyError"):
        return EngineUnavailableError(
            f"ChatterboxSynth dependencies are unavailable: {error}",
            **context,
            code="chatterbox.optional_dependency_missing",
        )
    if is_native("ModelNotInstalledError"):
        return EngineUnavailableError(
            f"ChatterboxSynth target {selection.target_id!r} is not installed: {error}",
            **context,
            code="chatterbox.model_not_installed",
        )
    if is_native("ModelLoadError"):
        if stage == "open":
            return EngineUnavailableError(
                f"Could not open installed ChatterboxSynth target {selection.target_id!r}: {error}",
                **context,
                code="chatterbox.model_unavailable",
            )
        return EngineBackendError(
            f"ChatterboxSynth {stage} failed: {error}",
            **context,
            code=f"chatterbox.{stage}_failed",
        )
    if is_native("AudioPromptError") or is_native("IncompatibleVoiceError"):
        return InvalidEngineVoiceError(
            f"ChatterboxSynth rejected the selected reference voice: {error}",
            **context,
            code="chatterbox.reference_invalid",
        )
    if is_native("InvalidGenerationConfigError"):
        return InvalidEngineOptionError(
            f"ChatterboxSynth rejected generation options: {error}",
            **context,
            code="chatterbox.generation_config_invalid",
        )
    if is_native("EmptyTextError"):
        return EmptySpeechTextError(str(error), **context, code="chatterbox.empty_text")
    if is_native("RuntimeClosedError"):
        return EngineBackendError(
            f"ChatterboxSynth runtime is closed: {error}",
            **context,
            code="chatterbox.runtime_closed",
        )
    if is_native("SynthesisError"):
        return EngineSynthesisError(
            f"ChatterboxSynth synthesis failed: {error}",
            **context,
            code="chatterbox.synthesis_failed",
        )
    if stage == "open":
        return EngineUnavailableError(
            f"Could not open ChatterboxSynth target {selection.target_id!r}: {error}",
            **context,
            code="chatterbox.runtime_open_failed",
        )
    return EngineBackendError(
        f"ChatterboxSynth {stage} failed: {error}",
        **context,
        code=f"chatterbox.{stage}_failed",
    )


def _voice_source(selection: EngineSelection) -> Mapping[str, Any]:
    source = selection.options.get("voice_source")
    if not isinstance(source, Mapping):
        raise EngineSelectionError(
            "ChatterboxSynth requires one pinned local or managed reference voice.",
            engine=ENGINE_ID,
            target_id=selection.target_id,
            code="chatterbox.reference_required",
        )
    return source


def _native_request(request: SpeechRequest, module: Any) -> Any:
    return module.SynthesisRequest(
        id=request.id,
        text=request.text,
        language=_engine_language(request.language),
    )


def _reject_unsupported(request: SpeechRequest, selection: EngineSelection) -> None:
    if request.voice is not None:
        raise InvalidEngineVoiceError(
            "ChatterboxSynth does not support named voices.",
            engine=ENGINE_ID,
            target_id=selection.target_id,
            voice=request.voice,
            request_id=request.id,
            code="chatterbox.named_voice_unsupported",
        )
    if request.speaker is not None:
        raise UnsupportedSynthesisFeatureError(
            "ChatterboxSynth does not support speaker selection.",
            engine=ENGINE_ID,
            target_id=selection.target_id,
            speaker=request.speaker,
            request_id=request.id,
            code="chatterbox.speaker_unsupported",
        )
    if (
        request.tokens
        or request.pronunciation_overrides
        or request.whole_request_phonemes
    ):
        raise UnsupportedSynthesisFeatureError(
            "ChatterboxSynth does not support linguistic tokens, pronunciation overrides, or whole-request phonemes.",
            engine=ENGINE_ID,
            target_id=selection.target_id,
            request_id=request.id,
            code="chatterbox.feature_unsupported",
        )


class ChatterboxSynthEngineSession:
    """One installed Chatterbox runtime with lazily prepared reference state."""

    def __init__(self, runtime: Any, selection: EngineSelection, module: Any) -> None:
        self._runtime = runtime
        self._selection = selection
        self._module = module
        self._prepared_voices: dict[str, Any] = {}

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        _reject_unsupported(request, self._selection)
        try:
            measured = self._runtime.measure_request(
                _native_request(request, self._module)
            )
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_error(
                exc, self._selection, request, stage="measure"
            ) from exc

        amount = _value(measured, "amount")
        fits = _value(measured, "fits")
        maximum = _value(measured, "maximum")
        unit = _value(measured, "unit")
        valid_maximum = maximum is None or (
            isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0
        )
        if (
            isinstance(amount, bool)
            or not isinstance(amount, int)
            or amount < 0
            or not isinstance(fits, bool)
            or not valid_maximum
            or (unit is not None and unit != "model_tokens")
        ):
            raise EngineBackendError(
                "ChatterboxSynth returned an invalid request measurement.",
                engine=ENGINE_ID,
                engine_version=self._selection.metadata.get("engine_version"),
                target_id=self._selection.target_id,
                language=request.language,
                request_id=request.id,
                code="chatterbox.measurement_invalid",
                details={
                    "amount": amount,
                    "fits": fits,
                    "maximum": maximum,
                    "unit": unit,
                },
            )
        return RequestMeasure(
            fits=fits,
            amount=amount,
            maximum=maximum,
            unit="model_tokens",
            source="chatterboxsynth.measure_request",
            details={
                "language": _engine_language(request.language),
                "model": self._selection.target_id,
                "quality": self._selection.options.get("quality"),
            },
        )

    def _prepared_voice(self) -> Any:
        source_value = _voice_source(self._selection)
        try:
            source = normalize_voice_source(source_value, allow_named=False)
        except (EngineError, TypeError, ValueError) as exc:
            raise InvalidEngineVoiceError(
                str(exc),
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                native_error_type=type(exc).__name__,
                code="chatterbox.reference_invalid",
            ) from exc
        if source.kind not in {"reference", "managed_reference"}:
            raise InvalidEngineVoiceError(
                "ChatterboxSynth accepts local or managed reference audio only.",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                code="chatterbox.reference_required",
            )
        key = json.dumps(
            dict(voice_source_identity(source)), sort_keys=True, separators=(",", ":")
        )
        if key in self._prepared_voices:
            return self._prepared_voices[key]

        options = self._selection.options
        try:
            resolved: ResolvedReferenceAudio = resolve_reference_audio(
                source,
                cache_dir=options.get("voice_cache_dir"),
                offline=self._selection.offline,
                catalog_paths=options.get("voice_catalog_path"),
            )
            prepared = self._runtime.prepare_voice(resolved.path)
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_error(exc, self._selection, stage="prepare") from exc
        self._prepared_voices[key] = prepared
        return prepared

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        _reject_unsupported(request, self._selection)
        options = self._selection.options
        try:
            config = self._module.GenerationConfig(
                max_new_tokens=options.get("max_new_tokens"),
                repetition_penalty=options.get("repetition_penalty"),
            )
            result = self._runtime.synthesize(
                request.text,
                voice=self._prepared_voice(),
                config=config,
            )
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_error(
                exc, self._selection, request, stage="synthesis"
            ) from exc

        source_identity = dict(voice_source_identity(_voice_source(self._selection)))
        rendered = RenderedSpeech(
            id=request.id,
            audio=_value(result, "audio"),
            sample_rate=_value(result, "sample_rate"),
            metadata={
                "model": _value(result, "model"),
                "profile": _value(result, "profile"),
                "generated_tokens": _value(result, "generated_tokens"),
                "stopped_on_eos": _value(result, "stopped_on_eos"),
                "voice_source_identity": source_identity,
            },
        )
        return validate_rendered_speech(
            request,
            rendered,
            engine=ENGINE_ID,
            engine_version=self._selection.metadata.get("engine_version"),
            target_id=self._selection.target_id,
        )


class ChatterboxSynthEngineAdapter:
    """ChatterboxSynth discovery, profile selection, and request synthesis."""

    id = ENGINE_ID
    package_name = PACKAGE_NAME

    def version(self) -> str | None:
        return distribution_version(self.package_name)

    def probe_api(self) -> EngineApiProbe:
        probe = probe_public_api(
            engine=self.id,
            package=self.package_name,
            required_symbols=_REQUIRED_SYMBOLS,
            required_methods=_REQUIRED_METHODS,
            expected_api_version=SUPPORTED_REQUEST_API_VERSION,
            require_explicit_contract=True,
        )
        if not probe.compatible:
            return probe
        try:
            contract = _module().request_api_contract()
            required = {
                "supports_reference_voice": True,
                "supports_request_measurement": True,
                "supports_named_voices": False,
            }
            mismatches = {
                name: {"expected": expected, "actual": _value(contract, name)}
                for name, expected in required.items()
                if _value(contract, name) is not expected
            }
        except Exception as exc:  # noqa: BLE001 - Contract failures are probe results.
            return replace(
                probe,
                compatible=False,
                status="api_probe_failed",
                failed_stage="request_capabilities",
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

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            voice_binding_namespace=self.id,
            voice_binding_scope="target",
            option_names=CHATTERBOX_OPTION_NAMES,
            supports_named_voices=False,
            supports_reference_voice=True,
            supports_speakers=False,
            supports_pronunciation_overrides=False,
            supports_linguistic_tokens=False,
            supports_whole_request_phonemes=False,
            supports_request_measurement=True,
        )

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            frontend="chatterboxsynth",
            pronunciation="unsupported",
        )

    def discover(self, request: CatalogRequest) -> tuple[SynthesisTarget, ...]:
        language = _engine_language(request.language) if request.language else None
        module = _module()
        bundles = module.discover_bundles(
            offline=request.offline,
            refresh=request.refresh,
        )
        targets = tuple(_target_from_bundle(bundle) for bundle in bundles)
        if language is None:
            return targets
        return tuple(target for target in targets if language in target.languages)

    def resolve(
        self, request: EngineRequest
    ) -> tuple[EngineSelection, tuple[Any, ...]]:
        target_id = request.target_id
        if not isinstance(target_id, str) or not target_id.strip():
            raise EngineSelectionError(
                "A ChatterboxSynth target id is required.",
                engine=self.id,
                code="chatterbox.target_not_found",
            )

        options = dict(request.options)
        engine_options = dict(request.engine_options)
        unknown = sorted(set(engine_options) - CHATTERBOX_OPTION_NAMES)
        if unknown:
            raise EngineSelectionError(
                f"Unsupported ChatterboxSynth option(s): {', '.join(unknown)}.",
                engine=self.id,
                target_id=target_id,
                code="chatterbox.invalid_option",
                details={"options": unknown},
            )
        options.update(engine_options)

        if request.voice is not None:
            raise EngineSelectionError(
                "ChatterboxSynth does not support named voices; provide voice_source.",
                engine=self.id,
                target_id=target_id,
                code="chatterbox.named_voice_unsupported",
            )
        if request.speaker is not None:
            raise EngineSelectionError(
                "ChatterboxSynth does not support speaker selection.",
                engine=self.id,
                target_id=target_id,
                code="chatterbox.speaker_unsupported",
            )

        source_value = options.get("voice_source")
        if not isinstance(source_value, Mapping):
            raise EngineSelectionError(
                "ChatterboxSynth requires a pinned local or managed reference voice.",
                engine=self.id,
                target_id=target_id,
                code="chatterbox.reference_required",
            )
        try:
            source = normalize_voice_source(source_value, allow_named=False)
        except (EngineError, TypeError, ValueError) as exc:
            raise EngineSelectionError(
                str(exc),
                engine=self.id,
                target_id=target_id,
                code="chatterbox.reference_required",
                details={"native_error_type": type(exc).__name__},
            ) from exc
        if source.kind not in {"reference", "managed_reference"}:
            raise EngineSelectionError(
                "ChatterboxSynth accepts local or managed reference audio only.",
                engine=self.id,
                target_id=target_id,
                code="chatterbox.reference_required",
            )

        language = normalize_locale_tag(request.language or "en-US")
        try:
            native_language = _engine_language(language)
        except InvalidEngineLanguageError as exc:
            raise EngineSelectionError(
                str(exc),
                engine=self.id,
                target_id=target_id,
                language=language,
                code="chatterbox.language_incompatible",
            ) from exc

        try:
            targets = self.discover(
                CatalogRequest(
                    engine=self.id,
                    offline=request.offline,
                    refresh=request.refresh,
                )
            )
        except Exception as exc:
            if isinstance(exc, EngineSelectionError):
                raise
            raise EngineSelectionError(
                f"ChatterboxSynth target catalog is unavailable: {exc}",
                engine=self.id,
                target_id=target_id,
                code="chatterbox.target_catalog_unavailable",
                details={"native_error_type": type(exc).__name__},
            ) from exc
        target = _find_target(targets, target_id)
        if target is None:
            raise EngineSelectionError(
                f"ChatterboxSynth target {target_id!r} was not found.",
                engine=self.id,
                target_id=target_id,
                code="chatterbox.target_not_found",
            )

        quality = options.get("quality", target.metadata.get("default_profile"))
        if not isinstance(quality, str) or quality not in target.qualities:
            raise EngineSelectionError(
                f"ChatterboxSynth target {target.id!r} does not support quality {quality!r}.",
                engine=self.id,
                target_id=target.id,
                code="chatterbox.quality_incompatible",
                details={"available": target.qualities},
            )
        provider = options.get("provider", "cpu")
        if not isinstance(provider, str) or not provider.strip():
            raise EngineSelectionError(
                "ChatterboxSynth provider must be a non-empty string.",
                engine=self.id,
                target_id=target.id,
                code="chatterbox.invalid_option",
            )

        module = _module()
        try:
            generation = module.GenerationConfig(
                max_new_tokens=options.get("max_new_tokens"),
                repetition_penalty=options.get("repetition_penalty"),
            )
        except Exception as exc:
            synth_error = getattr(module, "ChatterboxSynthError", ())
            if isinstance(synth_error, type) and isinstance(exc, synth_error):
                raise EngineSelectionError(
                    f"Invalid ChatterboxSynth generation options: {exc}",
                    engine=self.id,
                    target_id=target.id,
                    code="chatterbox.generation_config_invalid",
                    details={"native_error_type": type(exc).__name__},
                ) from exc
            if isinstance(exc, (TypeError, ValueError)):
                raise EngineSelectionError(
                    f"Invalid ChatterboxSynth generation options: {exc}",
                    engine=self.id,
                    target_id=target.id,
                    code="chatterbox.generation_config_invalid",
                    details={"native_error_type": type(exc).__name__},
                ) from exc
            raise

        selected_options: dict[str, Any] = {
            "voice_source": source.as_mapping(),
            "quality": quality,
            "provider": provider.strip(),
        }
        if generation.max_new_tokens is not None:
            selected_options["max_new_tokens"] = generation.max_new_tokens
        if generation.repetition_penalty is not None:
            selected_options["repetition_penalty"] = generation.repetition_penalty
        for name in (
            "voice_catalog_path",
            "voice_cache_dir",
            "provider_options",
            "session_options",
            "cache_dir",
        ):
            if options.get(name) is not None:
                selected_options[name] = options[name]

        metadata = {
            "ref": target.metadata.get("ref"),
            "source_revision": target.metadata.get("source_revision"),
            "default_profile": target.metadata.get("default_profile"),
            "profiles": target.metadata.get("profiles", ()),
            "installed_profiles": target.metadata.get("installed_profiles", ()),
            "sample_rate": target.metadata.get("sample_rate", SAMPLE_RATE),
            "runtime_available": target.runtime_available,
            "engine_version": getattr(module, "__version__", self.version()),
            "native_language": native_language,
        }
        return (
            EngineSelection(
                engine=self.id,
                target_id=target.id,
                language=language,
                options=selected_options,
                metadata=metadata,
                offline=request.offline,
                refresh=request.refresh,
            ),
            (),
        )

    def validate_selection(
        self, selection: EngineSelection
    ) -> tuple[EngineDiagnostic, ...]:
        diagnostics: list[EngineDiagnostic] = []
        try:
            _engine_language(selection.language)
        except EngineError as exc:
            diagnostics.append(
                EngineDiagnostic(
                    code=getattr(exc, "code", "chatterbox.language_incompatible"),
                    severity="error",
                    message=str(exc),
                    field="render.target.language",
                )
            )
        source = selection.options.get("voice_source")
        if not isinstance(source, Mapping):
            diagnostics.append(
                EngineDiagnostic(
                    code="chatterbox.reference_required",
                    severity="error",
                    message="ChatterboxSynth requires pinned local or managed reference audio.",
                    field="render.target.voice",
                )
            )
        else:
            try:
                normalized = normalize_voice_source(source, allow_named=False)
            except (EngineError, TypeError, ValueError) as exc:
                diagnostics.append(
                    EngineDiagnostic(
                        code="chatterbox.reference_required",
                        severity="error",
                        message=str(exc),
                        field="render.target.voice",
                    )
                )
            else:
                if normalized.kind not in {"reference", "managed_reference"}:
                    diagnostics.append(
                        EngineDiagnostic(
                            code="chatterbox.reference_required",
                            severity="error",
                            message="ChatterboxSynth accepts reference audio only.",
                            field="render.target.voice",
                        )
                    )
                else:
                    diagnostics.extend(
                        validate_voice_source(
                            normalized,
                            offline=selection.offline,
                            catalog_paths=selection.options.get("voice_catalog_path"),
                        )
                    )
        return tuple(diagnostics)

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> Mapping[str, Any]:
        module = _module()
        options = selection.options
        source = _voice_source(selection)
        runtime_identity = module.runtime_identity()
        identity: dict[str, Any] = {
            "engine": self.id,
            "engine_version": getattr(module, "__version__", self.version()),
            "runtime_identity": (
                dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {}
            ),
            "target_id": selection.target_id,
            "model_revision": selection.metadata.get("source_revision"),
            "quality": options.get("quality"),
            "provider": options.get("provider"),
            "language": _engine_language(selection.language),
            "voice": dict(voice_source_identity(source)),
            "generation": {
                "max_new_tokens": options.get("max_new_tokens"),
                "repetition_penalty": options.get("repetition_penalty"),
            },
        }
        for name in ("provider_options", "session_options"):
            if options.get(name) is not None:
                identity[name] = options[name]
        return identity

    def target_metadata(self, selection: EngineSelection) -> Mapping[str, Any]:
        return dict(selection.metadata)

    def open(
        self, selection: EngineSelection
    ) -> AbstractContextManager[ChatterboxSynthEngineSession]:
        module = _module()
        options = selection.options
        bundle_ref = selection.metadata.get("ref") or selection.target_id
        try:
            runtime = module.ChatterboxRuntime.from_installed(
                bundle_ref,
                quality=options.get("quality"),
                provider=options.get("provider", "cpu"),
                provider_options=options.get("provider_options"),
                session_options=options.get("session_options"),
                cache_dir=options.get("cache_dir"),
            )
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_error(exc, selection, stage="open") from exc

        @contextmanager
        def session() -> Iterator[ChatterboxSynthEngineSession]:
            try:
                yield ChatterboxSynthEngineSession(runtime, selection, module)
            finally:
                try:
                    runtime.close()
                except EngineError:
                    raise
                except Exception as exc:
                    raise _translate_error(exc, selection, stage="close") from exc

        return session()


__all__ = [
    "CHATTERBOX_OPTION_NAMES",
    "ENGINE_ID",
    "PACKAGE_NAME",
    "SUPPORTED_LANGUAGES",
    "ChatterboxSynthEngineAdapter",
    "ChatterboxSynthEngineSession",
]
