"""Request-centric adapter for the public SoproSynth runtime API."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from numbers import Real
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

ENGINE_ID = "sopro"
PACKAGE_NAME = "soprosynth"
SUPPORTED_LANGUAGES = frozenset({"en", "pt", "fr", "de"})
SOPRO_OPTION_NAMES = frozenset(
    {
        "voice_source",
        "voice_catalog_path",
        "voice_cache_dir",
        "profile",
        "providers",
        "provider_options",
        "session_options",
        "reference_seconds",
        "reference_resampler",
        "temperature",
        "top_p",
        "top_k",
        "steps",
        "max_seconds",
        "min_seconds",
        "random_seed",
        "text_chunk_size",
        "output_mode",
        "voice_level",
        "voice_level_gain_db",
    }
)


def _module() -> Any:
    """Import SoproSynth only when a Sopro operation is requested."""
    import soprosynth

    return soprosynth


def _bundle_value(bundle: Any, name: str, default: Any = None) -> Any:
    if isinstance(bundle, Mapping):
        return bundle.get(name, default)
    return getattr(bundle, name, default)


def _bundle_languages(bundle: Any) -> tuple[str, ...]:
    raw_languages = _bundle_value(bundle, "languages", ()) or ()
    if isinstance(raw_languages, str):
        raw_languages = (raw_languages,)
    return tuple(
        dict.fromkeys(
            language
            for raw in raw_languages
            if isinstance(raw, str)
            and (language := language_base(normalize_locale_tag(raw)))
            in SUPPORTED_LANGUAGES
        )
    )


def _target_from_bundle(bundle: Any) -> SynthesisTarget:
    bundle_id = _bundle_value(bundle, "id")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise InvalidEngineModelError(
            "SoproSynth discovery returned a bundle without an id.",
            engine=ENGINE_ID,
            code="sopro.catalog_invalid",
        )
    raw_metadata = _bundle_value(bundle, "metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    source_revision = _bundle_value(bundle, "source_revision")
    source_commit = _bundle_value(bundle, "source_commit")
    bundle_ref = _bundle_value(bundle, "ref")
    profiles = tuple(
        value
        for value in (_bundle_value(bundle, "profiles", ()) or ())
        if isinstance(value, str) and value
    )
    languages = _bundle_languages(bundle)
    sample_rate = _bundle_value(bundle, "sample_rate")
    if type(sample_rate) is not int or sample_rate <= 0:
        sample_rate = None
    metadata.update(
        {
            "bundle_ref": bundle_ref,
            "source_revision": source_revision,
            "source_commit": source_commit,
            "profiles": profiles,
            "languages": languages,
            "sample_rate": sample_rate,
            "max_input_tokens": metadata.get("max_input_tokens"),
        }
    )
    return SynthesisTarget(
        engine=ENGINE_ID,
        id=bundle_id,
        display_name=str(_bundle_value(bundle, "display_name") or bundle_id),
        languages=languages,
        status="ready",
        sample_rate=sample_rate,
        voices=(),
        default_voice=None,
        aliases=tuple(
            value
            for value in (_bundle_value(bundle, "aliases", ()) or ())
            if isinstance(value, str) and value
        ),
        capabilities=frozenset(
            {"reference_voice", "request_measurement", "voice_level"}
        ),
        metadata=metadata,
    )


def _target_catalog(
    *, offline: bool, refresh: bool, cache_dir: Any = None
) -> tuple[SynthesisTarget, ...]:
    module = _module()
    bundles = module.discover_bundles(
        cache_dir=cache_dir,
        offline=offline,
        refresh=refresh,
    )
    return tuple(_target_from_bundle(bundle) for bundle in bundles)


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


def _engine_language(value: str | None) -> str | None:
    if value is None or value.casefold() == "auto":
        return None
    normalized = normalize_locale_tag(value)
    language = language_base(normalized)
    if language not in SUPPORTED_LANGUAGES:
        raise InvalidEngineLanguageError(
            f"SoproSynth does not support language {value!r}.",
            engine=ENGINE_ID,
            language=value,
            code="sopro.language_incompatible",
        )
    return language


def _native_request(request: SpeechRequest, module: Any) -> Any:
    return module.SynthesisRequest(
        id=request.id,
        text=request.text,
        language=_engine_language(request.language),
    )


def _error_context(
    selection: EngineSelection, request: SpeechRequest | None, error: Exception
) -> dict[str, Any]:
    return {
        "engine": ENGINE_ID,
        "engine_version": distribution_version(PACKAGE_NAME),
        "target_id": selection.target_id,
        "language": request.language if request is not None else selection.language,
        "request_id": request.id if request is not None else None,
        "native_error_type": type(error).__name__,
    }


def _translate_error(
    error: Exception,
    selection: EngineSelection,
    request: SpeechRequest | None = None,
    *,
    stage: str = "synthesis",
) -> EngineError:
    module = _module()
    name = type(error).__name__
    context = _error_context(selection, request, error)
    named_mappings: dict[str, type[EngineError]] = {
        "EmptyTextError": EmptySpeechTextError,
        "InvalidLanguageError": InvalidEngineLanguageError,
        "InvalidVoiceError": InvalidEngineVoiceError,
        "InvalidGenerationConfigError": InvalidEngineOptionError,
        "InvalidRequestError": InvalidSpeechRequestError,
        "UnsupportedFeatureError": UnsupportedSynthesisFeatureError,
    }
    mapped_type = named_mappings.get(name)
    native_type = getattr(module, name, None)
    if (
        mapped_type is not None
        and isinstance(native_type, type)
        and isinstance(error, native_type)
    ):
        return mapped_type(str(error), **context)

    if stage == "open":
        if name in {
            "BundleNotFoundError",
            "OfflineAssetError",
            "AssetAccessError",
            "AssetCacheError",
            "UnsupportedBundleError",
        }:
            return EngineBackendError(
                f"SoproSynth target {selection.target_id!r} is not installed or cannot be opened: {error}",
                **context,
                code="sopro.profile_unavailable",
            )
        code = "sopro.session_open_failed"
        message = f"Could not open SoproSynth target {selection.target_id!r}: {error}"
    elif stage == "prepare":
        code = "sopro.reference_prepare_failed"
        message = f"SoproSynth could not prepare the selected reference voice: {error}"
    elif stage == "measure":
        code = "sopro.measurement_failed"
        message = f"SoproSynth request measurement failed: {error}"
    elif stage == "close":
        code = "sopro.session_close_failed"
        message = f"SoproSynth runtime close failed: {error}"
    else:
        code = "sopro.synthesis_failed"
        message = f"SoproSynth synthesis failed: {error}"
    return EngineBackendError(message, **context, code=code)


def _positive_maximum(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


class SoproSynthEngineSession:
    """One loaded Sopro runtime with lazily prepared reference voices."""

    def __init__(self, runtime: Any, selection: EngineSelection) -> None:
        self._runtime = runtime
        self._selection = selection
        self._prepared_voices: dict[str, Any] = {}

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        if request.voice is not None:
            raise InvalidEngineVoiceError(
                "SoproSynth does not support named voices.",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                voice=request.voice,
                request_id=request.id,
                code="sopro.named_voice_unsupported",
            )
        if request.speaker is not None:
            raise UnsupportedSynthesisFeatureError(
                "SoproSynth does not support speaker selection.",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                speaker=request.speaker,
                request_id=request.id,
            )
        if (
            request.tokens
            or request.pronunciation_overrides
            or request.whole_request_phonemes
        ):
            raise UnsupportedSynthesisFeatureError(
                "SoproSynth does not support linguistic tokens, pronunciation overrides, or whole-request phonemes.",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        module = _module()
        try:
            measured = self._runtime.measure_request(
                _native_request(request, module),
                maximum=self._selection.metadata.get("max_input_tokens"),
            )
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_error(
                exc, self._selection, request, stage="measure"
            ) from exc
        amount = getattr(measured, "amount", None)
        maximum = _positive_maximum(getattr(measured, "maximum", None))
        if type(amount) is not int or amount < 0:
            raise EngineBackendError(
                "SoproSynth returned an invalid request measurement.",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
                code="sopro.measurement_invalid",
                details={"amount": amount, "maximum": maximum},
            )
        fits = getattr(measured, "fits", None)
        if not isinstance(fits, bool):
            fits = None if maximum is None else amount <= maximum
        return RequestMeasure(
            fits=fits,
            amount=amount,
            maximum=maximum,
            unit="model_tokens",
            source="soprosynth.measure_request",
            details={"language": _engine_language(request.language)},
        )

    def _voice_source(self) -> Mapping[str, Any]:
        source = self._selection.options.get("voice_source")
        if not isinstance(source, Mapping):
            raise EngineSelectionError(
                "SoproSynth requires one pinned local or managed reference voice.",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                code="sopro.reference_required",
            )
        return source

    def _cache_key(self, source: Mapping[str, Any]) -> str:
        identity = {
            "source": dict(voice_source_identity(source)),
            "reference_seconds": self._selection.options.get("reference_seconds"),
            "reference_resampler": self._selection.options.get("reference_resampler"),
            "target_id": self._selection.target_id,
            "source_revision": self._selection.metadata.get("source_revision"),
        }
        return json.dumps(identity, sort_keys=True, separators=(",", ":"))

    def _prepared_voice(self) -> Any:
        source = self._voice_source()
        key = self._cache_key(source)
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
            prepared = self._runtime.prepare_voice(
                resolved.path,
                reference_seconds=options.get("reference_seconds"),
                resampler=options.get("reference_resampler"),
            )
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_error(exc, self._selection, stage="prepare") from exc
        self._prepared_voices[key] = prepared
        return prepared

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        if request.voice is not None:
            raise InvalidEngineVoiceError(
                "SoproSynth does not support named voices.",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                voice=request.voice,
                request_id=request.id,
                code="sopro.named_voice_unsupported",
            )
        if request.speaker is not None:
            raise UnsupportedSynthesisFeatureError(
                "SoproSynth does not support speaker selection.",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                speaker=request.speaker,
                request_id=request.id,
            )
        if (
            request.tokens
            or request.pronunciation_overrides
            or request.whole_request_phonemes
        ):
            raise UnsupportedSynthesisFeatureError(
                "SoproSynth does not support linguistic tokens, pronunciation overrides, or whole-request phonemes.",
                engine=ENGINE_ID,
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        module = _module()
        options = self._selection.options
        try:
            result = self._runtime.synthesize(
                _native_request(request, module),
                voice=self._prepared_voice(),
                config=module.GenerationConfig(
                    temperature=options.get("temperature"),
                    top_p=options.get("top_p"),
                    top_k=options.get("top_k"),
                    steps=options.get("steps"),
                    max_seconds=options.get("max_seconds"),
                    min_seconds=options.get("min_seconds"),
                    seed=options.get("random_seed"),
                    text_chunk_size=options.get("text_chunk_size"),
                ),
                voice_level=module.VoiceLevelConfig(
                    mode=options.get("voice_level", "off"),
                    gain_db=options.get("voice_level_gain_db"),
                ),
                output=module.SoproOutputConfig(
                    mode=options.get("output_mode", "native")
                ),
            )
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_error(exc, self._selection, request) from exc

        result_metadata = dict(getattr(result, "metadata", {}) or {})
        source_identity = dict(voice_source_identity(self._voice_source()))
        prepared_voice = self._prepared_voices[self._cache_key(self._voice_source())]
        voice_identity = getattr(prepared_voice, "identity", {})
        metadata = {
            **result_metadata,
            "model_id": result_metadata.get("model_id", self._selection.target_id),
            "model_revision": self._selection.metadata.get("source_revision"),
            "voice_identity": dict(voice_identity)
            if isinstance(voice_identity, Mapping)
            else {},
            "voice_source_identity": source_identity,
            "effective_generation": result_metadata.get("effective_generation", {}),
            "sopro_output": result_metadata.get("sopro_output", {}),
            "voice_level_application": result_metadata.get(
                "voice_level_application", {}
            ),
            "runtime_identity": result_metadata.get("runtime_identity", {}),
        }
        rendered = RenderedSpeech(
            id=request.id,
            audio=result.audio,
            sample_rate=result.sample_rate,
            warnings=tuple(getattr(result, "warnings", ()) or ()),
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


class SoproSynthEngineAdapter:
    """SoproSynth discovery, reference selection, and lazy request synthesis."""

    id = ENGINE_ID
    package_name = PACKAGE_NAME

    def version(self) -> str | None:
        return distribution_version(self.package_name)

    def probe_api(self) -> EngineApiProbe:
        probe = probe_public_api(
            engine=self.id,
            package=self.package_name,
            required_symbols=(
                "SoproRuntime",
                "SoproSynthError",
                "SynthesisRequest",
                "SynthesisResult",
                "GenerationConfig",
                "VoiceLevelConfig",
                "SoproOutputConfig",
                "PreparedVoice",
                "RequestMeasure",
                "DiscoveredBundle",
                "discover_bundles",
                "runtime_identity",
                "REQUEST_API_VERSION",
                "request_api_contract",
            ),
            required_methods={
                "SoproRuntime": (
                    "load",
                    "prepare_voice",
                    "measure_request",
                    "synthesize",
                    "close",
                ),
                "__module__": (
                    "discover_bundles",
                    "runtime_identity",
                    "request_api_contract",
                ),
            },
            expected_api_version=SUPPORTED_REQUEST_API_VERSION,
            require_explicit_contract=True,
        )
        if not probe.compatible:
            return probe
        module = _module()
        try:
            contract = module.request_api_contract()
            required = {
                "supports_reference_voice": True,
                "supports_request_measurement": True,
                "supports_named_voices": False,
            }
            mismatches = {
                name: {"expected": value, "actual": getattr(contract, name, None)}
                for name, value in required.items()
                if getattr(contract, name, None) is not value
            }
        except Exception as exc:  # noqa: BLE001 - Public contract failures are probe results.
            return replace(
                probe,
                compatible=False,
                status="api_probe_failed",
                failed_stage="contract_capabilities",
                failed_symbol="request_api_contract",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        if mismatches:
            return replace(
                probe,
                compatible=False,
                status="api_incompatible",
                failed_stage="contract_capabilities",
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
            option_names=SOPRO_OPTION_NAMES,
            supports_named_voices=False,
            supports_reference_voice=True,
            supports_speakers=False,
            supports_pronunciation_overrides=False,
            supports_linguistic_tokens=False,
            supports_whole_request_phonemes=False,
            supports_voice_level_calibration=True,
            supports_request_measurement=True,
        )

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            frontend="soprosynth",
            pronunciation="unsupported",
            capacity_key="max_input_tokens",
            capacity_metric="model_tokens",
        )

    def discover(self, request: CatalogRequest) -> tuple[SynthesisTarget, ...]:
        cache_dir = None
        # Catalog path is an engine-owned discovery cache hint, not voice identity.
        try:
            options = getattr(request, "options", {})
            if isinstance(options, Mapping):
                cache_dir = options.get("voice_catalog_path")
        except Exception:  # noqa: BLE001 - CatalogRequest currently has no engine options.
            cache_dir = None
        targets = _target_catalog(
            offline=request.offline,
            refresh=request.refresh,
            cache_dir=cache_dir,
        )
        if request.language is None:
            return targets
        requested = _engine_language(request.language)
        return tuple(
            target
            for target in targets
            if not target.languages or requested in target.languages
        )

    def resolve(self, request: Any) -> tuple[EngineSelection, tuple[Any, ...]]:
        module = _module()
        options = dict(getattr(request, "options", {}) or {})
        engine_options = dict(getattr(request, "engine_options", {}) or {})
        unknown = sorted(set(engine_options) - SOPRO_OPTION_NAMES)
        if unknown:
            raise EngineSelectionError(
                f"Unsupported SoproSynth option(s): {', '.join(unknown)}.",
                engine=self.id,
                target_id=getattr(request, "target_id", None),
                code="sopro.invalid_option",
                details={"options": unknown},
            )
        options.update(engine_options)
        if getattr(request, "voice", None) is not None:
            raise EngineSelectionError(
                "SoproSynth does not support named voices; provide voice_source.",
                engine=self.id,
                target_id=getattr(request, "target_id", None),
                code="sopro.named_voice_unsupported",
            )
        if getattr(request, "speaker", None) is not None:
            raise EngineSelectionError(
                "SoproSynth does not support speaker selection.",
                engine=self.id,
                target_id=getattr(request, "target_id", None),
                code="sopro.named_voice_unsupported",
            )
        source_value = options.get("voice_source")
        if source_value is None:
            raise EngineSelectionError(
                "SoproSynth requires one pinned local or managed reference voice.",
                engine=self.id,
                target_id=getattr(request, "target_id", None),
                code="sopro.reference_required",
            )
        try:
            source = normalize_voice_source(source_value, allow_named=False)
        except EngineError as exc:
            raise EngineSelectionError(
                str(exc),
                engine=self.id,
                target_id=getattr(request, "target_id", None),
                code=getattr(exc, "code", "sopro.reference_invalid"),
                details=getattr(exc, "details", {}),
            ) from exc
        if source.kind not in {"reference", "managed_reference"}:
            raise EngineSelectionError(
                "SoproSynth requires a local or managed reference voice.",
                engine=self.id,
                target_id=getattr(request, "target_id", None),
                code="sopro.reference_required",
            )
        options["voice_source"] = source.as_mapping()

        target_id_value = getattr(request, "target_id", None)
        if not isinstance(target_id_value, str) or not target_id_value.strip():
            raise EngineSelectionError(
                "A SoproSynth target id is required.",
                engine=self.id,
                code="sopro.target_not_found",
            )
        offline = bool(getattr(request, "offline", False))
        refresh = bool(getattr(request, "refresh", False))
        try:
            targets = _target_catalog(
                offline=offline,
                refresh=refresh,
                cache_dir=options.get("voice_catalog_path"),
            )
        except Exception as exc:
            raise EngineSelectionError(
                f"SoproSynth target catalog is unavailable: {exc}",
                engine=self.id,
                target_id=target_id_value,
                code="sopro.target_catalog_unavailable",
                details={"native_error_type": type(exc).__name__},
            ) from exc
        target = _find_target(targets, target_id_value)
        if target is None:
            raise EngineSelectionError(
                f"SoproSynth target {target_id_value!r} was not found.",
                engine=self.id,
                target_id=target_id_value,
                code="sopro.target_not_found",
            )
        try:
            language = normalize_locale_tag(
                getattr(request, "language", None) or "en-US"
            )
            native_language = _engine_language(language)
            if target.languages and native_language not in target.languages:
                raise InvalidEngineLanguageError(
                    f"SoproSynth target {target.id!r} does not support {language!r}.",
                    engine=self.id,
                    target_id=target.id,
                    language=language,
                    code="sopro.language_incompatible",
                )
            generation = module.GenerationConfig(
                temperature=options.get("temperature"),
                top_p=options.get("top_p"),
                top_k=options.get("top_k"),
                steps=options.get("steps"),
                max_seconds=options.get("max_seconds"),
                min_seconds=options.get("min_seconds"),
                seed=options.get("random_seed"),
                text_chunk_size=options.get("text_chunk_size"),
            )
            output = module.SoproOutputConfig(mode=options.get("output_mode", "native"))
            voice_level = module.VoiceLevelConfig(
                mode=options.get("voice_level", "off"),
                gain_db=options.get("voice_level_gain_db"),
            )
            reference_seconds = options.get("reference_seconds")
            if reference_seconds is not None:
                if isinstance(reference_seconds, bool) or not isinstance(
                    reference_seconds, Real
                ):
                    raise ValueError(
                        "reference_seconds must be a finite positive number"
                    )
                try:
                    reference_seconds = float(reference_seconds)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "reference_seconds must be a finite positive number"
                    ) from exc
                if not math.isfinite(reference_seconds) or reference_seconds <= 0:
                    raise ValueError(
                        "reference_seconds must be a finite positive number"
                    )
            if reference_seconds is not None:
                options["reference_seconds"] = reference_seconds
            reference_resampler = options.get("reference_resampler")
            if reference_resampler not in (None, "linear", "audiosig"):
                raise ValueError(
                    "reference_resampler must be 'linear', 'audiosig', or None"
                )
        except EngineError:
            raise
        except module.SoproSynthError as exc:
            raise EngineSelectionError(
                f"Invalid SoproSynth selection options: {exc}",
                engine=self.id,
                target_id=target.id,
                code="sopro.invalid_option",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise EngineSelectionError(
                f"Invalid SoproSynth selection options: {exc}",
                engine=self.id,
                target_id=target.id,
                code="sopro.invalid_option",
            ) from exc

        selected_options: dict[str, Any] = {
            "voice_source": source.as_mapping(),
            "profile": options.get("profile", "wasm-uint8"),
            "output_mode": output.mode,
            "voice_level": voice_level.mode,
        }
        for name in (
            "voice_catalog_path",
            "voice_cache_dir",
            "providers",
            "provider_options",
            "session_options",
            "reference_seconds",
            "reference_resampler",
        ):
            if options.get(name) is not None:
                selected_options[name] = options[name]
        selected_options.update(
            {
                name: getattr(generation, attr)
                for name, attr in (
                    ("temperature", "temperature"),
                    ("top_p", "top_p"),
                    ("top_k", "top_k"),
                    ("steps", "steps"),
                    ("max_seconds", "max_seconds"),
                    ("min_seconds", "min_seconds"),
                    ("random_seed", "seed"),
                    ("text_chunk_size", "text_chunk_size"),
                )
                if getattr(generation, attr) is not None
            }
        )
        if voice_level.gain_db is not None:
            selected_options["voice_level_gain_db"] = voice_level.gain_db
        metadata = {
            "bundle_ref": target.metadata.get("bundle_ref"),
            "source_revision": target.metadata.get("source_revision"),
            "source_commit": target.metadata.get("source_commit"),
            "profiles": target.metadata.get("profiles", ()),
            "languages": target.languages,
            "sample_rate": target.sample_rate,
            "max_input_tokens": _positive_maximum(
                target.metadata.get("max_input_tokens")
            ),
        }
        selection = EngineSelection(
            engine=self.id,
            target_id=target.id,
            language=language,
            voice=None,
            speaker=None,
            options=selected_options,
            metadata=metadata,
            offline=offline,
            refresh=refresh,
        )
        return selection, ()

    def validate_selection(
        self, selection: EngineSelection
    ) -> tuple[EngineDiagnostic, ...]:
        diagnostics: list[EngineDiagnostic] = []
        if selection.voice is not None or selection.speaker is not None:
            diagnostics.append(
                EngineDiagnostic(
                    code="sopro.named_voice_unsupported",
                    severity="error",
                    message="SoproSynth accepts reference sources, not named voices or speakers.",
                    field="render.target.voice",
                )
            )
        source = selection.options.get("voice_source")
        if source is None:
            diagnostics.append(
                EngineDiagnostic(
                    code="sopro.reference_required",
                    severity="error",
                    message="SoproSynth requires one pinned local or managed reference voice.",
                    field="render.target.voice",
                )
            )
        else:
            diagnostics.extend(
                validate_voice_source(
                    source,
                    offline=selection.offline,
                    catalog_paths=selection.options.get("voice_catalog_path"),
                )
            )
        try:
            language = _engine_language(selection.language)
        except EngineError as exc:
            diagnostics.append(
                EngineDiagnostic(
                    code=getattr(exc, "code", "sopro.language_incompatible"),
                    severity="error",
                    message=str(exc),
                    field="render.target.language",
                )
            )
        else:
            languages = selection.metadata.get("languages", ())
            if languages and language not in languages:
                diagnostics.append(
                    EngineDiagnostic(
                        code="sopro.language_incompatible",
                        severity="error",
                        message=f"SoproSynth target {selection.target_id!r} does not support {selection.language!r}.",
                        field="render.target.language",
                    )
                )
        return tuple(diagnostics)

    def target_metadata(self, selection: EngineSelection) -> Mapping[str, Any]:
        return dict(selection.metadata)

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> Mapping[str, Any]:
        module = _module()
        options = selection.options
        source = options.get("voice_source")
        if not isinstance(source, Mapping):
            raise EngineSelectionError(
                "SoproSynth requires one pinned reference voice.",
                engine=self.id,
                target_id=selection.target_id,
                code="sopro.reference_required",
            )
        return {
            "engine": self.id,
            "engine_version": self.version(),
            "runtime_identity": dict(module.runtime_identity()),
            "target_id": selection.target_id,
            "bundle_ref": selection.metadata.get("bundle_ref"),
            "source_revision": selection.metadata.get("source_revision"),
            "source_commit": selection.metadata.get("source_commit"),
            "voice": dict(voice_source_identity(source)),
            "profile": options.get("profile", "wasm-uint8"),
            "providers": options.get("providers", "cpu"),
            "provider_options": options.get("provider_options"),
            "session_options": options.get("session_options"),
            "generation": {
                key: options.get(key)
                for key in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "steps",
                    "max_seconds",
                    "min_seconds",
                    "random_seed",
                    "text_chunk_size",
                )
            },
            "reference_prepare": {
                "reference_seconds": options.get("reference_seconds"),
                "reference_resampler": options.get("reference_resampler"),
            },
            "output": {"mode": options.get("output_mode", "native")},
            "voice_level": {
                "mode": options.get("voice_level", "off"),
                "gain_db": options.get("voice_level_gain_db"),
            },
        }

    def open(
        self, selection: EngineSelection
    ) -> AbstractContextManager[SoproSynthEngineSession]:
        module = _module()
        options = selection.options
        bundle_ref = selection.metadata.get("bundle_ref") or selection.target_id
        try:
            runtime = module.SoproRuntime.load(
                bundle_ref,
                profile=options.get("profile", "wasm-uint8"),
                providers=options.get("providers", "cpu"),
                provider_options=options.get("provider_options"),
                session_options=options.get("session_options"),
            )
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_error(exc, selection, stage="open") from exc

        @contextmanager
        def session() -> Any:
            try:
                yield SoproSynthEngineSession(runtime, selection)
            finally:
                try:
                    runtime.close()
                except EngineError:
                    raise
                except Exception as exc:
                    raise _translate_error(exc, selection, stage="close") from exc

        return session()


__all__ = [
    "ENGINE_ID",
    "PACKAGE_NAME",
    "SOPRO_OPTION_NAMES",
    "SoproSynthEngineAdapter",
    "SoproSynthEngineSession",
]
