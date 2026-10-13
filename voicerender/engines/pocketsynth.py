"""Request-centric adapter for the published PocketSynth runtime API."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
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
    language_tags_match,
    normalize_gender,
    normalize_language_key,
    normalize_locale_tag,
)
from .profiles import make_renderprofile_fragment

POCKET_OPTION_NAMES = frozenset(
    {
        "precision",
        "voice_source",
        "temperature",
        "lsd_steps",
        "max_frames",
        "frames_after_eos",
        "cache_dir",
        "catalog_path",
        "providers",
        "provider_options",
        "session_options",
        "voice_level",
        "force_download",
    }
)
_GENERATION_OPTION_NAMES = frozenset(
    {"temperature", "lsd_steps", "max_frames", "frames_after_eos"}
)


def _generation_config(module: Any, options: Mapping[str, Any]) -> Any:
    """Build Pocket generation configuration from explicit request overrides only."""
    kwargs: dict[str, Any] = {}
    if "temperature" in options:
        kwargs["temperature"] = float(options["temperature"])
    if "lsd_steps" in options:
        kwargs["lsd_steps"] = int(options["lsd_steps"])
    if "max_frames" in options:
        kwargs["max_frames"] = options["max_frames"]
    if "frames_after_eos" in options:
        kwargs["frames_after_eos"] = options["frames_after_eos"]
    return module.GenerationConfig(**kwargs)


class PocketSelectionError(ValueError):
    def __init__(self, code: str, message: str, field: str) -> None:
        super().__init__(message)
        self.diagnostic_code = code
        self.diagnostic_field = field


def _bundle_value(bundle: Any, name: str, default: Any = None) -> Any:
    if isinstance(bundle, Mapping):
        return bundle.get(name, default)
    return getattr(bundle, name, default)


def _pocket_target_voice(
    voice_id: str,
    *,
    detail: Any | None,
    bundle_language: str,
    languages: tuple[str, ...],
) -> TargetVoice:
    """Project descriptive voice metadata and authoritative bundle capability."""
    detail_language = (
        normalize_locale_tag(_bundle_value(detail, "language")) if detail else ""
    )
    detail_locale = (
        normalize_locale_tag(_bundle_value(detail, "locale")) if detail else ""
    )
    locale = detail_locale or detail_language or bundle_language
    voice_language = language_base(detail_language or locale or bundle_language)
    raw_label = _bundle_value(detail, "language_label") if detail else None
    language_label = raw_label.strip() if isinstance(raw_label, str) else ""
    if (
        not language_label
        or language_label.lower() in {"unknown", "na"}
        or (
            len(language_label) == 2
            and language_label.isalpha()
            and language_label.isupper()
        )
    ):
        language_label = locale or voice_language or "unknown"
    return TargetVoice(
        id=voice_id,
        gender=normalize_gender(_bundle_value(detail, "gender") if detail else None),
        language=voice_language or "unknown",
        locale=locale or "unknown",
        language_label=language_label,
        languages=languages,
    )


def _discovered_bundle_fields(
    bundle: Any,
) -> tuple[
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
    int | None,
    Mapping[str, Any],
    tuple[TargetVoice, ...],
]:
    bundle_id = str(_bundle_value(bundle, "id", ""))
    display_name = str(_bundle_value(bundle, "display_name", bundle_id))
    raw_language = _bundle_value(bundle, "language")
    language = (
        normalize_locale_tag(raw_language)
        if isinstance(raw_language, str) and raw_language.lower() != "unknown"
        else ""
    )
    languages = (language,) if language else ()
    voices = tuple(
        str(value) for value in (_bundle_value(bundle, "predefined_voices", ()) or ())
    )
    qualities = tuple(
        str(value) for value in (_bundle_value(bundle, "precisions", ()) or ())
    )
    raw_metadata = _bundle_value(bundle, "metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    source_revision = _bundle_value(bundle, "source_revision")
    maximum = _bundle_value(bundle, "max_tokens")
    default_voice = _bundle_value(bundle, "default_voice")
    metadata.update(
        {
            "language": language
            or (raw_language if isinstance(raw_language, str) else ""),
            "profiles": {quality: {} for quality in qualities},
            "qualities": qualities,
            "default_voice": default_voice,
            "source_revision": source_revision,
            "max_tokens": maximum,
            "max_token_per_chunk": maximum,
            "runtime_available": bool(_bundle_value(bundle, "runtime_available", True)),
        }
    )
    raw_voice_details = _bundle_value(bundle, "voice_details", ()) or ()
    details_by_id = {
        str(_bundle_value(detail, "id")): detail
        for detail in raw_voice_details
        if _bundle_value(detail, "id")
    }
    voice_details = tuple(
        _pocket_target_voice(
            voice_id,
            detail=details_by_id.get(voice_id),
            bundle_language=language,
            languages=languages,
        )
        for voice_id in voices
    )
    return (
        bundle_id,
        display_name,
        languages,
        voices,
        _bundle_value(bundle, "sample_rate"),
        metadata,
        voice_details,
    )


def _bundle_fields(
    bundle: Any,
) -> tuple[
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
    int | None,
    Mapping[str, Any],
    tuple[TargetVoice, ...],
]:
    if (
        _bundle_value(bundle, "ref") is not None
        and _bundle_value(bundle, "precisions") is not None
    ):
        return _discovered_bundle_fields(bundle)
    metadata = getattr(bundle, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    bundle_id = str(getattr(bundle, "id", None) or getattr(bundle, "bundle_id", ""))
    display_name = str(
        metadata.get("display_name") or metadata.get("name") or bundle_id
    )
    raw_language = metadata.get("language")
    language = (
        normalize_locale_tag(raw_language) if isinstance(raw_language, str) else ""
    )
    languages = (language,) if language else ()
    raw_voices = metadata.get("predefined_voice_names", getattr(bundle, "voices", ()))
    voices = tuple(str(name) for name in raw_voices if isinstance(name, str))
    profiles = metadata.get("profiles", {})
    qualities = (
        tuple(sorted(str(name) for name in profiles))
        if isinstance(profiles, Mapping)
        else ()
    )
    raw_voice_details = metadata.get("voice_details")
    details_by_id = (
        {
            str(detail["id"]): detail
            for detail in raw_voice_details
            if isinstance(detail, Mapping)
            and isinstance(detail.get("id"), str)
            and detail.get("id")
        }
        if isinstance(raw_voice_details, (list, tuple))
        else {}
    )
    voice_details = tuple(
        _pocket_target_voice(
            voice_id,
            detail=details_by_id.get(voice_id),
            bundle_language=language,
            languages=languages,
        )
        for voice_id in voices
    )
    sample_rate = getattr(bundle, "sample_rate", None)
    normalized_metadata = {
        **dict(metadata),
        "language": language,
        "profiles": dict(profiles) if isinstance(profiles, Mapping) else {},
        "qualities": qualities,
    }
    return (
        bundle_id,
        display_name,
        languages,
        voices,
        sample_rate,
        normalized_metadata,
        tuple(voice_details),
    )


def _matches_language(requested: str, available: str) -> bool:
    return language_tags_match(requested, available)


def _voice_source(selection: EngineSelection) -> Mapping[str, Any] | None:
    source = selection.metadata.get("voice_source")
    if not isinstance(source, Mapping):
        return None
    kind = source.get("kind")
    value = source.get("value", source.get("path"))
    if (
        kind not in {"named", "reference", "managed_reference"}
        or not isinstance(value, str)
        or not value
    ):
        raise ValueError(
            "pocket.voice_source_invalid: expected a named, reference, or managed reference source"
        )
    if kind in {"reference", "managed_reference"} and not isinstance(
        source.get("sha256"), str
    ):
        raise ValueError(
            f"pocket.voice_source_invalid: {kind} voice source requires sha256"
        )
    normalized: dict[str, Any] = {"kind": kind, "value": value}
    for name in (
        "sha256",
        "source_revision",
        "source_repository",
        "source_path",
        "license",
        "dataset",
        "variant",
    ):
        item = source.get(name)
        if item is not None:
            if not isinstance(item, str) or not item:
                raise ValueError(
                    f"pocket.voice_source_invalid: {name} must be a non-empty string"
                )
            normalized[name] = item
    return normalized


def _voice_identity(selection: EngineSelection) -> dict[str, str] | None:
    source = _voice_source(selection)
    if source is not None:
        if source["kind"] == "reference":
            return {"kind": "reference", "sha256": str(source["sha256"])}
        if source["kind"] == "managed_reference":
            identity = {
                "kind": "managed_reference",
                "ref": str(source["value"]),
                "sha256": str(source["sha256"]),
            }
            if source.get("source_revision") is not None:
                identity["source_revision"] = str(source["source_revision"])
            return identity
        return {"kind": "named", "value": str(source["value"])}
    if selection.voice:
        return {"kind": "named", "value": selection.voice}
    return None


def _reference_path(source: Mapping[str, Any]) -> Path:
    return Path(str(source["value"])).expanduser()


def _reference_sha256(source: Mapping[str, Any]) -> str:
    path = _reference_path(source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = str(source["sha256"])
    if digest != expected:
        raise ValueError(
            f"pocket.reference_voice_changed: expected {expected}, found {digest} for {path}"
        )
    return digest


def _manager(*, options: Mapping[str, Any], offline: bool) -> Any:
    import pocketsynth

    return pocketsynth.BundleAssetManager(
        cache_dir=options.get("cache_dir"),
        catalog_path=options.get("catalog_path"),
        offline=offline,
    )


def _discover_bundles(
    *,
    options: Mapping[str, Any],
    language: str | None,
    offline: bool,
    refresh: bool,
) -> tuple[Any, ...]:
    import pocketsynth

    discovery = getattr(pocketsynth, "discover_bundles", None)
    if callable(discovery):
        return tuple(
            discovery(
                language=language,
                offline=offline,
                refresh=refresh,
                cache_dir=options.get("cache_dir"),
                catalog_path=options.get("catalog_path"),
            )
        )
    manager = _manager(options=options, offline=offline)
    return tuple(manager.list_bundles(language=language, refresh=refresh))


def _find_bundle(
    bundles: tuple[Any, ...], target_id: str
) -> (
    tuple[
        Any,
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            int | None,
            Mapping[str, Any],
            tuple[TargetVoice, ...],
        ],
    ]
    | None
):
    for bundle in bundles:
        fields = _bundle_fields(bundle)
        aliases = getattr(bundle, "aliases", ()) or ()
        if target_id == fields[0] or target_id in aliases:
            return bundle, fields
    return None


def _translate_pocket_error(
    error: Exception,
    selection: EngineSelection,
    request: SpeechRequest | None = None,
    *,
    opening: bool = False,
) -> EngineError:
    import pocketsynth

    error_map = (
        ("SynthesisInputTooLongError", SpeechRequestTooLongError),
        ("EmptyTextError", EmptySpeechTextError),
        ("InvalidLanguageError", InvalidEngineLanguageError),
        ("BundleLanguageError", InvalidEngineLanguageError),
        ("InvalidVoiceError", InvalidEngineVoiceError),
        ("InvalidGenerationConfigError", InvalidEngineOptionError),
        ("InvalidRequestError", InvalidSpeechRequestError),
        ("UnsupportedFeatureError", UnsupportedSynthesisFeatureError),
        ("BundleNotFoundError", InvalidEngineModelError),
        ("UnsupportedBundleError", InvalidEngineModelError),
        ("ModelInferenceError", EngineBackendError),
        ("RuntimeClosedError", EngineBackendError),
    )
    error_type: type[EngineError] = EngineBackendError
    for native_name, mapped_type in error_map:
        native_type = getattr(pocketsynth, native_name, None)
        if isinstance(native_type, type) and isinstance(error, native_type):
            error_type = mapped_type
            break
    else:
        if opening and isinstance(error, (TypeError, ValueError)):
            error_type = InvalidEngineOptionError
        elif isinstance(error, (TypeError, ValueError)):
            error_type = InvalidSpeechRequestError

    context: dict[str, Any] = {
        "engine": "pocket",
        "engine_version": distribution_version("pocketsynth"),
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
            source="pocketsynth.runtime",
        )
    return error_type(str(error), **context)


class PocketSynthEngineSession:
    """Adapt one exact VoiceRender request to PocketSynth's strict runtime API."""

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
        self._voices: dict[tuple[str, ...], Any] = {}

    def _voice(self, request: SpeechRequest) -> Any:
        request_source = request.options.get("voice_source")
        source = None
        if isinstance(request_source, Mapping):
            source_selection = EngineSelection(
                engine=self._selection.engine,
                target_id=self._selection.target_id,
                language=self._selection.language,
                voice=request.voice,
                options=self._selection.options,
                metadata={"voice_source": request_source},
            )
            source = _voice_source(source_selection)
        if source is None:
            source = _voice_source(self._selection)
        if source is None:
            voice_name = request.voice or self._selection.voice
            if not voice_name:
                raise ValueError("PocketSynth requires a predefined or reference voice")
            source = {"kind": "named", "value": voice_name}

        kind = str(source["kind"])
        key: tuple[str, ...]
        if kind == "reference":
            _reference_sha256(source)
            key = ("reference", str(source["sha256"]))
            voice_input: str | Path = _reference_path(source)
        elif kind == "managed_reference":
            key = (
                "managed_reference",
                str(source["value"]),
                str(source["sha256"]),
                str(source.get("source_revision") or ""),
            )
            voice_input = str(source["value"])
        else:
            key = ("named", str(source["value"]))
            voice_input = str(source["value"])
        if key not in self._voices:
            if kind == "managed_reference":
                try:
                    prepared = self._runtime.prepare_voice(voice_input)
                except Exception as exc:
                    if self._selection.offline:
                        raise EngineBackendError(
                            f"Managed Pocket prompt {source['value']!r} is unavailable offline: {exc}",
                            engine="pocket",
                            target_id=self._selection.target_id,
                            native_error_type=type(exc).__name__,
                            details={
                                "voice_prompt": source["value"],
                                "expected_sha256": source["sha256"],
                                "expected_revision": source.get("source_revision"),
                                "offline": True,
                            },
                            code="pocket.voice_prompt_offline_unavailable",
                        ) from exc
                    raise EngineBackendError(
                        f"Failed to prepare managed Pocket prompt {source['value']!r}: {exc}",
                        engine="pocket",
                        target_id=self._selection.target_id,
                        native_error_type=type(exc).__name__,
                        details={"voice_prompt": source["value"]},
                        code="pocket.managed_reference_prepare_failed",
                    ) from exc
                _verify_managed_voice(prepared, source, self._selection)
            else:
                prepared = self._runtime.prepare_voice(voice_input)
            self._voices[key] = prepared
        return self._voices[key]

    def _native_request(self, request: SpeechRequest, module: Any) -> Any:
        if request.tokens:
            raise UnsupportedSynthesisFeatureError(
                "PocketSynth does not support linguistic tokens",
                engine="pocket",
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        if request.pronunciation_overrides:
            raise UnsupportedSynthesisFeatureError(
                "PocketSynth does not support pronunciation overrides",
                engine="pocket",
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        if request.whole_request_phonemes is not None:
            raise UnsupportedSynthesisFeatureError(
                "PocketSynth does not support whole-request phonemes",
                engine="pocket",
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        if request.speaker is not None:
            raise UnsupportedSynthesisFeatureError(
                "PocketSynth does not support speaker selection",
                engine="pocket",
                target_id=self._selection.target_id,
                request_id=request.id,
            )
        return module.SynthesisRequest(
            id=request.id,
            text=request.text,
            language=request.language,
        )

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        import pocketsynth

        native = self._native_request(request, pocketsynth)
        measure_request = getattr(self._runtime, "measure_request", None)
        if callable(measure_request):
            measured = measure_request(native)
            return RequestMeasure(
                fits=measured.fits,
                amount=measured.amount,
                maximum=measured.maximum,
                unit="model_tokens",
                source="pocketsynth.measure_request",
            )

        frontend = getattr(self._runtime, "frontend", None)
        encode = getattr(frontend, "encode", None)
        if not callable(encode):
            return RequestMeasure(
                fits=None,
                amount=None,
                maximum=None,
                unit="unknown",
                source="pocketsynth.no_capacity_measurement",
            )
        amount = len(encode(request.text))
        metadata = getattr(self._runtime, "metadata", None)
        raw_maximum = getattr(metadata, "max_token_per_chunk", None)
        maximum = int(raw_maximum) if raw_maximum is not None else None
        return RequestMeasure(
            fits=amount <= maximum if maximum is not None else None,
            amount=amount,
            maximum=maximum,
            unit="model_tokens",
            source="pocketsynth.frontend",
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        import pocketsynth

        try:
            native = self._native_request(request, pocketsynth)
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_pocket_error(exc, self._selection, request) from exc
        try:
            voice = self._voice(request)
        except (OSError, ValueError) as exc:
            raise InvalidEngineVoiceError(
                str(exc),
                engine="pocket",
                target_id=self._selection.target_id,
                language=request.language,
                voice=request.voice,
                request_id=request.id,
                native_error_type=type(exc).__name__,
            ) from exc
        try:
            result = self._runtime.synthesize(
                native,
                voice=voice,
                config=self._generation,
                voice_level=self._voice_level,
            )
        except EngineSynthesisError:
            raise
        except Exception as exc:
            raise _translate_pocket_error(exc, self._selection, request) from exc

        metadata = dict(result.metadata or {})
        voice_identity = _voice_identity_for_request(self._selection, request)
        metadata.update(
            bundle_id=self._selection.target_id,
            precision=self._selection.options.get("precision", "int8"),
            voice=voice_identity,
            voice_level=voice_level_metadata(
                metadata.get("voice_level_application"),
                self._voice_level.mode,
                identity_keys=("identity", "calibration_identity"),
                revision_keys=("catalog_revision", "bundle_revision"),
            ),
        )
        if (
            voice_identity is not None
            and voice_identity.get("kind") == "managed_reference"
        ):
            prepared_fingerprint = getattr(voice, "fingerprint", None)
            if prepared_fingerprint is not None:
                metadata["prepared_voice_fingerprint"] = prepared_fingerprint
        if metadata.get("token_count") is None:
            measured = self.measure(request)
            if measured.amount is not None:
                metadata["token_count"] = measured.amount
        rendered = RenderedSpeech(
            id=result.id,
            audio=result.audio,
            sample_rate=result.sample_rate,
            warnings=tuple(result.warnings),
            word_timings=(),
            metadata=metadata,
        )
        return validate_rendered_speech(
            request,
            rendered,
            engine="pocket",
            engine_version=distribution_version("pocketsynth"),
            target_id=self._selection.target_id,
        )


def _verify_managed_voice(
    prepared: Any, source: Mapping[str, Any], selection: EngineSelection
) -> None:
    metadata = getattr(prepared, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    prompt = getattr(prepared, "voice_prompt", None)
    actual_ref = getattr(prompt, "ref", None) or metadata.get("managed_ref")
    actual_sha256 = getattr(prompt, "sha256", None) or metadata.get("source_sha256")
    actual_revision = getattr(prompt, "source_revision", None) or metadata.get(
        "source_revision"
    )
    expected_revision = source.get("source_revision")
    if (
        actual_ref != source["value"]
        or actual_sha256 != source["sha256"]
        or (expected_revision is not None and actual_revision != expected_revision)
    ):
        raise EngineBackendError(
            "Prepared Pocket voice prompt provenance does not match the resolved render plan.",
            engine="pocket",
            target_id=selection.target_id,
            voice=str(source["value"]),
            details={
                "expected_ref": source["value"],
                "actual_ref": actual_ref,
                "expected_sha256": source["sha256"],
                "actual_sha256": actual_sha256,
                "expected_revision": expected_revision,
                "actual_revision": actual_revision,
            },
            code="pocket.managed_reference_changed",
        )


def _voice_identity_for_request(
    selection: EngineSelection, request: SpeechRequest
) -> dict[str, str] | None:
    source = request.options.get("voice_source")
    if isinstance(source, Mapping):
        request_selection = EngineSelection(
            engine=selection.engine,
            target_id=selection.target_id,
            language=selection.language,
            options=selection.options,
            metadata={"voice_source": source},
        )
        return _voice_identity(request_selection)
    return _voice_identity(selection) or (
        {"kind": "named", "value": request.voice} if request.voice else None
    )


class PocketSynthEngineAdapter:
    """Pocket bundle discovery, selection, runtime and request adaptation."""

    id = "pocket"
    package_name = "pocketsynth"

    def version(self) -> str | None:
        return distribution_version(self.package_name)

    def probe_api(self) -> EngineApiProbe:
        return probe_public_api(
            engine=self.id,
            package=self.package_name,
            required_symbols=(
                "PocketRuntime",
                "SynthesisRequest",
                "SynthesisResult",
                "GenerationConfig",
                "VoiceLevelConfig",
                "SynthesisInputTooLongError",
                "BundleAssetManager",
                "VoicePromptInfo",
                "PreparedVoice",
                "inspect_voice_prompt",
                "list_voice_prompts",
            ),
            required_methods={
                "PocketRuntime": (
                    "from_resolved",
                    "prepare_voice",
                    "synthesize",
                    "close",
                ),
                "BundleAssetManager": ("resolve_bundle",),
                "__module__": ("inspect_voice_prompt", "list_voice_prompts"),
            },
            expected_api_version=SUPPORTED_REQUEST_API_VERSION,
            contract_required_symbols=(
                "RequestMeasure",
                "discover_bundles",
                "runtime_identity",
            ),
            contract_required_methods={
                "PocketRuntime": ("measure_request",),
                "__module__": ("discover_bundles", "runtime_identity"),
            },
            legacy_required_methods={"BundleAssetManager": ("list_bundles",)},
        )

    def compatible_api(self) -> bool:
        return self.probe_api().compatible

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            voice_binding_namespace="pocket",
            supports_named_voices=True,
            supports_reference_voice=True,
            supports_qualities=True,
            option_names=POCKET_OPTION_NAMES,
            supports_voice_level_calibration=True,
            supports_request_measurement=True,
        )

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            frontend="pocketsynth",
            pronunciation="unsupported",
            capacity_key="max_token_per_chunk",
            capacity_metric="model_tokens",
        )

    def list_voice_prompts(
        self,
        *,
        dataset: str | None = None,
        variant: str | None = None,
        license: str | None = None,
        offline: bool = False,
        refresh: bool = False,
    ) -> tuple[Any, ...]:
        import pocketsynth

        return tuple(
            pocketsynth.list_voice_prompts(
                dataset=dataset,
                variant=variant,
                license=license,
                offline=offline,
                refresh=refresh,
            )
        )

    def discover(self, request: CatalogRequest) -> tuple[SynthesisTarget, ...]:
        requested = normalize_locale_tag(request.language) if request.language else None
        upstream_language = language_base(requested) if requested else None
        try:
            bundles = _discover_bundles(
                options={},
                language=upstream_language,
                offline=request.offline,
                refresh=request.refresh,
            )
        except ImportError:
            return ()
        targets = []
        for bundle in bundles:
            (
                bundle_id,
                display_name,
                languages,
                voices,
                sample_rate,
                metadata,
                voice_details,
            ) = _bundle_fields(bundle)
            raw_default_voice = metadata.get("default_voice")
            runtime_available = bool(metadata.get("runtime_available", True))
            status = "ready" if runtime_available else "runtime_unavailable"
            targets.append(
                SynthesisTarget(
                    engine=self.id,
                    id=bundle_id,
                    display_name=display_name,
                    status=status,
                    runtime_available=runtime_available,
                    languages=languages,
                    sample_rate=sample_rate,
                    voices=voices,
                    voice_details=voice_details,
                    default_voice=(
                        str(raw_default_voice)
                        if isinstance(raw_default_voice, str)
                        else None
                    ),
                    qualities=tuple(metadata["qualities"]),
                    aliases=tuple(getattr(bundle, "aliases", ()) or ()),
                    capabilities=frozenset({"predefined_voice", "reference_voice"}),
                    metadata=metadata,
                )
            )
        return tuple(
            target
            for target in targets
            if requested is None
            or any(
                language_tags_match(requested, available)
                for available in target.languages
            )
        )

    def validate_request(self, request: Any) -> None:
        """Validate bundle selection without touching PocketSynth runtime APIs."""
        language = normalize_language_key(getattr(request, "language", None) or "en-us")
        options = dict(getattr(request, "options", {}) or {})
        options.update(dict(getattr(request, "engine_options", {}) or {}))
        if not (getattr(request, "target_id", None) or options.get("bundle")):
            raise ValueError(
                "pocket.bundle_required: PocketSynth requires a bundle; pass --model <bundle-id> "
                f"or run `voicerender voices list --engine pocket --lang {language}`."
            )

    def resolve(self, request: Any) -> tuple[EngineSelection, tuple[Any, ...]]:
        language = normalize_language_key(getattr(request, "language", None) or "en-us")
        options = dict(getattr(request, "options", {}) or {})
        options.update(dict(getattr(request, "engine_options", {}) or {}))
        target_id = getattr(request, "target_id", None) or options.get("bundle")
        self.validate_request(request)
        precision = options.get("precision", options.get("quality", "int8"))
        if not isinstance(precision, str) or not precision:
            raise ValueError("pocket.precision_invalid: precision must be int8 or fp32")
        voice_source = options.get("voice_source")
        if (
            isinstance(voice_source, Mapping)
            and voice_source.get("kind") == "managed_reference"
        ):
            prompt_ref = voice_source.get("value")
            if not isinstance(prompt_ref, str) or not prompt_ref:
                raise PocketSelectionError(
                    "pocket.voice_prompt_invalid",
                    "PocketSynth managed voice prompts require a non-empty reference.",
                    "synthesis.voice_prompt",
                )
            import pocketsynth

            try:
                prompt_info = pocketsynth.inspect_voice_prompt(
                    prompt_ref,
                    cache_dir=options.get("cache_dir"),
                    catalog_path=options.get("catalog_path"),
                    offline=bool(getattr(request, "offline", False)),
                    refresh=bool(getattr(request, "refresh", False)),
                )
            except (OSError, pocketsynth.PocketSynthError) as exc:
                raise PocketSelectionError(
                    "pocket.voice_prompt_unavailable",
                    f"PocketSynth managed voice prompt metadata is unavailable: {exc}",
                    "synthesis.voice_prompt",
                ) from exc
            voice_source = {
                "kind": "managed_reference",
                "value": prompt_info.ref,
                "sha256": prompt_info.sha256,
                "source_revision": prompt_info.source_revision,
                "source_repository": prompt_info.source_repository,
                "source_path": prompt_info.source_path,
                "license": prompt_info.license,
                "dataset": prompt_info.dataset,
                "variant": prompt_info.variant,
            }
        metadata = (
            {"voice_source": dict(voice_source)}
            if isinstance(voice_source, Mapping)
            else {}
        )
        runtime_options = {
            key: value
            for key, value in options.items()
            if key in POCKET_OPTION_NAMES and key != "voice_source"
        }
        runtime_options["precision"] = precision
        return (
            EngineSelection(
                engine=self.id,
                target_id=str(target_id),
                language=language,
                voice=getattr(request, "voice", None),
                speaker=getattr(request, "speaker", None),
                options=runtime_options,
                metadata=metadata,
                offline=bool(getattr(request, "offline", False)),
                refresh=bool(getattr(request, "refresh", False)),
            ),
            (),
        )

    def validate_selection(self, selection: EngineSelection) -> tuple[Any, ...]:
        from .base import EngineDiagnostic

        try:
            import pocketsynth
        except ImportError as exc:
            return (
                EngineDiagnostic(
                    code="pocket.runtime_unavailable",
                    severity="error",
                    message=f"PocketSynth runtime dependencies are unavailable: {exc}",
                    field="synthesis.engine",
                ),
            )
        try:
            _generation_config(pocketsynth, selection.options)
        except (TypeError, ValueError) as exc:
            return (
                EngineDiagnostic(
                    code="pocket.generation_config_invalid",
                    severity="error",
                    message=f"Invalid PocketSynth generation options: {exc}",
                    field="render.options",
                ),
            )
        try:
            bundles = _discover_bundles(
                options=selection.options,
                language=None,
                offline=selection.offline,
                refresh=selection.refresh,
            )
        except (OSError, ValueError, pocketsynth.PocketSynthError) as exc:
            return (
                EngineDiagnostic(
                    code="pocket.catalog_unavailable",
                    severity="error",
                    message=f"PocketSynth bundle catalog is unavailable: {exc}",
                    field="synthesis.engine",
                ),
            )
        found = _find_bundle(bundles, selection.target_id)
        if found is None:
            return (
                EngineDiagnostic(
                    code="pocket.bundle_not_found",
                    severity="error",
                    message=f"PocketSynth bundle {selection.target_id!r} was not found.",
                    field="render.target.id",
                ),
            )
        _bundle, fields = found
        (
            bundle_id,
            _display_name,
            languages,
            voices,
            _sample_rate,
            metadata,
            _voice_details,
        ) = fields
        diagnostics = []
        if languages and not any(
            _matches_language(selection.language, item) for item in languages
        ):
            diagnostics.append(
                EngineDiagnostic(
                    code="engine_language_incompatible",
                    severity="error",
                    message=(
                        f"PocketSynth bundle {bundle_id!r} supports {', '.join(languages)}, "
                        f"not {selection.language!r}."
                    ),
                    field="render.target.language",
                )
            )
        precision = str(selection.options.get("precision", "int8"))
        if precision not in metadata["qualities"]:
            diagnostics.append(
                EngineDiagnostic(
                    code="pocket.precision_unavailable",
                    severity="error",
                    message=(
                        f"Precision {precision!r} is not available on PocketSynth bundle "
                        f"{bundle_id!r}."
                    ),
                    field="render.options.precision",
                )
            )
        try:
            source = _voice_source(selection)
        except ValueError as exc:
            return (
                EngineDiagnostic(
                    code="pocket.voice_source_invalid",
                    severity="error",
                    message=str(exc),
                    field="render.target.voice",
                ),
            )
        voice_kind = (
            source["kind"]
            if source is not None
            else ("named" if selection.voice else None)
        )
        voice_name = str(source["value"]) if source is not None else selection.voice
        if voice_kind is None:
            diagnostics.append(
                EngineDiagnostic(
                    code="pocket.voice_required",
                    severity="error",
                    message="Select a PocketSynth predefined voice or reference WAV voice.",
                    field="render.target.voice",
                )
            )
        elif voice_kind == "named" and voice_name not in voices:
            diagnostics.append(
                EngineDiagnostic(
                    code="pocket.voice_not_found",
                    severity="error",
                    message=(
                        f"Predefined voice {voice_name!r} is not available on "
                        f"PocketSynth bundle {bundle_id!r}."
                    ),
                    field="render.target.voice",
                )
            )
        elif voice_kind == "reference" and source is not None:
            try:
                _reference_sha256(source)
            except (OSError, ValueError) as exc:
                diagnostics.append(
                    EngineDiagnostic(
                        code="pocket.reference_voice_invalid",
                        severity="error",
                        message=str(exc),
                        field="render.target.voice",
                    )
                )
        return tuple(diagnostics)

    def target_metadata(self, selection: EngineSelection) -> Mapping[str, Any]:
        try:
            import pocketsynth
        except ImportError:
            return {}
        try:
            bundles = _discover_bundles(
                options=selection.options,
                language=None,
                offline=selection.offline,
                refresh=selection.refresh,
            )
        except (OSError, ValueError, pocketsynth.PocketSynthError):
            return {}
        found = _find_bundle(bundles, selection.target_id)
        if found is None:
            return {}
        (
            bundle_id,
            _display_name,
            languages,
            voices,
            sample_rate,
            metadata,
            _voice_details,
        ) = found[1]
        result: dict[str, Any] = {
            "bundle_id": bundle_id,
            "languages": languages,
            "predefined_voices": voices,
            "sample_rate": sample_rate,
            "max_token_per_chunk": metadata.get("max_token_per_chunk"),
            "precision_profiles": metadata["qualities"],
            "source_revision": metadata.get("source_revision"),
        }
        if "voice_source" in selection.metadata:
            result["voice_source"] = selection.metadata["voice_source"]
        return result

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> Mapping[str, Any]:
        voice = _voice_identity(selection)
        if voice is None:
            voice = {"kind": "named", "value": selection.voice or ""}
        identity: dict[str, Any] = {
            "source_revision": selection.metadata.get("source_revision"),
            "engine": self.id,
            "engine_version": self.version(),
            "target_id": selection.target_id,
            "language": normalize_language_key(selection.language),
            "precision": selection.options.get("precision", "int8"),
            "voice_level": selection.options.get("voice_level", "off"),
            "voice": voice,
            "generation": {
                key: selection.options[key]
                for key in sorted(_GENERATION_OPTION_NAMES)
                if key in selection.options
            },
        }
        try:
            import pocketsynth
        except ImportError:
            return identity
        runtime_identity = getattr(pocketsynth, "runtime_identity", None)
        if callable(runtime_identity):
            identity["engine_identity"] = dict(runtime_identity())
        return identity

    def open(
        self, selection: EngineSelection
    ) -> AbstractContextManager[PocketSynthEngineSession]:
        import pocketsynth

        options = dict(selection.options)
        try:
            generation = _generation_config(pocketsynth, options)
            voice_level = pocketsynth.VoiceLevelConfig(
                mode=options.get("voice_level", "off")
            )
            bundle = _manager(
                options=options, offline=selection.offline
            ).resolve_bundle(
                selection.target_id,
                precision=str(options.get("precision", "int8")),
                refresh_catalog=selection.refresh,
                force_download=bool(options.get("force_download", False)),
            )
            runtime = pocketsynth.PocketRuntime.from_resolved(
                bundle,
                cache_dir=options.get("cache_dir"),
                offline=selection.offline,
                providers=options.get("providers"),
                provider_options=options.get("provider_options"),
                session_options=options.get("session_options"),
            )
        except Exception as exc:
            raise _translate_pocket_error(exc, selection, opening=True) from exc

        @contextmanager
        def session() -> Any:
            try:
                yield PocketSynthEngineSession(
                    runtime, selection, generation, voice_level
                )
            finally:
                runtime.close()

        return session()
