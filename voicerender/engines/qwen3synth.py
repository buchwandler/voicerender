"""Lazy Qwen3Synth adapter for VoiceRender."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from ..errors import (
    EngineBackendError,
    EngineCapacityError,
    EngineError,
    EngineSelectionError,
    EngineUnavailableError,
    InvalidEngineVoiceError,
    UnsupportedSynthesisFeatureError,
)
from ..voice_sources import (
    normalize_voice_source,
    resolve_prepared_voice,
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
    validate_rendered_speech,
)
from .catalog import CatalogRequest, SynthesisTarget
from .metadata import language_base, normalize_locale_tag
from .profiles import make_renderprofile_fragment
from .selection import EngineRequest

ENGINE_ID = "qwen3"
PACKAGE_NAME = "qwen3synth"

QWEN3_QVOICE_FORMAT = "qwen3.qvoice.v2"
QWEN3_OPTION_NAMES = frozenset(
    {
        "quality",
        "provider",
        "catalog_path",
        "cache_dir",
        "temperature",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
        "seed",
        "reference_text",
        "voice_source",
    }
)

_REQUIRED_SYMBOLS = (
    "Qwen3Runtime",
    "SynthesisRequest",
    "SynthesisResult",
    "GenerationConfig",
    "ReferenceVoice",
    "PreparedVoice",
    "RequestMeasure",
    "inspect_voice_profile",
    "discover_bundles",
    "runtime_identity",
    "REQUEST_API_VERSION",
    "request_api_contract",
)
_REQUIRED_METHODS = {
    "Qwen3Runtime": (
        "from_installed",
        "measure_request",
        "prepare_voice",
        "synthesize",
        "close",
    ),
    "PreparedVoice": ("load",),
    "__module__": ("discover_bundles", "runtime_identity", "request_api_contract"),
}


def _module() -> Any:
    return importlib.import_module(PACKAGE_NAME)


def _value(owner: Any, name: str, default: Any = None) -> Any:
    if isinstance(owner, Mapping):
        return owner.get(name, default)
    return getattr(owner, name, default)


def _contract_value(contract: Any, name: str, default: Any = None) -> Any:
    return _value(contract, name, default)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(
        dict.fromkeys(item for item in value if isinstance(item, str) and item)
    )


def _target_from_bundle(bundle: Any) -> SynthesisTarget:
    bundle_id = _value(bundle, "id")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ValueError("Qwen3Synth discovery returned a bundle without an id")

    model_type = _value(bundle, "model_type")
    raw_metadata = _value(bundle, "metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    languages = _string_tuple(_value(bundle, "languages", ()))
    precisions = _string_tuple(_value(bundle, "precisions", ()))
    sample_rate = _value(bundle, "sample_rate")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
    ):
        sample_rate = None
    runtime_available = _value(bundle, "runtime_available", False) is True
    ref = _value(bundle, "ref")
    display_name = _value(bundle, "display_name")
    layout = _value(bundle, "layout")
    source_revision = _value(bundle, "source_revision")

    metadata.update(
        {
            "ref": ref,
            "model_type": model_type,
            "model_id": raw_metadata.get("model_id"),
            "layout": layout,
            "sample_rate": sample_rate,
            "languages": languages,
            "precisions": precisions,
            "source_revision": source_revision,
            "runtime_available": runtime_available,
        }
    )
    if model_type == "base":
        capabilities = frozenset(
            {"reference_voice", "prepared_voice", "request_measurement"}
        )
    elif model_type == "voice_design":
        capabilities = frozenset({"voice_design"})
    else:
        capabilities = frozenset()

    return SynthesisTarget(
        engine=ENGINE_ID,
        id=bundle_id,
        display_name=(
            display_name
            if isinstance(display_name, str) and display_name
            else bundle_id
        ),
        languages=languages,
        status="ready",
        runtime_available=runtime_available,
        sample_rate=sample_rate,
        qualities=precisions,
        capabilities=capabilities,
        metadata=metadata,
    )


def _find_target(
    targets: tuple[SynthesisTarget, ...], target_id: str
) -> SynthesisTarget | None:
    requested = target_id.casefold()
    return next(
        (target for target in targets if target.id.casefold() == requested), None
    )


def _language(value: str | None) -> str:
    normalized = normalize_locale_tag(value or "en-US")
    return language_base(normalized) or normalized


def _reference_text(selection: EngineSelection, source: Any) -> str:
    value = selection.options.get("reference_text")
    if value is None and source.origin is not None:
        value = source.origin.get("reference_text")
    if not isinstance(value, str) or not value.strip():
        raise EngineSelectionError(
            "Qwen3Synth reference voices require an exact reference transcript.",
            engine=ENGINE_ID,
            target_id=selection.target_id,
            code="qwen3.reference_text_required",
        )
    return value


def _selected_voice_source(selection: EngineSelection) -> Any:
    value = selection.options.get("voice_source")
    if not isinstance(value, Mapping):
        raise EngineSelectionError(
            "Qwen3Synth requires a pinned reference or prepared voice_source.",
            engine=ENGINE_ID,
            target_id=selection.target_id,
            code="qwen3.reference_required",
        )
    return normalize_voice_source(value, allow_named=False)


def _voice_identity(selection: EngineSelection) -> dict[str, Any]:
    source = _selected_voice_source(selection)
    identity = dict(voice_source_identity(source))
    if source.kind in {"reference", "managed_reference"}:
        reference_text = _reference_text(selection, source)
        identity["reference_text_sha256"] = hashlib.sha256(
            reference_text.encode("utf-8")
        ).hexdigest()
    return identity


def _translate_qwen_error(
    exc: Exception,
    selection: EngineSelection,
    *,
    operation: str,
    request: SpeechRequest | None = None,
    fallback_type: type[EngineError] | None = None,
    fallback_code: str | None = None,
) -> EngineError:
    if isinstance(exc, EngineError):
        return exc
    module = sys.modules.get(PACKAGE_NAME)
    native_mappings: tuple[tuple[str, type[EngineError], str], ...] = (
        ("ModelNotInstalledError", EngineUnavailableError, "qwen3.model_not_installed"),
        ("CatalogError", EngineUnavailableError, "qwen3.catalog_unavailable"),
        ("SynthesisInputTooLongError", EngineCapacityError, "qwen3.request_too_long"),
        (
            "InvalidReferenceAudioError",
            InvalidEngineVoiceError,
            "qwen3.reference_invalid",
        ),
        (
            "InvalidReferenceTranscriptError",
            InvalidEngineVoiceError,
            "qwen3.reference_text_invalid",
        ),
        (
            "IncompatibleVoiceProfileError",
            InvalidEngineVoiceError,
            "qwen3.profile_incompatible",
        ),
        (
            "InvalidModelLayoutError",
            InvalidEngineVoiceError,
            "qwen3.profile_incompatible",
        ),
        (
            "UnsupportedModelCapabilityError",
            UnsupportedSynthesisFeatureError,
            "qwen3.capability_unsupported",
        ),
        ("ModelGenerationError", EngineBackendError, "qwen3.backend_failed"),
    )
    error_type = fallback_type or EngineBackendError
    code = fallback_code or "qwen3.backend_failed"
    if module is not None:
        for native_name, mapped_type, mapped_code in native_mappings:
            native_type = getattr(module, native_name, None)
            if isinstance(native_type, type) and isinstance(exc, native_type):
                error_type, code = mapped_type, mapped_code
                break
    native_name = type(exc).__name__
    version = getattr(module, "__version__", None) if module is not None else None
    message = f"Qwen3Synth {operation} failed: {exc}"
    return error_type(
        message,
        engine=ENGINE_ID,
        engine_version=version if isinstance(version, str) else None,
        target_id=selection.target_id,
        language=request.language if request is not None else selection.language,
        voice=request.voice if request is not None else None,
        speaker=request.speaker if request is not None else None,
        request_id=request.id if request is not None else None,
        native_error_type=native_name,
        details={"native_error_type": native_name},
        code=code,
    )


class Qwen3SynthEngineSession:
    """One installed Qwen Base runtime and its lazily prepared voice state."""

    def __init__(self, runtime: Any, selection: EngineSelection, module: Any) -> None:
        self._runtime = runtime
        self._selection = selection
        self._module = module
        self._voice: Any | None = None
        self._voice_identity = _voice_identity(selection)

    def measure(self, request: SpeechRequest) -> RequestMeasure:
        native_request = self._module.SynthesisRequest(
            id=request.id,
            text=request.text,
            language=_language(request.language),
        )
        try:
            measured = self._runtime.measure_request(native_request)
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_qwen_error(
                exc, self._selection, operation="request measurement", request=request
            ) from exc

        raw_amount = _value(measured, "amount")
        raw_maximum = _value(measured, "maximum")
        raw_fits = _value(measured, "fits")
        amount = (
            raw_amount
            if isinstance(raw_amount, int)
            and not isinstance(raw_amount, bool)
            and raw_amount >= 0
            else None
        )
        maximum = (
            raw_maximum
            if isinstance(raw_maximum, int)
            and not isinstance(raw_maximum, bool)
            and raw_maximum > 0
            else None
        )
        invalid_fields = []
        if amount is None:
            invalid_fields.append("amount")
        if maximum is None:
            invalid_fields.append("maximum")
        if not isinstance(raw_fits, bool):
            invalid_fields.append("fits")
        return RequestMeasure(
            fits=raw_fits if isinstance(raw_fits, bool) else None,
            amount=amount,
            maximum=maximum,
            unit="model_tokens" if amount is not None else "unknown",
            source=(
                "qwen3synth.invalid_measurement"
                if invalid_fields
                else "qwen3synth.measure_request"
            ),
            details={
                "model_id": self._selection.target_id,
                **({"invalid_fields": invalid_fields} if invalid_fields else {}),
            },
        )

    def _load_voice(self) -> Any:
        if self._voice is not None:
            return self._voice
        source = _selected_voice_source(self._selection)
        if source.kind == "prepared":
            resolved = resolve_prepared_voice(
                source, expected_format=QWEN3_QVOICE_FORMAT
            )
            try:
                self._voice = self._module.PreparedVoice.load(resolved.path)
            except Exception as exc:
                raise _translate_qwen_error(
                    exc,
                    self._selection,
                    operation="prepared voice load",
                    fallback_type=InvalidEngineVoiceError,
                    fallback_code="qwen3.prepared_voice_invalid",
                ) from exc
            return self._voice

        resolved_audio = resolve_reference_audio(
            source,
            cache_dir=self._selection.options.get("cache_dir"),
            offline=self._selection.offline,
            catalog_paths=self._selection.options.get("catalog_path"),
        )
        reference_text = _reference_text(self._selection, source)
        native_reference = self._module.ReferenceVoice(
            audio=resolved_audio.path,
            transcript=reference_text,
            language=_language(self._selection.language),
            source_id=hashlib.sha256(
                json.dumps(
                    self._voice_identity, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
        )
        try:
            self._voice = self._runtime.prepare_voice(native_reference)
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_qwen_error(
                exc, self._selection, operation="reference voice preparation"
            ) from exc
        return self._voice

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        native_request = self._module.SynthesisRequest(
            id=request.id,
            text=request.text,
            language=_language(request.language),
        )
        try:
            generation = self._module.GenerationConfig(
                temperature=self._selection.options.get("temperature", 0.9),
                top_k=self._selection.options.get("top_k", 50),
                repetition_penalty=self._selection.options.get(
                    "repetition_penalty", 1.05
                ),
                max_new_tokens=self._selection.options.get("max_new_tokens", 2048),
                seed=self._selection.options.get("seed"),
            )
            result = self._runtime.synthesize(
                native_request, voice=self._load_voice(), config=generation
            )
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_qwen_error(
                exc, self._selection, operation="synthesis", request=request
            ) from exc

        metadata = {
            "voice": self._voice_identity,
            "qwen3": {
                "target_id": self._selection.target_id,
                "model_revision": self._selection.metadata.get("source_revision"),
                "quality": self._selection.options.get("quality"),
                "provider": self._selection.options.get("provider"),
            },
        }
        rendered = RenderedSpeech(
            id=_value(result, "request_id"),
            audio=_value(result, "samples"),
            sample_rate=_value(result, "sample_rate"),
            metadata=metadata,
        )
        return validate_rendered_speech(
            request,
            rendered,
            engine=ENGINE_ID,
            engine_version=self._selection.metadata.get("engine_version"),
            target_id=self._selection.target_id,
        )


class Qwen3SynthEngineAdapter:
    id = ENGINE_ID
    package_name = PACKAGE_NAME

    def version(self) -> str | None:
        from .base import distribution_version

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
            module = _module()
            contract = module.request_api_contract()
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

        required_capabilities = {
            "supports_reference_voice": True,
            "supports_request_measurement": True,
        }
        mismatches = {
            name: {
                "expected": expected,
                "actual": _contract_value(contract, name),
            }
            for name, expected in required_capabilities.items()
            if _contract_value(contract, name) is not expected
        }
        supports_voice_design = (
            _contract_value(contract, "supports_voice_design", False) is True
        )
        if supports_voice_design:
            missing_symbols = tuple(
                name
                for name in ("VoiceDesignRequest",)
                if not callable(getattr(module, name, None))
            )
            runtime = getattr(module, "Qwen3Runtime", None)
            design_voice = getattr(runtime, "design_voice", None)
            missing_methods = (
                ("Qwen3Runtime.design_voice",) if not callable(design_voice) else ()
            )
            if missing_symbols or missing_methods:
                return replace(
                    probe,
                    compatible=False,
                    status="api_incompatible",
                    failed_stage="voice_design_capability",
                    failed_symbol=(missing_symbols or missing_methods)[0],
                    missing_symbols=(*probe.missing_symbols, *missing_symbols),
                    missing_methods=(*probe.missing_methods, *missing_methods),
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

        return replace(
            probe,
            details={
                **dict(probe.details),
                **required_capabilities,
                "supports_voice_design": supports_voice_design,
            },
        )

    def compatible_api(self) -> bool:
        return self.probe_api().compatible

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            voice_binding_namespace="qwen3",
            voice_binding_scope="target",
            option_names=QWEN3_OPTION_NAMES,
            supports_named_voices=False,
            supports_reference_voice=True,
            supports_speakers=False,
            supports_pronunciation_overrides=False,
            supports_linguistic_tokens=False,
            supports_whole_request_phonemes=False,
            supports_request_measurement=True,
            supports_qualities=True,
        )

    def renderprofile_fragment(
        self, selection: EngineSelection
    ) -> RenderProfileFragment:
        return make_renderprofile_fragment(
            self,
            selection,
            frontend=self.package_name,
            pronunciation="unsupported",
        )

    def discover(self, request: CatalogRequest) -> tuple[SynthesisTarget, ...]:
        module = _module()
        bundles = module.discover_bundles(
            offline=request.offline,
            refresh=request.refresh,
        )
        targets = tuple(_target_from_bundle(bundle) for bundle in bundles)
        if request.language is None:
            return targets
        requested = _language(request.language)
        return tuple(
            target
            for target in targets
            if not target.languages
            or requested in {_language(language) for language in target.languages}
        )

    def resolve(
        self, request: EngineRequest
    ) -> tuple[EngineSelection, tuple[Any, ...]]:
        target_id = request.target_id
        if not isinstance(target_id, str) or not target_id.strip():
            raise EngineSelectionError(
                "A Qwen3Synth target id is required.",
                engine=self.id,
                code="qwen3.target_not_found",
            )

        targets = self.discover(
            CatalogRequest(
                engine=self.id,
                offline=request.offline,
                refresh=request.refresh,
            )
        )
        target = _find_target(targets, target_id)
        if target is None:
            raise EngineSelectionError(
                f"Qwen3Synth target {target_id!r} was not found.",
                engine=self.id,
                target_id=target_id,
                code="qwen3.target_not_found",
            )
        if target.metadata.get("model_type") != "base":
            raise EngineSelectionError(
                "Qwen3 VoiceDesign targets create reference assets; use `voicerender voice design` instead of a render profile.",
                engine=self.id,
                target_id=target.id,
                code="qwen3.voice_design_target_not_renderable",
            )
        if request.voice is not None:
            raise EngineSelectionError(
                "Qwen3Synth does not support named voices; provide voice_source.",
                engine=self.id,
                target_id=target.id,
                code="qwen3.reference_required",
            )
        if request.speaker is not None:
            raise EngineSelectionError(
                "Qwen3Synth does not support speaker selection.",
                engine=self.id,
                target_id=target.id,
                code="qwen3.speaker_unsupported",
            )

        options = dict(request.options)
        engine_options = dict(request.engine_options)
        unknown = sorted(set(engine_options) - QWEN3_OPTION_NAMES)
        if unknown:
            raise EngineSelectionError(
                f"Unsupported Qwen3Synth option(s): {', '.join(unknown)}.",
                engine=self.id,
                target_id=target.id,
                code="qwen3.invalid_option",
                details={"options": unknown},
            )
        options.update(engine_options)

        language = normalize_locale_tag(request.language or "en-US")
        engine_language = _language(language)
        if target.languages and engine_language not in {
            _language(item) for item in target.languages
        }:
            raise EngineSelectionError(
                f"Qwen3Synth target {target.id!r} does not support {language!r}.",
                engine=self.id,
                target_id=target.id,
                code="qwen3.language_incompatible",
            )

        quality = options.get("quality", "int4")
        if not isinstance(quality, str) or (
            target.qualities and quality not in target.qualities
        ):
            raise EngineSelectionError(
                f"Qwen3Synth target {target.id!r} does not support quality {quality!r}.",
                engine=self.id,
                target_id=target.id,
                code="qwen3.quality_incompatible",
            )
        provider = options.get("provider", "cpu")
        if not isinstance(provider, str) or not provider.strip():
            raise EngineSelectionError(
                "Qwen3Synth provider must be a non-empty string.",
                engine=self.id,
                target_id=target.id,
                code="qwen3.invalid_option",
            )

        source_value = options.get("voice_source")
        if not isinstance(source_value, Mapping):
            raise EngineSelectionError(
                "Qwen3Synth requires a pinned reference or prepared voice_source.",
                engine=self.id,
                target_id=target.id,
                code="qwen3.reference_required",
            )
        try:
            source = normalize_voice_source(source_value, allow_named=False)
        except EngineError as exc:
            raise EngineSelectionError(
                str(exc),
                engine=self.id,
                target_id=target.id,
                code=getattr(exc, "code", "qwen3.reference_invalid"),
                details=getattr(exc, "details", {}),
            ) from exc
        if source.kind == "prepared":
            if source.format != QWEN3_QVOICE_FORMAT:
                raise EngineSelectionError(
                    f"Qwen3Synth requires prepared format {QWEN3_QVOICE_FORMAT!r}.",
                    engine=self.id,
                    target_id=target.id,
                    code="qwen3.prepared_voice_invalid",
                    details={"format": source.format},
                )
            if options.get("reference_text") is not None:
                raise EngineSelectionError(
                    "reference_text applies only to reference voice sources.",
                    engine=self.id,
                    target_id=target.id,
                    code="qwen3.invalid_option",
                )
        else:
            reference_text = options.get("reference_text")
            if reference_text is None and source.origin is not None:
                reference_text = source.origin.get("reference_text")
            if not isinstance(reference_text, str) or not reference_text.strip():
                raise EngineSelectionError(
                    "Qwen3Synth reference voices require an exact reference transcript.",
                    engine=self.id,
                    target_id=target.id,
                    code="qwen3.reference_text_required",
                )
            options["reference_text"] = reference_text

        module = _module()
        try:
            generation = module.GenerationConfig(
                temperature=options.get("temperature", 0.9),
                top_k=options.get("top_k", 50),
                repetition_penalty=options.get("repetition_penalty", 1.05),
                max_new_tokens=options.get("max_new_tokens", 2048),
                seed=options.get("seed"),
            )
        except (TypeError, ValueError) as exc:
            raise EngineSelectionError(
                f"Invalid Qwen3Synth generation options: {exc}",
                engine=self.id,
                target_id=target.id,
                code="qwen3.generation_config_invalid",
            ) from exc

        selected_options = {
            **options,
            "voice_source": source.as_mapping(),
            "quality": quality,
            "provider": provider,
            "temperature": generation.temperature,
            "top_k": generation.top_k,
            "repetition_penalty": generation.repetition_penalty,
            "max_new_tokens": generation.max_new_tokens,
        }
        if generation.seed is None:
            selected_options.pop("seed", None)
        else:
            selected_options["seed"] = generation.seed
        metadata = {
            "model_type": target.metadata.get("model_type"),
            "model_id": target.metadata.get("model_id"),
            "layout": target.metadata.get("layout"),
            "source_revision": target.metadata.get("source_revision"),
            "runtime_available": target.runtime_available,
            "sample_rate": target.sample_rate,
            "precisions": target.qualities,
            "engine_version": getattr(module, "__version__", self.version()),
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
        source_value = selection.options.get("voice_source")
        if not isinstance(source_value, Mapping):
            return (
                EngineDiagnostic(
                    code="qwen3.reference_required",
                    severity="error",
                    message="Qwen3Synth requires a pinned reference or prepared voice_source.",
                    field="render.target.voice",
                ),
            )
        try:
            source = normalize_voice_source(source_value, allow_named=False)
        except EngineError as exc:
            return (
                EngineDiagnostic(
                    code=getattr(exc, "code", "qwen3.reference_invalid"),
                    severity="error",
                    message=str(exc),
                    field="render.target.voice",
                    details=getattr(exc, "details", {}),
                ),
            )

        diagnostics = list(
            validate_voice_source(
                source,
                offline=selection.offline,
                catalog_paths=selection.options.get("catalog_path"),
            )
        )
        if diagnostics:
            return tuple(diagnostics)
        if source.kind != "prepared":
            return tuple(diagnostics)
        if source.format != QWEN3_QVOICE_FORMAT:
            diagnostics.append(
                EngineDiagnostic(
                    code="qwen3.prepared_voice_invalid",
                    severity="error",
                    message=f"Qwen3Synth requires prepared format {QWEN3_QVOICE_FORMAT!r}.",
                    field="render.target.voice",
                    details={"format": source.format},
                )
            )
            return tuple(diagnostics)

        try:
            resolved = resolve_prepared_voice(
                source, expected_format=QWEN3_QVOICE_FORMAT
            )
            profile = _module().inspect_voice_profile(resolved.path)
        except EngineError as exc:
            diagnostics.append(
                EngineDiagnostic(
                    code=getattr(exc, "code", "qwen3.prepared_voice_invalid"),
                    severity="error",
                    message=str(exc),
                    field="render.target.voice",
                    details=getattr(exc, "details", {}),
                )
            )
            return tuple(diagnostics)
        except Exception as exc:  # noqa: BLE001 - inspection failures become source diagnostics.
            error = _translate_qwen_error(
                exc,
                selection,
                operation="prepared voice inspection",
                fallback_type=InvalidEngineVoiceError,
                fallback_code="qwen3.prepared_voice_invalid",
            )
            diagnostics.append(
                EngineDiagnostic(
                    code=error.code,
                    severity="error",
                    message=str(error),
                    field="render.target.voice",
                    details=error.details,
                )
            )
            return tuple(diagnostics)

        expected_model_id = selection.metadata.get("model_id")
        expected_revision = selection.metadata.get("source_revision")
        expected_layout = selection.metadata.get("layout")
        mismatches: dict[str, dict[str, Any]] = {}
        for name, declared, actual in (
            ("model_id", source.model_id, profile.model_id),
            ("model_revision", source.model_revision, profile.model_revision),
            ("layout", source.layout, profile.layout),
            ("model_id", expected_model_id, profile.model_id),
            ("model_revision", expected_revision, profile.model_revision),
            ("layout", expected_layout, profile.layout),
        ):
            if declared is not None and actual != declared:
                mismatches[name] = {"expected": declared, "actual": actual}
        if mismatches:
            diagnostics.append(
                EngineDiagnostic(
                    code="qwen3.profile_incompatible",
                    severity="error",
                    message="Prepared Qwen voice metadata is incompatible with the selected Base target.",
                    field="render.target.voice",
                    details={"mismatches": mismatches},
                )
            )
        return tuple(diagnostics)

    def canonical_synthesis_identity(
        self, selection: EngineSelection
    ) -> Mapping[str, Any]:
        module = _module()
        runtime_identity = module.runtime_identity()
        options = selection.options
        return {
            "engine": self.id,
            "engine_version": getattr(module, "__version__", self.version()),
            "runtime_identity": (
                dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {}
            ),
            "target_id": selection.target_id,
            "model_revision": selection.metadata.get("source_revision"),
            "quality": options.get("quality"),
            "provider": options.get("provider"),
            "language": _language(selection.language),
            "voice": _voice_identity(selection),
            "generation": {
                name: options.get(name)
                for name in (
                    "temperature",
                    "top_k",
                    "repetition_penalty",
                    "max_new_tokens",
                    "seed",
                )
            },
        }

    @contextmanager
    def open(self, selection: EngineSelection) -> Iterator[Qwen3SynthEngineSession]:
        module = _module()
        options = selection.options
        try:
            runtime = module.Qwen3Runtime.from_installed(
                selection.target_id,
                quality=options.get("quality", "int4"),
                provider=options.get("provider", "cpu"),
                catalog=options.get("catalog_path"),
                cache_dir=options.get("cache_dir"),
            )
        except EngineError:
            raise
        except Exception as exc:
            raise _translate_qwen_error(
                exc,
                selection,
                operation="runtime open",
                fallback_type=EngineUnavailableError,
                fallback_code="qwen3.backend_failed",
            ) from exc
        try:
            yield Qwen3SynthEngineSession(runtime, selection, module)
        finally:
            try:
                runtime.close()
            except EngineError:
                raise
            except Exception as exc:
                raise _translate_qwen_error(
                    exc, selection, operation="runtime close"
                ) from exc


__all__ = [
    "ENGINE_ID",
    "PACKAGE_NAME",
    "QWEN3_OPTION_NAMES",
    "Qwen3SynthEngineAdapter",
]
