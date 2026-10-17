"""Explicit Qwen VoiceDesign asset generation."""

from __future__ import annotations

import importlib
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engines.api_probe import EngineApiProbe
from .engines.qwen3synth import ENGINE_ID, PACKAGE_NAME, Qwen3SynthEngineAdapter
from .errors import (
    ArtifactError,
    EngineApiIncompatibleError,
    EngineBackendError,
    EngineError,
    EngineUnavailableError,
    InvalidEngineOptionError,
    UnsupportedSynthesisFeatureError,
)
from .storage import atomic_wav, file_hash, install_file_no_replace
from .voice_source_codec import save_voice_source
from .voice_sources import local_reference_source, normalize_voice_source

VOICE_DESIGN_TARGET_ID = "voice-design-1.7b"
VOICE_DESIGN_ORIGIN_SCHEMA = "voicerender.voice-origin.qwen3-design.v1"


@dataclass(frozen=True, slots=True)
class Qwen3VoiceDesignRecipe:
    prompt: str
    reference_text: str
    language: str = "en"
    target_id: str = VOICE_DESIGN_TARGET_ID
    quality: str = "int4"
    provider: str = "cpu"
    seed: int | None = None
    temperature: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        _validate_recipe(self)


@dataclass(frozen=True, slots=True)
class DesignedReference:
    path: Path
    sha256: str
    source: Mapping[str, Any]
    model_id: str
    runtime_identity: Mapping[str, Any]


def _invalid_recipe(recipe: Qwen3VoiceDesignRecipe, field: str, message: str) -> None:
    raise InvalidEngineOptionError(
        message,
        code="qwen3.voice_design_recipe_invalid",
        engine=ENGINE_ID,
        target_id=recipe.target_id,
        details={"field": field},
    )


def _validate_recipe(recipe: Qwen3VoiceDesignRecipe) -> None:
    for field in ("prompt", "reference_text", "language", "target_id", "provider"):
        value = getattr(recipe, field)
        if not isinstance(value, str) or not value.strip():
            _invalid_recipe(
                recipe, field, f"Qwen VoiceDesign requires a non-empty {field}."
            )
    if not isinstance(recipe.quality, str) or recipe.quality not in {"int4", "fp32"}:
        _invalid_recipe(
            recipe, "quality", "Qwen VoiceDesign quality must be 'int4' or 'fp32'."
        )
    if recipe.seed is not None and (
        isinstance(recipe.seed, bool) or not isinstance(recipe.seed, int)
    ):
        _invalid_recipe(recipe, "seed", "Qwen VoiceDesign seed must be an integer.")
    if (
        isinstance(recipe.temperature, bool)
        or not isinstance(recipe.temperature, (int, float))
        or not math.isfinite(float(recipe.temperature))
        or not 0 < recipe.temperature <= 5
    ):
        _invalid_recipe(
            recipe, "temperature", "Temperature must be finite and in (0, 5]."
        )
    if (
        isinstance(recipe.top_k, bool)
        or not isinstance(recipe.top_k, int)
        or recipe.top_k < 0
    ):
        _invalid_recipe(recipe, "top_k", "top_k must be a non-negative integer.")
    if (
        isinstance(recipe.repetition_penalty, bool)
        or not isinstance(recipe.repetition_penalty, (int, float))
        or not math.isfinite(float(recipe.repetition_penalty))
        or recipe.repetition_penalty <= 0
    ):
        _invalid_recipe(
            recipe,
            "repetition_penalty",
            "repetition_penalty must be finite and positive.",
        )
    if (
        isinstance(recipe.max_new_tokens, bool)
        or not isinstance(recipe.max_new_tokens, int)
        or not 1 <= recipe.max_new_tokens <= 8192
    ):
        _invalid_recipe(
            recipe, "max_new_tokens", "max_new_tokens must be between 1 and 8192."
        )


def _probe_voice_design() -> EngineApiProbe:
    probe = Qwen3SynthEngineAdapter().probe_api()
    if not probe.compatible:
        if probe.status == "package_missing":
            raise EngineUnavailableError(
                "Qwen3Synth is not installed; install the qwen3 extra to design a voice.",
                engine=ENGINE_ID,
                target_id=VOICE_DESIGN_TARGET_ID,
                engine_version=probe.module_version,
                native_error_type=probe.error_type,
                code="qwen3.package_missing",
            )
        raise EngineApiIncompatibleError(
            f"Qwen3Synth public API is incompatible: {probe.error_message or probe.status}",
            engine=ENGINE_ID,
            target_id=VOICE_DESIGN_TARGET_ID,
            engine_version=probe.module_version,
            native_error_type=probe.error_type,
            code="qwen3.api_incompatible",
            details={
                "status": probe.status,
                "missing_symbols": probe.missing_symbols,
                "missing_methods": probe.missing_methods,
            },
        )
    if probe.details.get("supports_voice_design") is not True:
        raise UnsupportedSynthesisFeatureError(
            "The installed Qwen3Synth API does not support VoiceDesign.",
            engine=ENGINE_ID,
            target_id=VOICE_DESIGN_TARGET_ID,
            engine_version=probe.module_version,
            code="qwen3.voice_design_unsupported",
        )
    return probe


def _translate_error(
    exc: Exception,
    *,
    operation: str,
    target_id: str,
    engine_version: str | None,
) -> EngineError:
    if isinstance(exc, EngineError):
        return exc
    native_name = type(exc).__name__
    if native_name == "ModelNotInstalledError":
        error_type: type[EngineError] = EngineUnavailableError
        code = "qwen3.model_not_installed"
    elif native_name == "CatalogError":
        error_type = EngineUnavailableError
        code = "qwen3.catalog_unavailable"
    elif native_name == "UnsupportedModelCapabilityError":
        error_type = UnsupportedSynthesisFeatureError
        code = "qwen3.capability_unsupported"
    else:
        error_type = EngineBackendError
        code = "qwen3.backend_failed"
    return error_type(
        f"Qwen3 VoiceDesign {operation} failed: {exc}",
        engine=ENGINE_ID,
        engine_version=engine_version,
        target_id=target_id,
        native_error_type=native_name,
        code=code,
    )


def _output_path(path: str | Path, *, suffix: str, force: bool, label: str) -> Path:
    target = Path(path).expanduser().absolute()
    if target.suffix.casefold() != suffix:
        raise ArtifactError(f"{label} must use the {suffix} extension")
    if target.exists() and target.is_dir():
        raise ArtifactError(f"{label} is a directory: {target}")
    if (target.exists() or target.is_symlink()) and not force:
        raise ArtifactError(
            f"{label} already exists; pass force=True to replace it",
            code="qwen3.voice_design_output_exists",
        )
    return target


def design_qwen3_reference(
    recipe: Qwen3VoiceDesignRecipe,
    output: str | Path,
    *,
    cache_dir: str | Path | None = None,
    catalog: str | Path | None = None,
    source_output: str | Path | None = None,
    force: bool = False,
) -> DesignedReference:
    """Generate one fixed reference WAV from an already-installed VoiceDesign model."""
    _validate_recipe(recipe)
    target = _output_path(
        output, suffix=".wav", force=force, label="VoiceDesign output"
    )
    sidecar = (
        _output_path(
            source_output, suffix=".toml", force=force, label="Voice source sidecar"
        )
        if source_output is not None
        else None
    )
    if sidecar == target:
        raise ArtifactError(
            "VoiceDesign WAV and source sidecar must use different paths"
        )

    probe = _probe_voice_design()
    module = importlib.import_module(PACKAGE_NAME)
    engine_version = probe.module_version
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".voicerender-voice-design-", suffix=".wav", dir=target.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    runtime: Any | None = None
    try:
        try:
            runtime = module.Qwen3Runtime.from_installed(
                recipe.target_id,
                quality=recipe.quality,
                provider=recipe.provider,
                catalog=catalog,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise _translate_error(
                exc,
                operation="runtime opening",
                target_id=recipe.target_id,
                engine_version=engine_version,
            ) from exc

        try:
            runtime_identity = module.runtime_identity()
            model_id = runtime.model_id
            model_revision = runtime.model_revision
            if not isinstance(runtime_identity, Mapping):
                raise TypeError("runtime_identity() did not return a mapping")
            if not isinstance(model_id, str) or not model_id.strip():
                raise TypeError("Qwen3Synth runtime did not report its model id")
            if not isinstance(model_revision, str) or not model_revision.strip():
                raise TypeError("Qwen3Synth runtime did not report its model revision")
            qwen_version = getattr(module, "__version__", None)
            onnxvoice_version = runtime_identity.get("onnxvoice_version")
            if not isinstance(qwen_version, str) or not qwen_version.strip():
                raise TypeError("Qwen3Synth did not report its version")
            if not isinstance(onnxvoice_version, str) or not onnxvoice_version.strip():
                raise TypeError(
                    "Qwen3Synth runtime identity did not report ONNXVoice version"
                )
            request = module.VoiceDesignRequest(
                id="voicerender-voice-design",
                text=recipe.reference_text,
                description=recipe.prompt,
                language=recipe.language,
            )
            generation = module.GenerationConfig(
                temperature=recipe.temperature,
                top_k=recipe.top_k,
                repetition_penalty=recipe.repetition_penalty,
                max_new_tokens=recipe.max_new_tokens,
                seed=recipe.seed,
            )
            result = runtime.design_voice(request, config=generation)
        except Exception as exc:
            raise _translate_error(
                exc,
                operation="synthesis",
                target_id=recipe.target_id,
                engine_version=engine_version,
            ) from exc

        atomic_wav(temporary_path, result.samples, result.sample_rate)
        digest = file_hash(temporary_path)
        if force:
            os.replace(temporary_path, target)
        else:
            try:
                install_file_no_replace(temporary_path, target)
            except FileExistsError as exc:
                raise ArtifactError(
                    f"VoiceDesign output appeared during generation: {target}",
                    code="qwen3.voice_design_output_exists",
                ) from exc
            temporary_path.unlink()

        source: dict[str, Any] = local_reference_source(target)
        source["origin"] = {
            "schema": VOICE_DESIGN_ORIGIN_SCHEMA,
            "generator": ENGINE_ID,
            "target_id": recipe.target_id,
            "model_revision": model_revision,
            "model_id": model_id,
            "qwen3synth_version": qwen_version,
            "onnxvoice_version": onnxvoice_version,
            "prompt": recipe.prompt,
            "reference_text": recipe.reference_text,
            "language": recipe.language,
            "quality": recipe.quality,
            "provider": recipe.provider,
            "temperature": recipe.temperature,
            "top_k": recipe.top_k,
            "repetition_penalty": recipe.repetition_penalty,
            "max_new_tokens": recipe.max_new_tokens,
            "seed": recipe.seed,
            "runtime_identity": dict(runtime_identity),
        }
        normalized_source = normalize_voice_source(source).as_mapping()
        if normalized_source["sha256"] != digest:
            raise ArtifactError(
                "VoiceDesign output hash changed while materializing its source"
            )
        if sidecar is not None:
            save_voice_source(normalized_source, sidecar)
        return DesignedReference(
            path=target,
            sha256=digest,
            source=normalized_source,
            model_id=model_id,
            runtime_identity=dict(runtime_identity),
        )
    finally:
        if runtime is not None:
            runtime.close()
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "VOICE_DESIGN_ORIGIN_SCHEMA",
    "VOICE_DESIGN_TARGET_ID",
    "DesignedReference",
    "Qwen3VoiceDesignRecipe",
    "design_qwen3_reference",
]
