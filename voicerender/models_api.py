"""Public model management API for VoiceRender engines."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .engines.assets import (
    EngineAssetManager,
    ModelArtifact,
    ModelInstallRequest,
    ModelInstallResult,
    ModelManagementCapabilities,
    ModelRemoveRequest,
    ModelRemoveResult,
    ModelVerifyResult,
)
from .engines.registry import discover_targets, get_asset_manager, normalize_engine_id
from .errors import ProfileError
from .jsonutil import json_value


def _unsupported(engine: str, operation: str, target_id: str) -> str:
    return (
        f"Engine {engine!r} does not support model {operation} for target {target_id!r}"
    )


def _capabilities(
    engine: str,
) -> tuple[EngineAssetManager | None, ModelManagementCapabilities]:
    manager = get_asset_manager(engine)
    if manager is None:
        return None, ModelManagementCapabilities()
    capabilities = manager.capabilities()
    return manager, capabilities


def _from_target(target: Any) -> ModelArtifact:
    metadata = dict(target.metadata)
    qualities = tuple(target.qualities)
    if "installed_profiles" in metadata:
        raw_installed = metadata["installed_profiles"]
        profiles = (
            {profile for profile in raw_installed if isinstance(profile, str)}
            if isinstance(raw_installed, (list, tuple, set, frozenset))
            else set()
        )
        installed_qualities = tuple(
            quality for quality in qualities if quality in profiles
        )
        installed = bool(installed_qualities)
    else:
        installed = bool(target.runtime_available)
        installed_qualities = qualities if installed else ()
    source = metadata.get("source")
    revision = metadata.get("revision", metadata.get("source_revision"))
    path = metadata.get("path")
    return ModelArtifact(
        engine=target.engine,
        target_id=target.id,
        display_name=target.display_name,
        installed=installed,
        status=target.status,
        languages=tuple(target.languages),
        qualities=qualities,
        installed_qualities=installed_qualities,
        source=str(source) if source is not None else None,
        revision=str(revision) if revision is not None else None,
        path=Path(path) if isinstance(path, (str, Path)) else None,
        metadata=metadata,
    )


def list_models(
    engine: str,
    *,
    language: str | None = None,
    installed: bool | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> tuple[ModelArtifact, ...]:
    """List model artifacts without opening a synthesis runtime."""
    engine_id = normalize_engine_id(engine)
    manager, capabilities = _capabilities(engine_id)
    if manager is not None and capabilities.can_list:
        models = tuple(
            manager.list_models(language=language, offline=offline, refresh=refresh)
        )
    else:
        models = tuple(
            _from_target(target)
            for target in discover_targets(
                engine_id, language=language, offline=offline, refresh=refresh
            )
        )
    selected = (
        model for model in models if installed is None or model.installed is installed
    )
    return tuple(sorted(selected, key=lambda model: (model.engine, model.target_id)))


def model_status(
    engine: str, target_id: str, *, quality: str | None = None
) -> ModelArtifact:
    """Inspect one target using its manager or side-effect-free target discovery."""
    engine_id = normalize_engine_id(engine)
    if not target_id.strip():
        raise ProfileError("A model target identifier is required")
    manager, capabilities = _capabilities(engine_id)
    if manager is not None and capabilities.can_list:
        model = manager.inspect_model(target_id, quality=quality)
        if model.engine != engine_id or model.target_id != target_id:
            raise ProfileError(
                f"Engine {engine_id!r} returned a mismatched model artifact"
            )
        return model
    for model in list_models(engine_id):
        if model.target_id == target_id:
            if quality is not None and quality not in model.qualities:
                raise ProfileError(
                    f"Engine {engine_id!r} target {target_id!r} does not provide quality {quality!r}"
                )
            if quality is not None and "installed_profiles" in model.metadata:
                return replace(
                    model,
                    installed=quality in model.installed_qualities,
                )
            return model
    raise ProfileError(f"Engine {engine_id!r} has no model target {target_id!r}")


def install_model(request: ModelInstallRequest) -> ModelInstallResult:
    """Install a model through the engine's optional public asset manager."""
    engine_id = normalize_engine_id(request.engine)
    manager, capabilities = _capabilities(engine_id)
    if manager is None or not capabilities.can_install:
        return ModelInstallResult(
            engine_id,
            request.target_id,
            "unsupported",
            False,
            (_unsupported(engine_id, "install", request.target_id),),
        )
    if request.quality is not None and not capabilities.supports_quality:
        return ModelInstallResult(
            engine_id,
            request.target_id,
            "unsupported",
            False,
            (
                f"Engine {engine_id!r} does not support quality selection for model install",
            ),
        )
    if request.source is not None and not capabilities.supports_source:
        return ModelInstallResult(
            engine_id,
            request.target_id,
            "unsupported",
            False,
            (
                f"Engine {engine_id!r} does not support source selection for model install",
            ),
        )
    normalized = ModelInstallRequest(
        engine_id,
        request.target_id,
        request.quality,
        request.source,
        request.cache_dir,
        request.force,
    )
    try:
        return manager.install_model(normalized)
    except Exception as exc:  # noqa: BLE001 - Backend failures become stable results.
        return ModelInstallResult(
            engine_id,
            request.target_id,
            "failed",
            False,
            (f"Engine {engine_id!r} model install failed: {exc}",),
        )


def remove_model(request: ModelRemoveRequest) -> ModelRemoveResult:
    """Remove a model through the engine's optional public asset manager."""
    engine_id = normalize_engine_id(request.engine)
    manager, capabilities = _capabilities(engine_id)
    if manager is None or not capabilities.can_remove:
        return ModelRemoveResult(
            engine_id,
            request.target_id,
            "unsupported",
            False,
            (_unsupported(engine_id, "remove", request.target_id),),
        )
    if request.quality is not None and not capabilities.supports_quality:
        return ModelRemoveResult(
            engine_id,
            request.target_id,
            "unsupported",
            False,
            (
                f"Engine {engine_id!r} does not support quality selection for model removal",
            ),
        )
    try:
        return manager.remove_model(
            ModelRemoveRequest(
                engine_id, request.target_id, request.quality, request.cache_dir
            )
        )
    except Exception as exc:  # noqa: BLE001 - Backend failures become stable results.
        return ModelRemoveResult(
            engine_id,
            request.target_id,
            "failed",
            False,
            (f"Engine {engine_id!r} model remove failed: {exc}",),
        )


def verify_model(
    engine: str,
    target_id: str,
    *,
    quality: str | None = None,
    cache_dir: Path | None = None,
) -> ModelVerifyResult:
    """Verify an installed model through the engine's optional public API."""
    engine_id = normalize_engine_id(engine)
    manager, capabilities = _capabilities(engine_id)
    if manager is None or not capabilities.can_verify:
        return ModelVerifyResult(
            engine_id,
            target_id,
            "unsupported",
            False,
            (_unsupported(engine_id, "verify", target_id),),
        )
    if quality is not None and not capabilities.supports_quality:
        return ModelVerifyResult(
            engine_id,
            target_id,
            "unsupported",
            False,
            (
                f"Engine {engine_id!r} does not support quality selection for model verify",
            ),
        )
    try:
        return manager.verify_model(target_id, quality=quality, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 - Backend failures become stable results.
        return ModelVerifyResult(
            engine_id,
            target_id,
            "failed",
            False,
            (f"Engine {engine_id!r} model verify failed: {exc}",),
        )


def model_artifact_to_dict(model: ModelArtifact) -> dict[str, Any]:
    """Return the stable JSON representation of a model artifact."""
    return {
        "engine": model.engine,
        "target_id": model.target_id,
        "display_name": model.display_name,
        "installed": model.installed,
        "status": model.status,
        "languages": list(model.languages),
        "qualities": list(model.qualities),
        "installed_qualities": list(model.installed_qualities),
        "source": model.source,
        "revision": model.revision,
        "path": model.path.as_posix() if model.path is not None else None,
        "metadata": json_value(
            model.metadata,
            label=f"model {model.engine!r} target {model.target_id!r} metadata",
        ),
    }


def model_result_to_dict(
    result: ModelInstallResult | ModelRemoveResult | ModelVerifyResult,
) -> dict[str, Any]:
    """Return a stable JSON representation shared by model operations."""
    value: dict[str, Any] = {
        "engine": result.engine,
        "target_id": result.target_id,
        "status": result.status,
        "diagnostics": list(result.diagnostics),
    }
    if isinstance(result, ModelVerifyResult):
        value["verified"] = result.verified
    else:
        value["changed"] = result.changed
    value["artifact"] = (
        model_artifact_to_dict(result.artifact) if result.artifact is not None else None
    )
    return value


__all__ = [
    "ModelArtifact",
    "ModelInstallRequest",
    "ModelInstallResult",
    "ModelManagementCapabilities",
    "ModelRemoveRequest",
    "ModelRemoveResult",
    "ModelVerifyResult",
    "install_model",
    "list_models",
    "model_artifact_to_dict",
    "model_result_to_dict",
    "model_status",
    "remove_model",
    "verify_model",
]
