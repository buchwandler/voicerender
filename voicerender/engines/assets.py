"""Optional engine-owned model and asset management contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    engine: str
    target_id: str
    display_name: str
    installed: bool
    status: str
    languages: tuple[str, ...] = ()
    qualities: tuple[str, ...] = ()
    installed_qualities: tuple[str, ...] = ()
    source: str | None = None
    revision: str | None = None
    path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelManagementCapabilities:
    can_list: bool = True
    can_install: bool = False
    can_remove: bool = False
    can_verify: bool = False
    supports_quality: bool = False
    supports_source: bool = False


@dataclass(frozen=True, slots=True)
class ModelInstallRequest:
    engine: str
    target_id: str
    quality: str | None = None
    source: str | None = None
    cache_dir: Path | None = None
    force: bool = False


@dataclass(frozen=True, slots=True)
class ModelRemoveRequest:
    engine: str
    target_id: str
    quality: str | None = None
    cache_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class ModelInstallResult:
    engine: str
    target_id: str
    status: str
    changed: bool
    diagnostics: tuple[str, ...] = ()
    artifact: ModelArtifact | None = None


@dataclass(frozen=True, slots=True)
class ModelRemoveResult:
    engine: str
    target_id: str
    status: str
    changed: bool
    diagnostics: tuple[str, ...] = ()
    artifact: ModelArtifact | None = None


@dataclass(frozen=True, slots=True)
class ModelVerifyResult:
    engine: str
    target_id: str
    status: str
    verified: bool
    diagnostics: tuple[str, ...] = ()
    artifact: ModelArtifact | None = None


class EngineAssetManager(Protocol):
    """Optional public backend surface for managing engine model assets."""

    def capabilities(self) -> ModelManagementCapabilities: ...

    def list_models(
        self,
        *,
        language: str | None = None,
        offline: bool = False,
        refresh: bool = False,
    ) -> tuple[ModelArtifact, ...]: ...

    def inspect_model(
        self, target_id: str, *, quality: str | None = None
    ) -> ModelArtifact: ...

    def install_model(self, request: ModelInstallRequest) -> ModelInstallResult: ...

    def remove_model(self, request: ModelRemoveRequest) -> ModelRemoveResult: ...

    def verify_model(
        self,
        target_id: str,
        *,
        quality: str | None = None,
        cache_dir: Path | None = None,
    ) -> ModelVerifyResult: ...


__all__ = [
    "EngineAssetManager",
    "ModelArtifact",
    "ModelInstallRequest",
    "ModelInstallResult",
    "ModelManagementCapabilities",
    "ModelRemoveRequest",
    "ModelRemoveResult",
    "ModelVerifyResult",
]
