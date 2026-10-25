"""Explicit ChatterboxSynth model management through its public runtime API."""

from __future__ import annotations

import importlib
from pathlib import Path

from ..errors import ProfileError
from .assets import (
    ModelArtifact,
    ModelInstallRequest,
    ModelInstallResult,
    ModelManagementCapabilities,
    ModelRemoveRequest,
    ModelRemoveResult,
    ModelVerifyResult,
)
from .catalog import SynthesisTarget
from .registry import discover_targets

ENGINE_ID = "chatterbox"
_CANONICAL_TARGETS = frozenset({"nano-v2", "turbo"})


class ChatterboxSynthAssetManager:
    """Manage Chatterbox bundles without opening an inference runtime."""

    def capabilities(self) -> ModelManagementCapabilities:
        return ModelManagementCapabilities(
            can_list=True,
            can_install=True,
            can_remove=False,
            can_verify=False,
            supports_quality=True,
            supports_source=False,
        )

    @staticmethod
    def _targets(
        *,
        language: str | None = None,
        offline: bool = False,
        refresh: bool = False,
    ) -> tuple[SynthesisTarget, ...]:
        targets = discover_targets(
            ENGINE_ID,
            language=language,
            offline=offline,
            refresh=refresh,
        )
        return tuple(target for target in targets if target.id in _CANONICAL_TARGETS)

    @staticmethod
    def _artifact(
        target: SynthesisTarget, *, quality: str | None = None
    ) -> ModelArtifact:
        profiles = tuple(target.qualities)
        metadata = dict(target.metadata)
        raw_installed = metadata.get("installed_profiles", ())
        installed_profiles = (
            tuple(
                dict.fromkeys(
                    profile
                    for profile in raw_installed
                    if isinstance(profile, str) and profile
                )
            )
            if isinstance(raw_installed, (list, tuple, set, frozenset))
            else ()
        )
        installed_qualities = tuple(
            profile for profile in profiles if profile in installed_profiles
        )
        installed = (
            quality in installed_qualities
            if quality is not None
            else bool(installed_qualities)
        )
        revision = metadata.get("source_revision")
        return ModelArtifact(
            engine=ENGINE_ID,
            target_id=target.id,
            display_name=target.display_name,
            installed=installed,
            status=target.status,
            languages=tuple(target.languages),
            qualities=profiles,
            installed_qualities=installed_qualities,
            revision=str(revision) if revision is not None else None,
            metadata=metadata,
        )

    def list_models(
        self,
        *,
        language: str | None = None,
        offline: bool = False,
        refresh: bool = False,
    ) -> tuple[ModelArtifact, ...]:
        return tuple(
            self._artifact(target)
            for target in self._targets(
                language=language,
                offline=offline,
                refresh=refresh,
            )
        )

    def _target(self, target_id: str, *, refresh: bool = False) -> SynthesisTarget:
        if target_id not in _CANONICAL_TARGETS:
            raise ProfileError(
                "Chatterbox target must be a canonical target: 'nano-v2' or 'turbo'"
            )
        target = next(
            (item for item in self._targets(refresh=refresh) if item.id == target_id),
            None,
        )
        if target is None:
            raise ProfileError(
                f"Chatterbox catalog has no canonical target {target_id!r}"
            )
        return target

    def inspect_model(
        self, target_id: str, *, quality: str | None = None
    ) -> ModelArtifact:
        target = self._target(target_id)
        if quality is not None and quality not in target.qualities:
            raise ProfileError(
                f"Chatterbox target {target_id!r} does not support quality {quality!r}"
            )
        return self._artifact(target, quality=quality)

    def install_model(self, request: ModelInstallRequest) -> ModelInstallResult:
        if request.source is not None:
            return ModelInstallResult(
                ENGINE_ID,
                request.target_id,
                "unsupported",
                False,
                (
                    "ChatterboxSynth does not support source selection for model install",
                ),
            )
        try:
            target = self._target(request.target_id)
            quality = (
                request.quality
                if request.quality is not None
                else target.metadata.get("default_profile")
            )
            if not isinstance(quality, str) or quality not in target.qualities:
                raise ProfileError(
                    f"Chatterbox target {target.id!r} does not support quality {quality!r}"
                )
            installed_before = quality in self._artifact(target).installed_qualities
            backend = importlib.import_module("chatterboxsynth")
            backend.ChatterboxRuntime.download(
                target.id,
                quality=quality,
                cache_dir=request.cache_dir,
                force_download=request.force,
            )
            installed_target = self._target(target.id, refresh=True)
            artifact = self._artifact(installed_target, quality=quality)
            if quality not in artifact.installed_qualities:
                return ModelInstallResult(
                    ENGINE_ID,
                    target.id,
                    "failed",
                    False,
                    (
                        (
                            f"ChatterboxSynth download completed but quality {quality!r} "
                            f"is not reported as installed for target {target.id!r}"
                        ),
                    ),
                    artifact,
                )
            return ModelInstallResult(
                ENGINE_ID,
                target.id,
                "installed",
                not installed_before or request.force,
                (f"Installed Chatterbox target {target.id!r} quality {quality!r}",),
                artifact,
            )
        except Exception as exc:  # noqa: BLE001 - backend errors become stable results.
            return ModelInstallResult(
                ENGINE_ID,
                request.target_id,
                "failed",
                False,
                (f"Chatterbox model installation failed: {exc}",),
            )

    def remove_model(self, request: ModelRemoveRequest) -> ModelRemoveResult:
        return ModelRemoveResult(
            ENGINE_ID,
            request.target_id,
            "unsupported",
            False,
            ("ChatterboxSynth does not support model removal",),
        )

    def verify_model(
        self,
        target_id: str,
        *,
        quality: str | None = None,
        cache_dir: Path | None = None,
    ) -> ModelVerifyResult:
        del quality, cache_dir
        return ModelVerifyResult(
            ENGINE_ID,
            target_id,
            "unsupported",
            False,
            ("ChatterboxSynth does not support model verification",),
        )


__all__ = ["ChatterboxSynthAssetManager"]
