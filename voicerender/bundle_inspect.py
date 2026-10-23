"""Read-only composition of bundle cast requirements and render state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle_contract import BundleKind
from .bundle_roles import BundleRolesResult, inspect_bundle_roles
from .variants import BundleVariantsResult, list_bundle_variants


@dataclass(frozen=True, slots=True)
class BundleInspectResult:
    """Complete read-only bundle status from source roles through render state."""

    bundle: Path
    kind: BundleKind
    roles: BundleRolesResult
    variants: BundleVariantsResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "voicerender-bundle-inspect",
            "version": 1,
            "bundle": str(self.bundle),
            "kind": self.kind,
            "roles": self.roles.to_dict(),
            "variants": self.variants.to_dict(),
        }


def inspect_bundle(
    bundle: str | Path,
    *,
    profile: str | Path | None = None,
    variant: str | None = None,
    utterplan_root: str | Path | None = None,
    renderplan_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> BundleInspectResult:
    """Inspect role/cast state and compose it with the existing variant listing."""
    roles = inspect_bundle_roles(
        bundle,
        profile=profile,
        variant=variant,
        utterplan_root=utterplan_root,
        renderplan_root=renderplan_root,
    )
    variants = list_bundle_variants(
        bundle,
        renderplan_root=renderplan_root,
        output_root=output_root,
    )
    return BundleInspectResult(
        bundle=roles.bundle,
        kind=roles.kind,
        roles=roles,
        variants=variants,
    )


__all__ = ["BundleInspectResult", "inspect_bundle"]
