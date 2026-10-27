"""Read-only composition of bundle cast requirements and render state."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tomlkit
from renderplan import load_profile_for_reidentification

from .bundle_contract import BundleKind
from .bundle_roles import BundleRolesError, BundleRolesResult, inspect_bundle_roles
from .variants import (
    BundleVariantsResult,
    list_bundle_variants,
    load_bundle_variant_profile,
)


@dataclass(frozen=True, slots=True)
class ProfileDifference:
    """One canonical, safely serialized authored/snapshot profile difference."""

    role: str
    field: str
    authored_value: Any
    variant_value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "field": self.field,
            "authored_value": self.authored_value,
            "variant_value": self.variant_value,
        }


@dataclass(frozen=True, slots=True)
class ProfileComparison:
    """Read-only comparison of an authored profile with a variant snapshot."""

    status: Literal["match", "different", "error"]
    authored_profile_path: Path
    authored_profile_id: str | None
    authored_computed_profile_id: str | None
    authored_profile_id_valid: bool | None
    authored_profile_id_stale: bool | None
    variant_profile_path: Path | None
    variant_profile_id: str | None
    identity_match: bool | None
    differences: tuple[ProfileDifference, ...] = ()
    recommended_action: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "authored_profile_path": str(self.authored_profile_path),
            "authored_profile_id": self.authored_profile_id,
            "authored_computed_profile_id": self.authored_computed_profile_id,
            "authored_profile_id_valid": self.authored_profile_id_valid,
            "authored_profile_id_stale": self.authored_profile_id_stale,
            "variant_profile_path": (
                str(self.variant_profile_path) if self.variant_profile_path else None
            ),
            "variant_profile_id": self.variant_profile_id,
            "identity_match": self.identity_match,
            "differences": [difference.to_dict() for difference in self.differences],
            "recommended_action": self.recommended_action,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class BundleInspectResult:
    """Complete read-only bundle status from source roles through render state."""

    bundle: Path
    kind: BundleKind
    roles: BundleRolesResult
    variants: BundleVariantsResult
    profile_comparison: ProfileComparison | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "voicerender-bundle-inspect",
            "version": 1,
            "bundle": str(self.bundle),
            "kind": self.kind,
            "roles": self.roles.to_dict(),
            "variants": self.variants.to_dict(),
            "profile_comparison": (
                self.profile_comparison.to_dict() if self.profile_comparison else None
            ),
        }


def inspect_bundle(
    bundle: str | Path,
    *,
    profile: str | Path | None = None,
    variant: str | None = None,
    compare_profile: str | Path | None = None,
    utterplan_root: str | Path | None = None,
    renderplan_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> BundleInspectResult:
    """Inspect role/cast state and compose it with existing variant listing."""
    if compare_profile is not None and variant is None:
        raise BundleRolesError(
            "--compare-profile requires --variant",
            code="bundle.roles_invalid",
        )
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
    comparison = (
        compare_bundle_profile(
            bundle,
            variant,
            compare_profile,
            renderplan_root=renderplan_root,
        )
        if compare_profile is not None and variant is not None
        else None
    )
    return BundleInspectResult(
        bundle=roles.bundle,
        kind=roles.kind,
        roles=roles,
        variants=variants,
        profile_comparison=comparison,
    )


def compare_bundle_profile(
    bundle: str | Path,
    variant: str,
    authored_profile: str | Path,
    *,
    renderplan_root: str | Path | None = None,
) -> ProfileComparison:
    """Compare an authored profile with a validated immutable variant snapshot."""
    authored_path = Path(authored_profile).expanduser().resolve(strict=False)
    selected = load_bundle_variant_profile(
        bundle,
        variant,
        renderplan_root=renderplan_root,
    )
    snapshot = selected.profile
    snapshot_id = snapshot.profile_id or snapshot.computed_id()
    try:
        document = tomlkit.parse(authored_path.read_text(encoding="utf-8"))
        stored_profile_id = document.get("profile_id")
        authored_id = stored_profile_id if isinstance(stored_profile_id, str) else None
        authored = load_profile_for_reidentification(authored_path)
        computed_id = authored.computed_id()
        differences = _profile_differences(authored, snapshot)
    except Exception as exc:  # noqa: BLE001 - inspection returns structured diagnostics.
        return ProfileComparison(
            status="error",
            authored_profile_path=authored_path,
            authored_profile_id=None,
            authored_computed_profile_id=None,
            authored_profile_id_valid=None,
            authored_profile_id_stale=None,
            variant_profile_path=selected.path,
            variant_profile_id=snapshot_id,
            identity_match=None,
            error_code=getattr(exc, "code", "profile.compare_invalid"),
            error_message=str(exc) or type(exc).__name__,
        )

    identity_match = authored_id == snapshot_id
    different = bool(differences) or not identity_match
    return ProfileComparison(
        status="different" if different else "match",
        authored_profile_path=authored_path,
        authored_profile_id=authored_id,
        authored_computed_profile_id=computed_id,
        authored_profile_id_valid=authored_id == computed_id,
        authored_profile_id_stale=authored_id is not None
        and authored_id != computed_id,
        variant_profile_path=selected.path,
        variant_profile_id=snapshot_id,
        identity_match=identity_match,
        differences=tuple(differences),
        recommended_action=(
            "Re-project a new RenderPlan variant from the authored profile with "
            "utterplan bundle render."
            if different
            else None
        ),
    )


def _profile_differences(authored: Any, variant: Any) -> list[ProfileDifference]:
    authored_roles = getattr(authored, "roles", {})
    variant_roles = getattr(variant, "roles", {})
    differences: list[ProfileDifference] = []
    for role_name in sorted(set(authored_roles) | set(variant_roles)):
        authored_role = authored_roles.get(role_name)
        variant_role = variant_roles.get(role_name)
        authored_data = _role_inspection_data(authored_role)
        variant_data = _role_inspection_data(variant_role)
        _append_mapping_differences(
            differences,
            role_name,
            "",
            authored_data,
            variant_data,
        )
    return differences


def _role_inspection_data(role: Any) -> dict[str, Any] | None:
    if role is None:
        return None
    binding = getattr(role, "binding", None)
    if not isinstance(binding, Mapping):
        binding = None
    return {
        "language": getattr(role, "language", None),
        "binding": _inspection_binding(binding) if binding is not None else None,
    }


def _append_mapping_differences(
    differences: list[ProfileDifference],
    role: str,
    prefix: str,
    authored: Any,
    variant: Any,
) -> None:
    if isinstance(authored, Mapping) and isinstance(variant, Mapping):
        for key in sorted(set(authored) | set(variant), key=str):
            field = f"{prefix}.{key}" if prefix else str(key)
            _append_mapping_differences(
                differences,
                role,
                field,
                authored.get(key),
                variant.get(key),
            )
        return
    if isinstance(authored, list) and isinstance(variant, list) and authored == variant:
        return
    if authored != variant:
        differences.append(
            ProfileDifference(
                role=role,
                field=prefix,
                authored_value=authored,
                variant_value=variant,
            )
        )


def _inspection_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    safe = _redact_transcript_fields(dict(binding))
    if not isinstance(safe, dict):
        return {}
    if isinstance(binding.get("voice_source"), Mapping):
        safe["voice_source"] = _inspection_voice_source(binding["voice_source"])
    options = binding.get("options")
    if isinstance(options, Mapping):
        options_safe = _redact_transcript_fields(dict(options))
        if isinstance(options_safe, dict):
            if isinstance(options.get("voice_source"), Mapping):
                options_safe["voice_source"] = _inspection_voice_source(
                    options["voice_source"]
                )
            safe["options"] = options_safe
    return safe


def _inspection_voice_source(source: Mapping[str, Any]) -> dict[str, Any]:
    kind = source.get("kind") or source.get("type")
    safe: dict[str, Any] = {"kind": kind} if isinstance(kind, str) else {}
    fields: tuple[str, ...]
    if kind == "managed_reference":
        ref = source.get("value", source.get("ref", source.get("reference")))
        if isinstance(ref, str):
            safe["ref"] = ref
        fields = (
            "resolver",
            "sha256",
            "source_format",
            "source_revision",
            "source_repository",
            "source_path",
            "license",
            "dataset",
            "variant",
        )
    elif kind == "prepared":
        fields = ("sha256", "format", "model_id", "model_revision", "layout")
    elif kind == "reference":
        fields = ("sha256",)
    else:
        fields = ()
    for name in fields:
        value = source.get(name)
        if value is not None:
            safe[name] = value
    return safe


def _redact_transcript_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for name, item in value.items():
            if name in {"reference_text", "transcript"} and isinstance(item, str):
                digest_name = (
                    "reference_text_sha256"
                    if name == "reference_text"
                    else "transcript_sha256"
                )
                safe[str(digest_name)] = hashlib.sha256(
                    item.encode("utf-8")
                ).hexdigest()
            else:
                safe[str(name)] = _redact_transcript_fields(item)
        return safe
    if isinstance(value, (list, tuple)):
        return [_redact_transcript_fields(item) for item in value]
    return value


__all__ = [
    "BundleInspectResult",
    "ProfileComparison",
    "ProfileDifference",
    "compare_bundle_profile",
    "inspect_bundle",
]
