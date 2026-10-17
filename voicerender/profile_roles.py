"""Public APIs for inspecting and mutating shared RenderProfile roles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from renderplan import RenderProfile, RoleProfile

from .errors import ProfileError
from .renderprofile import create_render_profile, load_render_profile


def list_profile_roles(
    profile: str | Path | RenderProfile,
) -> Mapping[str, RoleProfile]:
    """Return the profile's role bindings without changing their semantics."""
    return load_render_profile(profile).roles


def set_profile_role(
    profile: str | Path | RenderProfile,
    role: str,
    *,
    engine: str,
    target_id: str,
    language: str,
    voice: str | None = None,
    speaker: str | int | None = None,
    voice_source: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    offline: bool = False,
    refresh: bool = False,
    make_default: bool = False,
) -> RenderProfile:
    """Add or replace a role using freshly resolved engine-derived facts."""
    _validate_role_name(role)
    original = load_render_profile(profile)
    proposed = create_render_profile(
        engine=engine,
        target_id=target_id,
        language=language,
        role=role,
        voice=voice,
        speaker=speaker,
        voice_source=voice_source,
        options=options,
        offline=offline,
        refresh=refresh,
    )
    roles = dict(original.roles)
    roles[role] = proposed.roles[role]
    default_role = role if make_default else original.default_role
    return replace(
        original,
        roles=roles,
        default_role=default_role,
        profile_id=None,
    ).with_identity()


def remove_profile_role(
    profile: str | Path | RenderProfile,
    role: str,
    *,
    new_default_role: str | None = None,
) -> RenderProfile:
    """Remove a role, requiring an explicit replacement for the current default."""
    _validate_role_name(role)
    original = load_render_profile(profile)
    if role not in original.roles:
        raise ProfileError(f"RenderProfile has no role {role!r}")
    if len(original.roles) == 1:
        raise ProfileError("Cannot remove the final RenderProfile role")
    if role == original.default_role and new_default_role is None:
        raise ProfileError("Removing the default role requires --new-default-role")

    roles = dict(original.roles)
    del roles[role]
    default_role = original.default_role
    if new_default_role is not None:
        _validate_role_name(new_default_role)
        if new_default_role not in roles:
            raise ProfileError(
                f"New default role {new_default_role!r} is not present after removal"
            )
        default_role = new_default_role
    return replace(
        original,
        roles=roles,
        default_role=default_role,
        profile_id=None,
    ).with_identity()


def set_default_profile_role(
    profile: str | Path | RenderProfile,
    role: str,
) -> RenderProfile:
    """Set the default role and recompute the profile's content identity."""
    _validate_role_name(role)
    original = load_render_profile(profile)
    if role not in original.roles:
        raise ProfileError(f"RenderProfile has no role {role!r}")
    if role == original.default_role:
        return original
    return replace(original, default_role=role, profile_id=None).with_identity()


def _validate_role_name(role: str) -> None:
    if not isinstance(role, str) or not role.strip():
        raise ProfileError("Profile role name must not be empty")


__all__ = [
    "list_profile_roles",
    "remove_profile_role",
    "set_default_profile_role",
    "set_profile_role",
]
