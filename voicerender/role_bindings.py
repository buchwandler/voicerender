"""Shared role binding display and canonical voice-identity helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from renderplan import RoleProfile

from .errors import EngineError
from .voice_sources import normalize_voice_source, voice_source_identity


@dataclass(frozen=True, slots=True)
class RoleVoiceIdentity:
    """Stable synthesis identity for a role, excluding delivery settings."""

    renderer: str
    target_id: str
    voice: str | None
    speaker: str | int | None
    source: Mapping[str, str] | None


@dataclass(frozen=True, slots=True)
class DuplicateVoiceGroup:
    """Required logical roles that resolve to the same canonical voice."""

    roles: tuple[str, ...]
    identity: RoleVoiceIdentity


def role_voice_selector(role: RoleProfile) -> str:
    """Return the established human selector used in profile role tables."""
    binding = role.binding
    voice = binding.get("voice")
    if voice is not None:
        return _display_value(voice)
    speaker = binding.get("speaker")
    if speaker is not None:
        return f"speaker:{_display_value(speaker)}"
    source = binding.get("voice_source")
    options = binding.get("options")
    if source is None and isinstance(options, Mapping):
        source = options.get("voice_source")
    if isinstance(source, Mapping):
        kind = source.get("kind") or source.get("type")
        return f"source:{_display_value(kind)}" if kind else "source"
    if source is not None:
        return "source"
    return "unknown"


def role_voice_identity(role: RoleProfile) -> RoleVoiceIdentity:
    """Normalize the role's canonical voice identity without engine probing."""
    binding = role.binding
    renderer = _required_binding_string(binding.get("renderer"), "renderer")
    target_id = _required_binding_string(binding.get("target_id"), "target_id")

    voice_value = binding.get("voice")
    if voice_value is not None and (
        not isinstance(voice_value, str) or not voice_value
    ):
        raise ValueError("binding.voice must be a non-empty string when present")
    speaker_value = binding.get("speaker")
    if speaker_value is not None and (
        isinstance(speaker_value, bool)
        or not isinstance(speaker_value, (str, int))
        or isinstance(speaker_value, str)
        and not speaker_value
    ):
        raise ValueError(
            "binding.speaker must be a non-empty string or integer when present"
        )

    source_value = binding.get("voice_source")
    options = binding.get("options")
    if source_value is None and isinstance(options, Mapping):
        source_value = options.get("voice_source")
    source_identity: Mapping[str, str] | None = None
    if source_value is not None:
        if not isinstance(source_value, Mapping):
            raise ValueError("binding.voice_source must be a table")
        try:
            normalized = normalize_voice_source(source_value, allow_named=False)
            source_identity = MappingProxyType(dict(voice_source_identity(normalized)))
        except (EngineError, ValueError, TypeError) as exc:
            raise ValueError(
                f"binding.voice_source cannot form a canonical identity: {exc}"
            ) from exc

    return RoleVoiceIdentity(
        renderer=renderer,
        target_id=target_id,
        voice=voice_value,
        speaker=speaker_value,
        source=source_identity,
    )


def duplicate_voice_groups(
    identities: Iterable[tuple[str, RoleVoiceIdentity]],
) -> tuple[DuplicateVoiceGroup, ...]:
    """Group duplicate identities while preserving first-required-role order."""
    groups: dict[tuple[Any, ...], tuple[RoleVoiceIdentity, list[str]]] = {}
    for role, identity in identities:
        key = _identity_key(identity)
        if key not in groups:
            groups[key] = (identity, [])
        groups[key][1].append(role)
    return tuple(
        DuplicateVoiceGroup(tuple(roles), identity)
        for identity, roles in groups.values()
        if len(roles) > 1
    )


def duplicate_roles_by_role(
    groups: Iterable[DuplicateVoiceGroup],
) -> Mapping[str, tuple[str, ...]]:
    """Map each duplicated role to the other roles in its duplicate group."""
    result: dict[str, tuple[str, ...]] = {}
    for group in groups:
        for role in group.roles:
            result[role] = tuple(other for other in group.roles if other != role)
    return MappingProxyType(result)


def _identity_key(identity: RoleVoiceIdentity) -> tuple[Any, ...]:
    source_key = (
        tuple(sorted(identity.source.items())) if identity.source is not None else None
    )
    return (
        identity.renderer,
        identity.target_id,
        identity.voice,
        identity.speaker,
        source_key,
    )


def _required_binding_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"binding.{field} must be a non-empty string")
    return value


def _display_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


__all__ = [
    "DuplicateVoiceGroup",
    "RoleVoiceIdentity",
    "duplicate_roles_by_role",
    "duplicate_voice_groups",
    "role_voice_identity",
    "role_voice_selector",
]
