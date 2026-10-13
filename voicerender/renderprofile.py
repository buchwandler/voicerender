"""VoiceRender ownership and lifecycle helpers for shared RenderProfiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from renderplan import (
    PausePolicy,
    PlanningPolicy,
    RenderProfile,
    RoleProfile,
    RoleRequirements,
    load_profile,
    save_profile,
)

from .engines.base import EngineSelection, RenderProfileFragment
from .engines.profiles import reject_reference_binding
from .engines.registry import get_adapter, normalize_engine_id, resolve_voice
from .engines.selection import EngineRequest
from .errors import ProfileError

_DEFAULT_PAUSES = PausePolicy(
    mode="manual",
    durations_ms={"clause": 180, "sentence": 500, "paragraph": 800},
)


def create_render_profile(
    *,
    engine: str,
    target_id: str,
    language: str,
    role: str = "narrator",
    voice: str | None = None,
    speaker: str | int | None = None,
    options: Mapping[str, Any] | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> RenderProfile:
    """Discover a binding and seed a user-owned shared RenderProfile."""
    if not role.strip():
        raise ProfileError("Profile role name must not be empty")
    if not language.strip():
        raise ProfileError("Profile language must not be empty")
    request = EngineRequest(
        engine=engine,
        target_id=target_id,
        language=language,
        voice=voice,
        speaker=speaker,
        options=dict(options or {}),
        offline=offline,
        refresh=refresh,
    )
    _, fragment = _resolve_fragment(request)
    role_profile = RoleProfile(
        language=language,
        binding=dict(fragment.binding),
        requirements=RoleRequirements(
            token_fields=fragment.token_fields,
            frontend=fragment.frontend,
            pronunciation=fragment.pronunciation,
        ),
        planner_capacity=fragment.planner_capacity,
        renderer_capacity=fragment.renderer_capacity,
    )
    return RenderProfile(
        roles={role: role_profile},
        default_role=role,
        planning=PlanningPolicy(
            split="sentence", capacity_fallback=("clause", "token")
        ),
        pauses=_DEFAULT_PAUSES,
    ).with_identity()


def load_render_profile(profile: str | Path | RenderProfile) -> RenderProfile:
    """Load a shared RenderProfile, preserving RenderPlan's strict codec checks."""
    if isinstance(profile, RenderProfile):
        return profile
    try:
        return load_profile(profile)
    except Exception as exc:
        raise ProfileError(f"Cannot load RenderProfile {profile}: {exc}") from exc


def save_render_profile(profile: RenderProfile, path: str | Path) -> Path:
    """Persist a shared RenderProfile using the RenderPlan codec."""
    try:
        return cast(Path, save_profile(profile, path))
    except Exception as exc:
        raise ProfileError(f"Cannot write RenderProfile {path}: {exc}") from exc


def validate_render_profile(
    profile: str | Path | RenderProfile,
) -> RenderProfile:
    """Check each profile binding against the installed engine adapter/API."""
    loaded = load_render_profile(profile)
    for role_name, role in loaded.roles.items():
        binding = role.binding
        reject_reference_binding(binding)
        renderer = _required_string(
            binding.get("renderer"), f"roles.{role_name}.binding.renderer"
        )
        target_id = _required_string(
            binding.get("target_id"), f"roles.{role_name}.binding.target_id"
        )
        options = binding.get("options", {})
        if not isinstance(options, Mapping):
            raise ProfileError(f"Role {role_name!r} binding options must be a table")
        model_revision = binding.get("model_revision")
        if model_revision is not None and (
            not isinstance(model_revision, str) or not model_revision
        ):
            raise ProfileError(
                f"Role {role_name!r} model_revision must be a non-empty string"
            )
        selection, fragment = _resolve_fragment(
            EngineRequest(
                engine=renderer,
                target_id=target_id,
                language=role.language,
                voice=_optional_string(binding.get("voice"), "voice"),
                speaker=binding.get("speaker"),
                options=dict(options),
            )
        )
        if selection.target_id != target_id:
            raise ProfileError(
                f"Role {role_name!r} resolves to unexpected target {selection.target_id!r}"
            )
        expected_requirements = RoleRequirements(
            token_fields=fragment.token_fields,
            frontend=fragment.frontend,
            pronunciation=fragment.pronunciation,
        )
        if (
            role.requirements != expected_requirements
            or role.renderer_capacity != fragment.renderer_capacity
            or role.planner_capacity != fragment.planner_capacity
        ):
            raise ProfileError(
                f"Role {role_name!r} engine-derived requirements/capacity are stale; "
                "write and review a refreshed proposal"
            )
        proposed_metadata = fragment.binding.get("renderer_metadata", {})
        current_metadata = binding.get("renderer_metadata", {})
        if not isinstance(proposed_metadata, Mapping):
            proposed_metadata = {}
        if not isinstance(current_metadata, Mapping):
            current_metadata = {}
        if dict(current_metadata) != dict(proposed_metadata):
            raise ProfileError(
                f"Role {role_name!r} target metadata is stale; "
                "write and review a refreshed proposal"
            )
    return loaded


def refresh_render_profile(
    profile: str | Path | RenderProfile,
) -> RenderProfile:
    """Return a refreshed proposal without changing user-owned profile policy."""
    original = load_render_profile(profile)
    refreshed_roles: dict[str, RoleProfile] = {}
    for role_name, role in original.roles.items():
        binding = dict(role.binding)
        reject_reference_binding(binding)
        renderer = _required_string(
            binding.get("renderer"), f"roles.{role_name}.renderer"
        )
        target_id = _required_string(
            binding.get("target_id"), f"roles.{role_name}.target_id"
        )
        options = binding.get("options", {})
        if not isinstance(options, Mapping):
            raise ProfileError(f"Role {role_name!r} binding options must be a table")
        _, fragment = _resolve_fragment(
            EngineRequest(
                engine=renderer,
                target_id=target_id,
                language=role.language,
                voice=_optional_string(binding.get("voice"), "voice"),
                speaker=binding.get("speaker"),
                options=dict(options),
                refresh=True,
            )
        )
        new_binding = dict(binding)
        new_metadata = fragment.binding.get("renderer_metadata")
        if isinstance(new_metadata, Mapping):
            new_binding["renderer_metadata"] = dict(new_metadata)
        if binding.get("model_revision") is None and fragment.binding.get(
            "model_revision"
        ):
            new_binding["model_revision"] = fragment.binding["model_revision"]
        refreshed_roles[role_name] = RoleProfile(
            language=role.language,
            binding=new_binding,
            requirements=RoleRequirements(
                token_fields=fragment.token_fields,
                frontend=fragment.frontend,
                pronunciation=fragment.pronunciation,
            ),
            planner_capacity=fragment.planner_capacity,
            renderer_capacity=fragment.renderer_capacity,
        )
    return replace(original, roles=refreshed_roles, profile_id=None).with_identity()


def _resolve_fragment(
    request: EngineRequest,
) -> tuple[EngineSelection, RenderProfileFragment]:
    if not request.engine:
        raise ProfileError("A renderer engine is required")
    adapter = get_adapter(request.engine)
    probe = adapter.probe_api()
    if not probe.compatible:
        raise ProfileError(f"Engine {adapter.id!r} API is incompatible: {probe.status}")
    selection, diagnostics = resolve_voice(request)
    errors = [
        item for item in diagnostics if getattr(item, "severity", None) == "error"
    ]
    if errors:
        details = "; ".join(f"{item.code}: {item.message}" for item in errors)
        raise ProfileError(f"Engine binding is invalid: {details}")
    if normalize_engine_id(selection.engine) != normalize_engine_id(adapter.id):
        raise ProfileError("Engine adapter resolved an inconsistent renderer binding")
    selection = _with_target_metadata(adapter, selection)
    try:
        fragment = adapter.renderprofile_fragment(selection)
    except (ProfileError, ValueError) as exc:
        raise ProfileError(
            f"Cannot create engine RenderProfile fragment: {exc}"
        ) from exc
    if not isinstance(fragment, RenderProfileFragment):
        raise ProfileError(
            f"Engine {adapter.id!r} returned an invalid RenderProfile fragment"
        )
    reject_reference_binding(fragment.binding)
    if fragment.binding.get("renderer") != adapter.id:
        raise ProfileError("RenderProfile fragment renderer does not match its adapter")
    if fragment.binding.get("target_id") != selection.target_id:
        raise ProfileError(
            "RenderProfile fragment target_id does not match its selection"
        )
    return selection, fragment


def _with_target_metadata(adapter: Any, selection: EngineSelection) -> EngineSelection:
    metadata_reader = getattr(adapter, "target_metadata", None)
    if not callable(metadata_reader):
        return selection
    try:
        target_metadata = metadata_reader(selection)
    except Exception as exc:
        raise ProfileError(f"Cannot discover target metadata: {exc}") from exc
    if not isinstance(target_metadata, Mapping):
        return selection
    return replace(
        selection,
        metadata={**dict(selection.metadata), **dict(target_metadata)},
    )


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


__all__ = [
    "create_render_profile",
    "load_render_profile",
    "refresh_render_profile",
    "save_render_profile",
    "validate_render_profile",
]
