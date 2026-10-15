"""Shared helpers for engine-owned RenderProfile facts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from renderplan import CapacityConstraint

from ..errors import EngineError, ProfileError
from ..voice_sources import normalize_voice_source, validate_voice_source
from .base import EngineSelection, RenderProfileFragment


class _AdapterIdentity(Protocol):
    id: str


def make_renderprofile_fragment(
    adapter: _AdapterIdentity,
    selection: EngineSelection,
    *,
    token_fields: tuple[str, ...] = (),
    frontend: str | None = None,
    pronunciation: str | None = None,
    capacity_key: str | None = None,
    capacity_metric: str | None = None,
) -> RenderProfileFragment:
    """Build a portable fragment from an adapter's target discovery facts."""
    if selection.engine != adapter.id:
        raise ProfileError(
            f"Cannot create a {adapter.id!r} profile fragment from "
            f"{selection.engine!r} selection"
        )
    metadata_reader = getattr(adapter, "target_metadata", None)
    raw_metadata = metadata_reader(selection) if callable(metadata_reader) else {}
    metadata = {
        **dict(selection.metadata),
        **(dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}),
    }

    maximum = metadata.get(capacity_key) if capacity_key else None
    renderer_capacity: CapacityConstraint | None = None
    if (
        capacity_metric is not None
        and isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and maximum > 0
    ):
        renderer_capacity = CapacityConstraint(
            metric=capacity_metric,
            maximum=maximum,
            guarantee="hard",
        )

    revision = selection.metadata.get("model_revision") or metadata.get(
        "model_revision"
    )
    source_revision = metadata.get("source_revision")
    renderer_metadata: dict[str, Any] = {
        "frontend": frontend or str(metadata.get("frontend") or adapter.id),
        "pronunciation": pronunciation or "renderer",
    }
    policy_revision = metadata.get("capacity_policy_revision") or metadata.get(
        "capacity_revision"
    )
    if isinstance(policy_revision, str) and policy_revision.strip():
        renderer_metadata["capacity_policy_revision"] = policy_revision.strip()
    binding: dict[str, Any] = {
        "renderer": adapter.id,
        "target_id": selection.target_id,
        "options": dict(selection.options),
        "renderer_metadata": renderer_metadata,
    }
    if selection.voice is not None:
        binding["voice"] = selection.voice
    if selection.speaker is not None:
        binding["speaker"] = selection.speaker
    if revision is not None:
        if not isinstance(revision, str) or not revision:
            raise ProfileError(
                "Engine model revision metadata must be a non-empty string"
            )
        binding["model_revision"] = revision
        renderer_metadata["model_revision"] = revision
    if source_revision is not None:
        if not isinstance(source_revision, str) or not source_revision:
            raise ProfileError(
                "Engine source revision metadata must be a non-empty string"
            )
        renderer_metadata["source_revision"] = source_revision
    if renderer_capacity is not None:
        renderer_metadata["capacity"] = renderer_capacity.to_dict()
        if capacity_key is not None:
            renderer_metadata[capacity_key] = renderer_capacity.maximum

    validate_voice_source_binding(binding)
    return RenderProfileFragment(
        binding=binding,
        token_fields=token_fields,
        frontend=frontend or str(metadata.get("frontend") or adapter.id),
        pronunciation=pronunciation or "renderer",
        renderer_capacity=renderer_capacity,
        # These backend limits count frontend tokens and are not portable planner limits.
        planner_capacity=None,
    )


def validate_voice_source_binding(binding: Mapping[str, Any]) -> None:
    """Validate the modern RenderPlan voice-source binding without fetching assets."""
    if not isinstance(binding, Mapping):
        raise ProfileError("RenderPlan binding must be a table")
    options = binding.get("options", {})
    if not isinstance(options, Mapping):
        raise ProfileError("RenderPlan binding options must be a table")
    if "reference_audio" in binding or "reference_audio" in options:
        raise ProfileError(
            "RenderPlan bindings must use options.voice_source, not reference_audio"
        )
    if "voice_source" in binding:
        raise ProfileError("voice_source must be located in binding.options")
    if "voice_source" not in options:
        return
    if binding.get("voice") is not None or binding.get("speaker") is not None:
        raise ProfileError("voice/speaker and voice_source cannot both be set")
    raw_source = options.get("voice_source")
    try:
        source = normalize_voice_source(raw_source)
    except (EngineError, ValueError) as exc:
        raise ProfileError(str(exc)) from exc
    if source.kind == "reference":
        diagnostics = validate_voice_source(source)
        if diagnostics:
            raise ProfileError(diagnostics[0].message)


def reject_reference_binding(binding: Mapping[str, Any]) -> None:
    """Deprecated compatibility wrapper for validate_voice_source_binding()."""
    validate_voice_source_binding(binding)


__all__ = [
    "make_renderprofile_fragment",
    "reject_reference_binding",
    "validate_voice_source_binding",
]
