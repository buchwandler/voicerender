"""Shared helpers for engine-owned RenderProfile facts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from renderplan import CapacityConstraint

from ..errors import ProfileError
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

    _reject_reference_binding(binding)
    return RenderProfileFragment(
        binding=binding,
        token_fields=token_fields,
        frontend=frontend or str(metadata.get("frontend") or adapter.id),
        pronunciation=pronunciation or "renderer",
        renderer_capacity=renderer_capacity,
        # These backend limits count frontend tokens and are not portable planner limits.
        planner_capacity=None,
    )


def reject_reference_binding(binding: Mapping[str, Any]) -> None:
    """Reject bindings whose reference audio depends on an external profile path."""
    _reject_reference_binding(binding)


def _reject_reference_binding(binding: Mapping[str, Any]) -> None:
    """Reject machine-local paths and validate portable managed-reference shape."""
    options = binding.get("options", {})
    options = options if isinstance(options, Mapping) else {}
    source = binding.get("voice_source", options.get("voice_source"))
    if (
        "reference_audio" in binding
        or "reference_audio" in options
        or isinstance(source, Mapping)
        and source.get("kind") == "reference"
    ):
        raise ProfileError(
            "Reference-audio bindings are not supported in RenderPlan profiles; "
            "use a named voice or a Voicefolio managed reference"
        )
    if isinstance(source, Mapping) and source.get("kind") == "managed_reference":
        resolver = source.get("resolver")
        ref = source.get("value")
        legacy_voicefolio_ref = (
            resolver is None
            and isinstance(ref, str)
            and ref.startswith("kyutai-tts-voices:")
        )
        if resolver != "voicefolio" and not legacy_voicefolio_ref:
            raise ProfileError(
                "Managed reference bindings require resolver='voicefolio'"
            )
        for name in (
            "value",
            "sha256",
            "source_revision",
            "source_repository",
            "source_path",
        ):
            value = source.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ProfileError(
                    f"Voicefolio managed-reference binding requires non-empty {name}"
                )


__all__ = ["make_renderprofile_fragment", "reject_reference_binding"]
