"""Typed synchronous events emitted by high-level render workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class RenderEventKind(str, Enum):
    PREFLIGHT_STARTED = "preflight_started"
    PREFLIGHT_FINISHED = "preflight_finished"
    ENGINE_OPEN_STARTED = "engine_open_started"
    ENGINE_OPEN_FINISHED = "engine_open_finished"
    SEGMENT_STARTED = "segment_started"
    SEGMENT_RENDERED = "segment_rendered"
    SEGMENT_REUSED = "segment_reused"
    SEGMENT_FAILED = "segment_failed"
    RENDER_COMPLETED = "render_completed"


@dataclass(frozen=True, slots=True)
class RenderEvent:
    """Immutable progress event; callbacks run synchronously on the render thread."""

    kind: RenderEventKind
    plan_id: str | None = None
    locator: str | None = None
    role: str | None = None
    engine: str | None = None
    status: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    rendered_count: int | None = None
    reused_count: int | None = None
    failed_count: int | None = None
    selected_count: int | None = None


RenderEventCallback = Callable[[RenderEvent], None]


__all__ = ["RenderEvent", "RenderEventCallback", "RenderEventKind"]
