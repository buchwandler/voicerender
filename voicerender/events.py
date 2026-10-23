"""Typed synchronous events emitted by high-level render workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .errors import RenderEventCallbackError
from .progress import EtaConfidence, RenderProgress

RenderScope = Literal["preflight", "render", "book"]


class RenderEventKind(str, Enum):
    PREFLIGHT_STARTED = "preflight_started"
    PREFLIGHT_ITEM_STARTED = "preflight_item_started"
    PREFLIGHT_ITEM_FINISHED = "preflight_item_finished"
    PREFLIGHT_FINISHED = "preflight_finished"
    RENDER_STARTED = "render_started"
    BOOK_STARTED = "book_started"
    BOOK_COMPLETED = "book_completed"
    CHAPTER_STARTED = "chapter_started"
    CHAPTER_FINISHED = "chapter_finished"
    CHAPTER_REUSED = "chapter_reused"
    CHAPTER_SKIPPED = "chapter_skipped"
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
    error_message: str | None = None
    error_stage: str | None = None
    error_root_cause_type: str | None = None
    error_root_cause_message: str | None = None
    render_diagnostics: str | None = None
    rendered_count: int | None = None
    reused_count: int | None = None
    failed_count: int | None = None
    selected_count: int | None = None
    scope: RenderScope | None = None
    phase: str | None = None
    phase_completed_count: int | None = None
    phase_total_count: int | None = None
    phase_progress: float | None = None
    chapter_id: str | None = None
    chapter_index: int | None = None
    chapter_count: int | None = None
    segment_number: int | None = None
    segment_id: str | None = None
    completed_count: int | None = None
    total_count: int | None = None
    progress: float | None = None
    elapsed_seconds: float | None = None
    phase_elapsed_seconds: float | None = None
    synthesis_seconds: float | None = None
    eta_seconds: float | None = None
    eta_confidence: EtaConfidence = "unavailable"
    progress_snapshot: RenderProgress | None = None


RenderEventCallback = Callable[[RenderEvent], None]


def emit_render_event(
    callback: RenderEventCallback | None,
    event: RenderEvent,
) -> None:
    """Synchronously dispatch one event and normalize callback failures."""

    if callback is None:
        return
    try:
        callback(event)
    except RenderEventCallbackError:
        raise
    except Exception as exc:
        raise RenderEventCallbackError(event.kind.value, exc) from exc


__all__ = [
    "EtaConfidence",
    "RenderEvent",
    "RenderEventCallback",
    "RenderEventKind",
    "RenderProgress",
    "RenderScope",
    "emit_render_event",
]
