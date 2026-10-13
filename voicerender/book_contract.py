"""Adapters from shared RenderPlan validation to book-facing errors."""

from __future__ import annotations

from pathlib import Path

from renderplan import (
    RenderPlan,
    RenderPlanSet,
    RenderPlanSetChapter,
    RenderPlanSetError,
    RenderProfile,
    load_renderplan_set,
    validate_renderplan_set,
    validate_renderplan_set_profile,
    validate_set_chapter_plan,
)

from .errors import VoiceRenderError


class BookRenderError(VoiceRenderError, ValueError):
    """A book workspace, variant, index, profile, or selector is invalid."""

    def __init__(self, message: str, *, code: str = "workspace.invalid") -> None:
        super().__init__(message)
        self.code = code


def load_validated_renderplan_set(path: Path) -> RenderPlanSet:
    try:
        renderplan_set = load_renderplan_set(path)
        validate_renderplan_set(renderplan_set)
    except (RenderPlanSetError, OSError, ValueError, TypeError) as exc:
        code = getattr(exc, "code", "index.invalid")
        if not isinstance(code, str):
            code = "index.invalid"
        raise BookRenderError(f"invalid renderplan-set.toml: {exc}", code=code) from exc
    return renderplan_set


def validate_profile_snapshot(
    renderplan_set: RenderPlanSet, profile_path: Path
) -> RenderProfile:
    try:
        return validate_renderplan_set_profile(
            renderplan_set,
            profile_path,
            expected_profile_id=renderplan_set.profile_id,
        )
    except RenderPlanSetError as exc:
        code = getattr(exc, "code", "profile.invalid")
        if not isinstance(code, str):
            code = "profile.invalid"
        raise BookRenderError(str(exc), code=code) from exc


def validate_chapter_plan(
    renderplan_set: RenderPlanSet,
    chapter: RenderPlanSetChapter,
    plan: RenderPlan,
) -> None:
    try:
        validate_set_chapter_plan(renderplan_set, chapter, plan)
    except (ValueError, TypeError) as exc:
        code = getattr(exc, "code", "plan.identity_mismatch")
        if not isinstance(code, str):
            code = "plan.identity_mismatch"
        raise BookRenderError(str(exc), code=code) from exc


__all__ = [
    "BookRenderError",
    "load_renderplan_set",
    "load_validated_renderplan_set",
    "validate_chapter_plan",
    "validate_profile_snapshot",
]
