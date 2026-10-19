"""Side-effect-free workload estimates for VoiceRender workflows."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from renderplan import RenderPlan, SpeechItem, load_manifest, load_plan

from .book import _resolve_request
from .errors import ArtifactError, PlanError
from .progress import EtaConfidence
from .renderplan_input import EmphasisPolicy
from .renderplan_render import (
    RENDER_MANIFEST_FILENAME,
    _select_speech_items,
)
from .storage import verified_wav


def estimate_render_document(
    workspace: str | Path,
    *,
    variant: str,
    speech: str = "all",
    renderplan_root: str | Path | None = None,
    output_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    reset_stale_output: bool = False,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
) -> RenderEstimate:
    """Estimate one document's selected RenderPlan work without side effects."""
    from .document import _resolve_document_request

    request = _resolve_document_request(
        workspace,
        variant=variant,
        speech=speech,
        renderplan_root=renderplan_root,
        output_root=output_root,
        cache_dir=cache_dir,
        force=force,
        reset_stale_output=reset_stale_output,
        emphasis_policy=emphasis_policy,
    )
    return estimate_renderplan(
        request.plan,
        output_dir=request.output_dir,
        speech_numbers=request.speech_numbers,
        force=force,
        reset_stale_output=reset_stale_output,
    )


@dataclass(frozen=True, slots=True)
class RenderEstimate:
    """Workload and verified-output reuse counts without a latency prediction."""

    selected_count: int
    known_output_reuse_count: int
    pending_count: int
    estimated_seconds: float | None = None
    confidence: EtaConfidence = "unavailable"
    source: str = "workload-only"


def estimate_renderplan(
    plan: RenderPlan | str | Path,
    *,
    output_dir: str | Path | None = None,
    speech_numbers: Collection[int] | None = None,
    force: bool = False,
    reset_stale_output: bool = False,
) -> RenderEstimate:
    """Inspect selected workload and verified output reuse without synthesis.

    ``output_dir`` is optional. Without it the estimate reports workload only;
    with it, valid artifacts in a matching RenderManifest count as reusable.
    No preflight capacity measurement is interpreted as latency evidence.
    """

    parsed_plan = load_plan(plan) if isinstance(plan, (str, Path)) else plan
    if not isinstance(parsed_plan, RenderPlan):
        raise PlanError("estimate input must be a RenderPlan")
    selected = _select_speech_items(parsed_plan, speech_numbers)
    reusable = 0
    if output_dir is not None and not (force or reset_stale_output):
        output = Path(output_dir).expanduser().resolve(strict=False)
        reusable = _verified_output_reuse_count(
            parsed_plan,
            output,
            selected,
            output / RENDER_MANIFEST_FILENAME,
        )
    return _estimate(len(selected), reusable)


def estimate_render_book(
    workspace: str | Path,
    *,
    variant: str,
    chapters: str = "all",
    speech: str = "all",
    renderplan_root: str | Path | None = None,
    output_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    reset_stale_output: bool = False,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
) -> RenderEstimate:
    """Inspect a selected book workload and verified chapter outputs only."""

    try:
        resolved_emphasis_policy = EmphasisPolicy(emphasis_policy)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "emphasis_policy must be 'reject' or 'defer-to-compose'"
        ) from exc
    request = _resolve_request(
        workspace,
        variant=variant,
        chapter_selector=chapters,
        speech_selector=speech,
        renderplan_root=renderplan_root,
        output_root=output_root,
        cache_dir=cache_dir,
        force=force,
        reset_stale_output=reset_stale_output,
        fail_fast=False,
        emphasis_policy=resolved_emphasis_policy,
    )
    selected_count = 0
    reusable_count = 0
    for chapter in request._loaded_chapters:
        selected = (
            chapter.plan.speech
            if chapter.speech_numbers is None
            else tuple(
                item
                for item in chapter.plan.speech
                if item.number in chapter.speech_numbers
            )
        )
        selected_count += len(selected)
        if not (force or reset_stale_output):
            reusable_count += _verified_output_reuse_count(
                chapter.plan,
                chapter.output_dir,
                selected,
                chapter.output_dir / RENDER_MANIFEST_FILENAME,
            )
    return _estimate(selected_count, reusable_count)


def _verified_output_reuse_count(
    plan: RenderPlan,
    output_dir: Path,
    selected: tuple[SpeechItem, ...],
    manifest_path: Path,
) -> int:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return 0
    try:
        manifest = load_manifest(manifest_path)
        manifest.validate_against(plan)
    except (ArtifactError, OSError, ValueError, TypeError, PlanError):
        return 0
    expected = {
        (item.number, item.segment_id or item.computed_id()): item
        for item in plan.speech
    }
    selected_keys = {
        (item.number, item.segment_id or item.computed_id()) for item in selected
    }
    verified: set[tuple[int, str]] = set()
    for artifact in manifest.speech:
        key = (artifact.segment_number, artifact.segment_id)
        item = expected.get(key)
        if key not in selected_keys or key in verified or item is None:
            continue
        if artifact.wav != item.wav:
            continue
        try:
            valid = verified_wav(
                output_dir / artifact.wav,
                {
                    "sha256": artifact.sha256.removeprefix("sha256:"),
                    "frames": artifact.frames,
                    "sample_rate": artifact.sample_rate,
                    "channels": artifact.channels,
                },
            )
        except (OSError, ValueError, TypeError):
            valid = False
        if valid:
            verified.add(key)
    return len(verified)


def _estimate(selected_count: int, reusable_count: int) -> RenderEstimate:
    reusable = min(max(reusable_count, 0), selected_count)
    return RenderEstimate(
        selected_count=selected_count,
        known_output_reuse_count=reusable,
        pending_count=selected_count - reusable,
        estimated_seconds=None,
        confidence="unavailable",
        source="workload-only",
    )


__all__ = [
    "RenderEstimate",
    "estimate_render_book",
    "estimate_render_document",
    "estimate_renderplan",
]
