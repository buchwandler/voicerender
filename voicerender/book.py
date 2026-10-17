"""Safe book-level orchestration for ordered RenderPlan variants."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any, Literal

from renderplan import (
    PreflightReport,
    RenderManifest,
    RenderPlan,
    RenderPlanSetChapter,
    load_manifest,
    load_plan,
    load_preflight,
)

from .book_contract import (
    BookRenderError,
    load_validated_renderplan_set,
    validate_chapter_plan,
    validate_profile_snapshot,
)
from .book_layout import (
    assert_safe_output_metadata as _assert_safe_output_metadata,
)
from .book_layout import (
    is_within as _is_within,
)
from .book_layout import (
    regular_file as _regular_file,
)
from .book_layout import (
    require_contained_file as _require_contained_file,
)
from .book_layout import (
    resolve_root as _resolve_root,
)
from .book_layout import (
    resolve_variant_directory,
)
from .book_layout import (
    resolve_workspace as _resolve_workspace,
)
from .errors import ArtifactError, PlanError
from .events import RenderEvent, RenderEventCallback, RenderEventKind, emit_render_event
from .preflight import preflight_render_plan
from .progress import EtaConfidence, ProgressTracker, RenderProgress, RollingEta
from .renderplan_input import EMPHASIS_COMPOSITION_CAPABILITY, EmphasisPolicy
from .renderplan_render import (
    RENDER_DIAGNOSTICS_FILENAME,
    RENDER_MANIFEST_FILENAME,
    render_renderplan,
)
from .storage import verified_wav

BOOK_REPORT_FILENAME = "voicerender-report.toml"
RECHECK_PREFLIGHT_FILENAME = "preflight-recheck.toml"
BookChapterStatus = Literal["complete", "partial", "failed", "skipped", "reused"]


@dataclass(frozen=True, slots=True)
class _LoadedChapter:
    entry: RenderPlanSetChapter
    plan_path: Path
    plan: RenderPlan
    output_dir: Path
    speech_numbers: frozenset[int] | None


@dataclass(frozen=True, slots=True)
class BookRenderRequest:
    """A fully resolved and identity-checked RenderPlan book request."""

    workspace: Path
    variant_name: str
    renderplan_variant_dir: Path
    output_variant_dir: Path
    chapters: tuple[RenderPlanSetChapter, ...]
    speech_numbers: frozenset[int] | None
    cache_dir: Path
    force: bool
    fail_fast: bool
    emphasis_policy: EmphasisPolicy
    chapter_selector: str
    speech_selector: str
    render_profile_id: str
    _loaded_chapters: tuple[_LoadedChapter, ...]
    reset_stale_output: bool = False


@dataclass(frozen=True, slots=True)
class ChapterRenderOutcome:
    index: int
    id: str
    source_number: int
    renderplan: Path
    render_plan_id: str
    output_dir: Path
    manifest: Path | None
    status: BookChapterStatus
    planned_speech: int
    selected_speech: int
    available_speech: int
    rendered: int
    reused: int
    manifest_complete: bool = False
    error_code: str | None = None
    error_message: str | None = None
    recheck_report: Path | None = None
    execution_fingerprint: tuple[dict[str, Any], ...] = ()
    pending_composition: bool = False

    preflight_report: Path | None = None
    replan_required: bool = False
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class BookRenderResult:
    request: BookRenderRequest
    report_path: Path
    outcomes: tuple[ChapterRenderOutcome, ...]
    elapsed_seconds: float = 0.0

    @property
    def workspace(self) -> Path:
        return self.request.workspace

    @property
    def variant(self) -> str:
        return self.request.variant_name

    @property
    def output_variant_dir(self) -> Path:
        return self.request.output_variant_dir

    @property
    def written(self) -> int:
        return sum(outcome.rendered for outcome in self.outcomes)

    @property
    def reused(self) -> int:
        return sum(outcome.reused for outcome in self.outcomes)

    @property
    def requested(self) -> int:
        return len(self.outcomes)

    @property
    def succeeded(self) -> int:
        return sum(
            outcome.status in {"complete", "partial", "reused"}
            for outcome in self.outcomes
        )

    @property
    def failed(self) -> int:
        return sum(outcome.status == "failed" for outcome in self.outcomes)

    @property
    def skipped(self) -> int:
        return sum(outcome.status == "skipped" for outcome in self.outcomes)

    @property
    def full_chapters(self) -> int:
        return sum(
            outcome.manifest_complete and outcome.status != "failed"
            for outcome in self.outcomes
        )

    @property
    def partial_chapters(self) -> int:
        return sum(outcome.status == "partial" for outcome in self.outcomes)

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def pending_composition(self) -> bool:
        return any(outcome.pending_composition for outcome in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        request = self.request
        return {
            "workspace": str(request.workspace),
            "variant": request.variant_name,
            "render_profile_id": request.render_profile_id,
            "renderplan_root": str(request.renderplan_variant_dir.parent),
            "renderplans": str(request.renderplan_variant_dir),
            "output": str(request.output_variant_dir),
            "cache": str(request.cache_dir),
            "emphasis_policy": request.emphasis_policy.value,
            "pending_composition": self.pending_composition,
            "report": str(self.report_path),
            "selection": {
                "chapters": request.chapter_selector,
                "speech": request.speech_selector,
            },
            "requested": self.requested,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "full_chapters": self.full_chapters,
            "partial_chapters": self.partial_chapters,
            "exit_code": self.exit_code,
            "items": [
                _outcome_to_dict(outcome, request.output_variant_dir)
                for outcome in self.outcomes
            ],
        }


class _BookProgress:
    def __init__(
        self,
        chapters: Sequence[_LoadedChapter],
        total_count: int,
    ) -> None:
        self.total_count = total_count
        self.tracker = ProgressTracker(total_count)
        self._engine_by_item: dict[tuple[str, int, str], str] = {}
        self._remaining_by_engine: Counter[str] = Counter()
        self._timings: dict[str, RollingEta] = {}
        for chapter in chapters:
            for item in self._selected_items(chapter):
                segment_id = item.segment_id or item.computed_id()
                engine = self._engine_name(item.binding)
                self._engine_by_item[(chapter.entry.id, item.number, segment_id)] = (
                    engine
                )
                self._remaining_by_engine[engine] += 1

    @staticmethod
    def _selected_items(chapter: _LoadedChapter) -> tuple[Any, ...]:
        if chapter.speech_numbers is None:
            return chapter.plan.speech
        return tuple(
            item
            for item in chapter.plan.speech
            if item.number in chapter.speech_numbers
        )

    @staticmethod
    def _engine_name(binding: Mapping[str, Any]) -> str:
        engine = binding.get("renderer", binding.get("engine"))
        return engine if isinstance(engine, str) and engine else "unknown"

    def _eta(self) -> tuple[float | None, EtaConfidence]:
        active = {
            engine: count
            for engine, count in self._remaining_by_engine.items()
            if count > 0
        }
        samples = [timing for timing in self._timings.values() if timing.sample_count]
        if not active:
            if not samples:
                return 0.0, "unavailable"
            return 0.0, min(
                (timing.confidence for timing in samples),
                key=("unavailable", "low", "medium", "high").index,
            )

        remaining_seconds = 0.0
        confidence: list[EtaConfidence] = []
        for engine, count in active.items():
            timing = self._timings.get(engine)
            if timing is None or timing.sample_count == 0:
                return None, "unavailable"
            estimate = timing.estimate(count)
            if estimate is None:
                return None, "unavailable"
            remaining_seconds += estimate
            confidence.append(timing.confidence)
        return remaining_seconds, min(
            confidence, key=("unavailable", "low", "medium", "high").index
        )

    def snapshot(self) -> RenderProgress:
        snapshot = self.tracker.snapshot()
        eta_seconds, confidence = self._eta()
        return replace(
            snapshot,
            eta_seconds=eta_seconds,
            eta_confidence=confidence,
        )

    def complete(
        self,
        chapter_id: str,
        segment_number: int,
        segment_id: str,
        outcome: Literal["rendered", "reused", "failed"],
        *,
        engine: str | None = None,
        synthesis_seconds: float | None = None,
    ) -> None:
        item_key = (chapter_id, segment_number, segment_id)
        engine_name = self._engine_by_item.get(item_key, engine or "unknown")
        if outcome == "rendered":
            self.tracker.complete("rendered", synthesis_seconds=synthesis_seconds)
        elif outcome == "reused":
            self.tracker.complete("reused")
        else:
            self.tracker.complete("failed")
        if self._remaining_by_engine[engine_name] > 0:
            self._remaining_by_engine[engine_name] -= 1
        if outcome == "rendered" and synthesis_seconds is not None:
            self._timings.setdefault(engine_name, RollingEta()).add_sample(
                synthesis_seconds
            )

    def complete_chapter(
        self,
        chapter: _LoadedChapter,
        outcome: Literal["reused", "failed"],
    ) -> None:
        for item in self._selected_items(chapter):
            self.complete(
                chapter.entry.id,
                item.number,
                item.segment_id or item.computed_id(),
                outcome,
            )

    def event(
        self,
        kind: RenderEventKind,
        *,
        chapter_id: str | None = None,
        chapter_index: int | None = None,
        chapter_count: int | None = None,
        phase: str = "render",
        status: str | None = None,
    ) -> RenderEvent:
        snapshot = self.snapshot()
        return RenderEvent(
            kind=kind,
            scope="book",
            phase=phase,
            chapter_id=chapter_id,
            chapter_index=chapter_index,
            chapter_count=chapter_count,
            status=status,
            completed_count=snapshot.completed_count,
            total_count=self.total_count,
            progress=snapshot.progress,
            rendered_count=snapshot.rendered_count,
            reused_count=snapshot.reused_count,
            failed_count=snapshot.failed_count,
            selected_count=self.total_count,
            elapsed_seconds=snapshot.elapsed_seconds,
            eta_seconds=snapshot.eta_seconds,
            eta_confidence=snapshot.eta_confidence,
            progress_snapshot=snapshot,
        )

    def wrap_child(
        self,
        event: RenderEvent,
        *,
        chapter_id: str,
        chapter_index: int,
        chapter_count: int,
    ) -> RenderEvent:
        if event.kind in {
            RenderEventKind.SEGMENT_RENDERED,
            RenderEventKind.SEGMENT_REUSED,
            RenderEventKind.SEGMENT_FAILED,
        }:
            if event.kind is RenderEventKind.SEGMENT_RENDERED:
                outcome: Literal["rendered", "reused", "failed"] = "rendered"
            elif event.kind is RenderEventKind.SEGMENT_REUSED:
                outcome = "reused"
            else:
                outcome = "failed"
            if event.segment_number is not None and event.segment_id is not None:
                self.complete(
                    chapter_id,
                    event.segment_number,
                    event.segment_id,
                    outcome,
                    engine=event.engine,
                    synthesis_seconds=event.synthesis_seconds,
                )
        snapshot = self.snapshot()
        return replace(
            event,
            scope="book",
            chapter_id=chapter_id,
            chapter_index=chapter_index,
            chapter_count=chapter_count,
            phase_completed_count=(
                event.phase_completed_count
                if event.phase_completed_count is not None
                else event.completed_count
            ),
            phase_total_count=(
                event.phase_total_count
                if event.phase_total_count is not None
                else event.total_count
            ),
            phase_progress=(
                event.phase_progress
                if event.phase_progress is not None
                else event.progress
            ),
            completed_count=snapshot.completed_count,
            total_count=self.total_count,
            progress=snapshot.progress,
            rendered_count=snapshot.rendered_count,
            reused_count=snapshot.reused_count,
            failed_count=snapshot.failed_count,
            selected_count=self.total_count,
            elapsed_seconds=snapshot.elapsed_seconds,
            eta_seconds=snapshot.eta_seconds,
            eta_confidence=snapshot.eta_confidence,
            progress_snapshot=snapshot,
        )


def render_book(
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
    fail_fast: bool = False,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
    on_event: RenderEventCallback | None = None,
) -> BookRenderResult:
    """Render selected chapters in a RenderPlan variant using shared preflight and render APIs."""
    book_started = monotonic()
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
        fail_fast=fail_fast,
        emphasis_policy=resolved_emphasis_policy,
    )
    request.output_variant_dir.mkdir(parents=True, exist_ok=True)
    request.cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = request.output_variant_dir / BOOK_REPORT_FILENAME
    if report_path.is_symlink():
        raise BookRenderError(
            f"book report path must not be a symlink: {report_path}",
            code="workspace.invalid",
        )

    book_total = sum(
        len(chapter.plan.speech)
        if chapter.speech_numbers is None
        else len(chapter.speech_numbers)
        for chapter in request._loaded_chapters
    )
    progress = _BookProgress(request._loaded_chapters, book_total)
    chapter_count = len(request._loaded_chapters)
    emit_render_event(
        on_event,
        progress.event(
            RenderEventKind.BOOK_STARTED,
            chapter_count=chapter_count,
        ),
    )

    outcomes: list[ChapterRenderOutcome] = []
    stopped = False
    for index, chapter in enumerate(request._loaded_chapters, start=1):
        chapter_started = monotonic()
        emit_render_event(
            on_event,
            progress.event(
                RenderEventKind.CHAPTER_STARTED,
                chapter_id=chapter.entry.id,
                chapter_index=index,
                chapter_count=chapter_count,
            ),
        )
        completed_before_chapter = progress.tracker.completed_count
        if stopped:
            outcome = _skipped_outcome(chapter, index)
            emit_render_event(
                on_event,
                progress.event(
                    RenderEventKind.CHAPTER_SKIPPED,
                    chapter_id=chapter.entry.id,
                    chapter_index=index,
                    chapter_count=chapter_count,
                    status="skipped",
                ),
            )
        else:
            if on_event is None:
                outcome = _render_chapter(request, chapter, index)
            else:

                def forward_child_event(
                    event: RenderEvent,
                    chapter_id: str = chapter.entry.id,
                    chapter_index: int = index,
                ) -> None:
                    emit_render_event(
                        on_event,
                        progress.wrap_child(
                            event,
                            chapter_id=chapter_id,
                            chapter_index=chapter_index,
                            chapter_count=chapter_count,
                        ),
                    )

                outcome = _render_chapter(
                    request,
                    chapter,
                    index,
                    on_event=forward_child_event,
                )
            if outcome.status == "reused":
                progress.complete_chapter(chapter, "reused")
                event_kind = RenderEventKind.CHAPTER_REUSED
            else:
                if (
                    outcome.status == "failed"
                    and progress.tracker.completed_count == completed_before_chapter
                ):
                    progress.complete_chapter(chapter, "failed")
                event_kind = RenderEventKind.CHAPTER_FINISHED
            emit_render_event(
                on_event,
                progress.event(
                    event_kind,
                    chapter_id=chapter.entry.id,
                    chapter_index=index,
                    chapter_count=chapter_count,
                    status=outcome.status,
                ),
            )
        outcome = replace(
            outcome,
            elapsed_seconds=max(0.0, monotonic() - chapter_started),
        )
        outcomes.append(outcome)
        if request.fail_fast and outcome.status == "failed":
            stopped = True

    result = BookRenderResult(
        request,
        report_path,
        tuple(outcomes),
        elapsed_seconds=max(0.0, monotonic() - book_started),
    )
    _write_book_report(result)
    result = replace(
        result,
        elapsed_seconds=max(0.0, monotonic() - book_started),
    )
    emit_render_event(
        on_event,
        progress.event(
            RenderEventKind.BOOK_COMPLETED,
            chapter_count=chapter_count,
            status="failed" if result.failed else "completed",
        ),
    )
    return result


def _resolve_request(
    workspace: str | Path,
    *,
    variant: str,
    chapter_selector: str,
    speech_selector: str,
    renderplan_root: str | Path | None,
    output_root: str | Path | None,
    cache_dir: str | Path | None,
    force: bool,
    reset_stale_output: bool,
    fail_fast: bool,
    emphasis_policy: EmphasisPolicy,
) -> BookRenderRequest:
    workspace_path = _resolve_workspace(workspace)
    renderplan_base = _resolve_root(
        renderplan_root or workspace_path / "renderplan", "RenderPlan"
    )
    variant_dir = resolve_variant_directory(renderplan_base, variant)

    set_path = variant_dir / "renderplan-set.toml"
    if set_path.is_symlink() or not _regular_file(set_path):
        raise BookRenderError(
            f"renderplan-set.toml must be a regular file: {set_path}",
            code="index.invalid",
        )
    renderplan_set = load_validated_renderplan_set(set_path)

    expected_variant_name = (
        f"{renderplan_set.variant}--{renderplan_set.profile_fingerprint}"
    )
    if variant != expected_variant_name:
        raise BookRenderError(
            f"variant directory {variant!r} does not match set identity {expected_variant_name!r}",
            code="variant.identity_mismatch",
        )

    profile_path = variant_dir / renderplan_set.snapshot
    _require_contained_file(
        profile_path,
        variant_dir,
        "RenderProfile snapshot",
        missing_code="profile.snapshot_missing",
        reject_symlink=False,
    )
    profile = validate_profile_snapshot(renderplan_set, profile_path)
    profile_id = profile.profile_id or profile.computed_id()

    available_chapters = {chapter.source_number for chapter in renderplan_set.chapters}
    selected_chapter_numbers = _resolve_selection(
        chapter_selector, available_chapters, "chapter"
    )
    selected_entries = tuple(
        chapter
        for chapter in renderplan_set.chapters
        if chapter.source_number in selected_chapter_numbers
    )
    if speech_selector.strip().casefold() != "all" and len(selected_entries) != 1:
        raise BookRenderError(
            "--speech selection requires exactly one selected chapter",
            code="selection.invalid",
        )
    speech_ranges = (
        None
        if speech_selector.strip().casefold() == "all"
        else _parse_number_ranges(speech_selector, "speech")
    )

    output_base = _resolve_root(
        output_root or workspace_path / "voicerender", "VoiceRender"
    )
    output_variant_dir = (output_base / variant).resolve(strict=False)
    if _is_within(output_base, variant_dir) or _is_within(
        output_variant_dir, variant_dir
    ):
        raise BookRenderError(
            "VoiceRender output root must not resolve inside the RenderPlan variant",
            code="workspace.invalid",
        )
    if output_variant_dir.exists() and not output_variant_dir.is_dir():
        raise BookRenderError(
            f"VoiceRender output variant is not a directory: {output_variant_dir}",
            code="workspace.invalid",
        )
    _assert_safe_output_metadata(
        output_variant_dir / BOOK_REPORT_FILENAME, output_variant_dir
    )

    loaded_chapters: list[_LoadedChapter] = []
    chapter_outputs: set[Path] = set()
    for entry in selected_entries:
        plan_candidate = variant_dir.joinpath(*PurePosixPath(entry.renderplan).parts)
        plan_path = _require_contained_file(
            plan_candidate,
            variant_dir,
            f"RenderPlan for {entry.id}",
            missing_code="plan.missing",
            reject_symlink=False,
        )
        try:
            plan = load_plan(plan_path)
        except (OSError, ValueError, TypeError) as exc:
            raise BookRenderError(
                f"cannot load RenderPlan for {entry.id}: {exc}", code="plan.missing"
            ) from exc
        validate_chapter_plan(renderplan_set, entry, plan)
        chapter_output = output_variant_dir / entry.id
        chapter_output_resolved = chapter_output.resolve(strict=False)
        if not _is_within(chapter_output_resolved, output_variant_dir):
            raise BookRenderError(
                f"chapter output path escapes the VoiceRender variant: {chapter_output}",
                code="workspace.invalid",
            )
        if chapter_output.exists() and not chapter_output.is_dir():
            raise BookRenderError(
                f"chapter output path is not a directory: {chapter_output}",
                code="workspace.invalid",
            )
        if chapter_output_resolved in chapter_outputs:
            raise BookRenderError(
                f"selected chapters share an output directory: {chapter_output}",
                code="workspace.invalid",
            )
        _assert_safe_output_metadata(
            chapter_output_resolved / RENDER_MANIFEST_FILENAME,
            chapter_output_resolved,
        )
        _assert_safe_output_metadata(
            chapter_output_resolved / "preflight.toml", chapter_output_resolved
        )
        _assert_safe_output_metadata(
            chapter_output_resolved / RECHECK_PREFLIGHT_FILENAME,
            chapter_output_resolved,
        )
        chapter_outputs.add(chapter_output_resolved)
        speech_numbers = (
            None
            if speech_ranges is None
            else _resolve_ranges(
                speech_ranges, {item.number for item in plan.speech}, "speech"
            )
        )
        loaded_chapters.append(
            _LoadedChapter(
                entry=entry,
                plan_path=plan_path,
                plan=plan,
                output_dir=chapter_output_resolved,
                speech_numbers=speech_numbers,
            )
        )

    resolved_cache = (
        Path(cache_dir).expanduser().resolve(strict=False)
        if cache_dir is not None
        else output_variant_dir / ".voicerender-cache"
    )
    if resolved_cache.exists() and not resolved_cache.is_dir():
        raise BookRenderError(
            f"cache path is not a directory: {resolved_cache}", code="workspace.invalid"
        )
    for chapter_output in chapter_outputs:
        if (
            resolved_cache == chapter_output
            or _is_within(resolved_cache, chapter_output)
            or _is_within(chapter_output, resolved_cache)
        ):
            raise BookRenderError(
                "cache directory must remain outside every chapter output directory",
                code="workspace.invalid",
            )

    return BookRenderRequest(
        workspace=workspace_path,
        variant_name=variant,
        renderplan_variant_dir=variant_dir,
        output_variant_dir=output_variant_dir,
        chapters=selected_entries,
        speech_numbers=(
            loaded_chapters[0].speech_numbers
            if loaded_chapters and speech_ranges is not None
            else None
        ),
        cache_dir=resolved_cache,
        force=force,
        fail_fast=fail_fast,
        emphasis_policy=emphasis_policy,
        chapter_selector=chapter_selector,
        speech_selector=speech_selector,
        render_profile_id=profile_id,
        _loaded_chapters=tuple(loaded_chapters),
        reset_stale_output=reset_stale_output,
    )


def _render_chapter(
    request: BookRenderRequest,
    chapter: _LoadedChapter,
    index: int,
    *,
    on_event: RenderEventCallback | None = None,
) -> ChapterRenderOutcome:
    entry = chapter.entry
    planned = len(chapter.plan.speech)
    selected = (
        planned if chapter.speech_numbers is None else len(chapter.speech_numbers)
    )
    manifest_path = chapter.output_dir / RENDER_MANIFEST_FILENAME

    if not request.force:
        existing = _verified_manifest_for_selection(chapter)
        if existing is not None and _render_diagnostics_reusable(
            chapter, request.emphasis_policy
        ):
            return ChapterRenderOutcome(
                index=index,
                id=entry.id,
                source_number=entry.source_number,
                renderplan=chapter.plan_path,
                render_plan_id=entry.render_plan_id,
                output_dir=chapter.output_dir,
                manifest=manifest_path,
                status="reused",
                planned_speech=planned,
                selected_speech=selected,
                available_speech=len(existing.speech),
                rendered=0,
                reused=selected,
                manifest_complete=existing.complete,
                pending_composition=any(
                    control.name == "emphasis"
                    for item in chapter.plan.speech
                    for control in item.controls
                ),
            )

    try:
        preflight_path = chapter.output_dir / "preflight.toml"
        _assert_safe_output_metadata(preflight_path, chapter.output_dir)
        recheck_report_path = chapter.output_dir / RECHECK_PREFLIGHT_FILENAME
        _assert_safe_output_metadata(recheck_report_path, chapter.output_dir)
        preflight_options: dict[str, Any] = {}
        if on_event is not None:
            preflight_options["on_event"] = on_event
        preflight = _get_preflight(
            chapter.plan,
            preflight_path,
            emphasis_policy=request.emphasis_policy,
            **preflight_options,
        )
    except BookRenderError as exc:
        return _failure_outcome(chapter, index, planned, selected, exc.code, str(exc))
    except Exception as exc:  # noqa: BLE001 - Chapter-specific preflight errors are reported and batch continues.
        return _failure_outcome(
            chapter,
            index,
            planned,
            selected,
            "preflight.reject",
            str(exc) or "RenderPlan preflight failed",
        )

    if preflight.status != "ok":
        code = (
            "preflight.replan" if preflight.status == "replan" else "preflight.reject"
        )
        return _failure_outcome(
            chapter,
            index,
            planned,
            selected,
            code,
            f"RenderPlan preflight status is {preflight.status!r}",
            preflight_report=preflight_path,
            replan_required=preflight.status == "replan",
        )

    try:
        _assert_safe_output_metadata(manifest_path, chapter.output_dir)
        render_options: dict[str, Any] = {}
        if on_event is not None:
            render_options["on_event"] = on_event
        result = render_renderplan(
            chapter.plan,
            preflight=preflight,
            output_dir=chapter.output_dir,
            cache_dir=request.cache_dir,
            force=request.force,
            reset_stale_output=request.reset_stale_output,
            recheck_report_path=recheck_report_path,
            speech_numbers=chapter.speech_numbers,
            emphasis_policy=request.emphasis_policy,
            **render_options,
        )
    except (ArtifactError, PlanError, OSError, ValueError, TypeError) as exc:
        error_code = getattr(exc, "code", None)
        return _failure_outcome(
            chapter,
            index,
            planned,
            selected,
            error_code if isinstance(error_code, str) else "render.failed",
            str(exc),
            manifest=manifest_path if manifest_path.exists() else None,
        )

    status: BookChapterStatus = "complete" if result.manifest.complete else "partial"
    if result.failed:
        status = "failed"
    return ChapterRenderOutcome(
        index=index,
        id=entry.id,
        source_number=entry.source_number,
        renderplan=chapter.plan_path,
        render_plan_id=entry.render_plan_id,
        output_dir=chapter.output_dir,
        manifest=result.manifest_path,
        status=status,
        planned_speech=result.planned_count,
        selected_speech=result.selected_count,
        available_speech=result.available_count,
        rendered=result.rendered_count,
        reused=result.reused_count,
        manifest_complete=result.manifest.complete,
        error_code=((result.error_code or "render.failed") if result.failed else None),
        error_message=result.error,
        recheck_report=(
            recheck_report_path if result.fresh_preflight is not None else None
        ),
        execution_fingerprint=result.execution_fingerprint,
        pending_composition=result.pending_composition,
        preflight_report=(
            recheck_report_path if result.fresh_preflight is not None else None
        ),
        replan_required=(
            result.fresh_preflight is not None
            and result.fresh_preflight.status == "replan"
        ),
    )


def _get_preflight(
    plan: RenderPlan,
    path: Path,
    *,
    emphasis_policy: EmphasisPolicy,
    on_event: RenderEventCallback | None = None,
) -> PreflightReport:
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if not _is_within(resolved, path.parent.resolve(strict=True)):
                raise BookRenderError(
                    f"preflight metadata symlink escapes chapter output: {path}",
                    code="workspace.invalid",
                )
        if path.is_file():
            try:
                existing = load_preflight(path)
                if _preflight_is_reusable(
                    existing, plan, emphasis_policy=emphasis_policy
                ):
                    return existing
            except (OSError, ValueError, TypeError):
                pass
    if on_event is None:
        return preflight_render_plan(plan, path, emphasis_policy=emphasis_policy)
    return preflight_render_plan(
        plan,
        path,
        emphasis_policy=emphasis_policy,
        on_event=on_event,
    )


def _preflight_is_reusable(
    report: PreflightReport,
    plan: RenderPlan,
    *,
    emphasis_policy: EmphasisPolicy = EmphasisPolicy.REJECT,
) -> bool:
    if report.status != "ok":
        return False
    try:
        report.validate_against(plan)
    except ValueError:
        return False
    checks = {
        (check.segment_number, check.segment_id): check for check in report.checks
    }
    required = [item for item in plan.speech if item.preflight_required]
    if len(checks) != len(required):
        return False
    for item in required:
        key = (item.number, item.segment_id or item.computed_id())
        check = checks.get(key)
        if check is None or check.status != "ok":
            return False
        if any(control.name == "emphasis" for control in item.controls) and (
            emphasis_policy is not EmphasisPolicy.DEFER_TO_COMPOSE
            or EMPHASIS_COMPOSITION_CAPABILITY not in (check.detail or "")
        ):
            return False
    return True


def _render_diagnostics_reusable(
    chapter: _LoadedChapter, emphasis_policy: EmphasisPolicy
) -> bool:
    expected = {
        (
            item.number,
            item.segment_id or item.computed_id(),
            sum(control.name == "emphasis" for control in item.controls),
        )
        for item in chapter.plan.speech
        if any(control.name == "emphasis" for control in item.controls)
    }
    if not expected:
        return True
    if emphasis_policy is not EmphasisPolicy.DEFER_TO_COMPOSE:
        return False
    path = chapter.output_dir / RENDER_DIAGNOSTICS_FILENAME
    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "voicerender.render-diagnostics.v1"
        or payload.get("render_plan_id")
        != (chapter.plan.plan_id or chapter.plan.computed_id())
        or payload.get("emphasis_policy") != emphasis_policy.value
        or payload.get("pending_composition") is not True
        or not isinstance(payload.get("deferred_controls"), list)
    ):
        return False
    observed: list[tuple[int, str, int]] = []
    for entry in payload["deferred_controls"]:
        if (
            not isinstance(entry, dict)
            or entry.get("capability") != EMPHASIS_COMPOSITION_CAPABILITY
            or type(entry.get("segment_number")) is not int
            or not isinstance(entry.get("segment_id"), str)
            or type(entry.get("count")) is not int
        ):
            return False
        observed.append((entry["segment_number"], entry["segment_id"], entry["count"]))
    return len(observed) == len(expected) and set(observed) == expected


def _verified_manifest_for_selection(chapter: _LoadedChapter) -> RenderManifest | None:
    path = chapter.output_dir / RENDER_MANIFEST_FILENAME
    if path.is_symlink() or not path.is_file():
        return None
    try:
        manifest = load_manifest(path)
        manifest.validate_against(chapter.plan)
        expected = {
            (item.number, item.segment_id or item.computed_id()): item
            for item in chapter.plan.speech
        }
        verified: set[tuple[int, str]] = set()
        for artifact in manifest.speech:
            key = (artifact.segment_number, artifact.segment_id)
            if (
                key in verified
                or key not in expected
                or artifact.wav != expected[key].wav
            ):
                return None
            if not verified_wav(
                chapter.output_dir / artifact.wav,
                {
                    "sha256": artifact.sha256.removeprefix("sha256:"),
                    "frames": artifact.frames,
                    "sample_rate": artifact.sample_rate,
                    "channels": artifact.channels,
                },
            ):
                return None
            verified.add(key)
        selected = (
            set(expected)
            if chapter.speech_numbers is None
            else {
                (number, key[1])
                for number in chapter.speech_numbers
                for key in expected
                if key[0] == number
            }
        )
        if not selected.issubset(verified):
            return None
        if chapter.speech_numbers is None and not manifest.complete:
            return None
        if manifest.complete and verified != set(expected):
            return None
        return manifest
    except (OSError, ValueError, TypeError):
        return None


def _parse_number_ranges(selector: str, label: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(selector, str) or not selector.strip():
        raise BookRenderError(
            f"{label} selector must not be empty", code="selection.invalid"
        )
    ranges: list[tuple[int, int]] = []
    for part in selector.split(","):
        token = part.strip()
        if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", token):
            raise BookRenderError(
                f"invalid {label} selector token {token!r}", code="selection.invalid"
            )
        values = token.split("-", maxsplit=1)
        start = int(values[0])
        end = int(values[1]) if len(values) == 2 else start
        if start <= 0 or end <= 0 or end < start:
            raise BookRenderError(
                f"{label} selector ranges must be positive and ascending",
                code="selection.invalid",
            )
        ranges.append((start, end))
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _resolve_selection(
    selector: str, available: set[int], label: str
) -> frozenset[int]:
    if not isinstance(selector, str) or not selector.strip():
        raise BookRenderError(
            f"{label} selector must not be empty", code="selection.invalid"
        )
    if selector.strip().casefold() == "all":
        return frozenset(available)
    ranges = _parse_number_ranges(selector, label)
    return _resolve_ranges(ranges, available, label)


def _resolve_ranges(
    ranges: Sequence[tuple[int, int]], available: set[int], label: str
) -> frozenset[int]:
    selected = {
        number
        for number in available
        if any(start <= number <= end for start, end in ranges)
    }
    requested_count = sum(end - start + 1 for start, end in ranges)
    if len(selected) != requested_count:
        for start, end in ranges:
            candidate = start
            while candidate <= end and candidate in available:
                candidate += 1
            if candidate <= end:
                raise BookRenderError(
                    f"unknown {label} number {candidate}", code="selection.invalid"
                )
    return frozenset(selected)


def _failure_outcome(
    chapter: _LoadedChapter,
    index: int,
    planned: int,
    selected: int,
    error_code: str,
    error_message: str,
    *,
    manifest: Path | None = None,
    preflight_report: Path | None = None,
    replan_required: bool = False,
) -> ChapterRenderOutcome:
    return ChapterRenderOutcome(
        index=index,
        id=chapter.entry.id,
        source_number=chapter.entry.source_number,
        renderplan=chapter.plan_path,
        render_plan_id=chapter.entry.render_plan_id,
        output_dir=chapter.output_dir,
        manifest=manifest,
        status="failed",
        planned_speech=planned,
        selected_speech=selected,
        available_speech=0,
        rendered=0,
        reused=0,
        error_code=error_code,
        error_message=error_message,
        preflight_report=preflight_report,
        replan_required=replan_required,
    )


def _skipped_outcome(chapter: _LoadedChapter, index: int) -> ChapterRenderOutcome:
    selected = (
        len(chapter.plan.speech)
        if chapter.speech_numbers is None
        else len(chapter.speech_numbers)
    )
    return ChapterRenderOutcome(
        index=index,
        id=chapter.entry.id,
        source_number=chapter.entry.source_number,
        renderplan=chapter.plan_path,
        render_plan_id=chapter.entry.render_plan_id,
        output_dir=chapter.output_dir,
        manifest=None,
        status="skipped",
        planned_speech=len(chapter.plan.speech),
        selected_speech=selected,
        available_speech=0,
        rendered=0,
        reused=0,
        error_code="batch.fail_fast",
        error_message="Skipped after an earlier chapter failed because --fail-fast is enabled.",
    )


def _outcome_to_dict(
    outcome: ChapterRenderOutcome, output_variant_dir: Path
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "index": outcome.index,
        "id": outcome.id,
        "source_number": outcome.source_number,
        "renderplan": _relative_posix(outcome.renderplan, output_variant_dir),
        "render_plan_id": outcome.render_plan_id,
        "output_dir": _relative_posix(outcome.output_dir, output_variant_dir),
        "status": outcome.status,
        "planned_speech": outcome.planned_speech,
        "selected_speech": outcome.selected_speech,
        "available_speech": outcome.available_speech,
        "rendered": outcome.rendered,
        "reused": outcome.reused,
        "manifest_complete": outcome.manifest_complete,
        "pending_composition": outcome.pending_composition,
        "replan_required": outcome.replan_required,
    }
    if outcome.manifest is not None:
        item["manifest"] = _relative_posix(outcome.manifest, output_variant_dir)
    if outcome.preflight_report is not None:
        item["preflight_report"] = _relative_posix(
            outcome.preflight_report, output_variant_dir
        )
    if outcome.recheck_report is not None:
        item["recheck_report"] = _relative_posix(
            outcome.recheck_report, output_variant_dir
        )
    if outcome.execution_fingerprint:
        item["execution_fingerprint"] = list(outcome.execution_fingerprint)
    if outcome.error_code is not None:
        item["error_code"] = outcome.error_code
    if outcome.error_message is not None:
        item["error_message"] = outcome.error_message
    return item


def _relative_posix(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, start=base)).as_posix()


def _write_book_report(result: BookRenderResult) -> None:
    request = result.request
    payload: dict[str, Any] = {
        "format": "voicerender-book-report",
        "version": 1,
        "variant": request.variant_name,
        "emphasis_policy": request.emphasis_policy.value,
        "pending_composition": result.pending_composition,
        "render_profile_id": request.render_profile_id,
        "requested": result.requested,
        "succeeded": result.succeeded,
        "failed": result.failed,
        "skipped": result.skipped,
        "full_chapters": result.full_chapters,
        "partial_chapters": result.partial_chapters,
        "selection": {
            "chapters": request.chapter_selector,
            "speech": request.speech_selector,
        },
        "item": [
            _outcome_to_dict(outcome, request.output_variant_dir)
            for outcome in result.outcomes
        ],
    }
    lines: list[str] = []
    for key, value in payload.items():
        if key in {"selection", "item"}:
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    selection = payload["selection"]
    assert isinstance(selection, Mapping)
    lines.extend(["", "[selection]"])
    for key, value in selection.items():
        lines.append(f"{key} = {_toml_value(value)}")
    items = payload["item"]
    assert isinstance(items, list)
    for item in items:
        assert isinstance(item, Mapping)
        lines.extend(["", "[[item]]"])
        for key, value in item.items():
            if key == "execution_fingerprint":
                key = "execution_fingerprint_json"
                value = json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            lines.append(f"{key} = {_toml_value(value)}")
    destination = result.report_path
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        raise BookRenderError(
            f"cannot write VoiceRender book report {destination}: {exc}",
            code="workspace.invalid",
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"unsupported TOML report value: {type(value).__name__}")


__all__ = [
    "BOOK_REPORT_FILENAME",
    "RECHECK_PREFLIGHT_FILENAME",
    "BookChapterStatus",
    "BookRenderError",
    "BookRenderRequest",
    "BookRenderResult",
    "ChapterRenderOutcome",
    "render_book",
]
