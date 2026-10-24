"""State-aware batch orchestration for canonical SSMD bundle variants."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from .bundle import BundleRenderResult, render_bundle
from .bundle_contract import BundleKind
from .error_diagnostics import diagnostic_summary_fields, exception_diagnostic
from .events import RenderEventCallback
from .session_pool import validate_max_open_sessions
from .variants import BundleVariant, VariantState, list_bundle_variants

BatchVariantStatus = Literal[
    "complete",
    "partial",
    "failed",
    "skipped_complete",
    "invalid",
    "not_attempted",
]
BatchAttemptStatus = Literal["complete", "failed"]


@dataclass(frozen=True, slots=True)
class BundleVariantRenderOutcome:
    """Selection, execution, and final inspection details for one variant."""

    name: str
    initial_state: VariantState
    plan_status: Literal["valid", "invalid"]
    selected: bool
    status: BatchVariantStatus
    final_state: VariantState | None
    attempt_status: BatchAttemptStatus | None = None
    exit_code: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_type: str | None = None
    error_stage: str | None = None
    error_root_cause_type: str | None = None
    error_root_cause_message: str | None = None
    render_diagnostics: Path | None = None
    validation_error: str | None = None
    output_error: str | None = None
    pending_composition: bool | None = None
    result: BundleRenderResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial_state": self.initial_state,
            "plan_status": self.plan_status,
            "selected": self.selected,
            "status": self.status,
            "attempt_status": self.attempt_status,
            "final_state": self.final_state,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "error_stage": self.error_stage,
            "error_root_cause_type": self.error_root_cause_type,
            "error_root_cause_message": self.error_root_cause_message,
            "render_diagnostics": self.render_diagnostics.as_posix()
            if self.render_diagnostics
            else None,
            "validation_error": self.validation_error,
            "output_error": self.output_error,
            "pending_composition": self.pending_composition,
            "result": self.result.to_dict() if self.result is not None else None,
        }


@dataclass(frozen=True, slots=True)
class BundleBatchRenderSummary:
    """Stable aggregate counts for a batch render."""

    discovered: int
    scheduled: int
    attempted: int
    completed: int
    partial: int
    failed: int
    skipped_complete: int
    invalid: int
    not_attempted: int

    def to_dict(self) -> dict[str, int]:
        return {
            "discovered": self.discovered,
            "scheduled": self.scheduled,
            "attempted": self.attempted,
            "completed": self.completed,
            "partial": self.partial,
            "failed": self.failed,
            "skipped_complete": self.skipped_complete,
            "invalid": self.invalid,
            "not_attempted": self.not_attempted,
        }


@dataclass(frozen=True, slots=True)
class BundleBatchRenderResult:
    """Ordered result of rendering eligible variants in one canonical bundle."""

    bundle: Path
    kind: BundleKind
    renderplan_root: Path
    output_root: Path
    force: bool
    variants: tuple[BundleVariantRenderOutcome, ...]
    summary: BundleBatchRenderSummary
    exit_code: int
    error_code: str | None = None
    error_message: str | None = None
    error_type: str | None = None
    error_stage: str | None = None
    error_root_cause_type: str | None = None
    error_root_cause_message: str | None = None
    render_diagnostics: Path | None = None

    @property
    def no_work(self) -> bool:
        return (
            self.summary.attempted == 0
            and self.summary.invalid == 0
            and self.summary.not_attempted == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "voicerender-bundle-batch-result",
            "version": 1,
            "kind": self.kind,
            "bundle": str(self.bundle),
            "renderplan_root": str(self.renderplan_root),
            "output_root": str(self.output_root),
            "force": self.force,
            "summary": self.summary.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "error_stage": self.error_stage,
            "error_root_cause_type": self.error_root_cause_type,
            "error_root_cause_message": self.error_root_cause_message,
            "render_diagnostics": self.render_diagnostics.as_posix()
            if self.render_diagnostics
            else None,
            "exit_code": self.exit_code,
        }


def render_bundle_batch(
    bundle: str | Path,
    *,
    chapters: str = "all",
    speech: str = "all",
    renderplan_root: str | Path | None = None,
    output_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    reset_stale: bool = False,
    emphasis_policy: Literal["reject", "defer-to-compose"] = "reject",
    fail_fast: bool = False,
    max_open_sessions: int = 1,
    on_progress: RenderEventCallback | None = None,
) -> BundleBatchRenderResult:
    """Render valid, incomplete variants in deterministic discovery order.

    Discovery is read-only. Each scheduled variant is handed to the existing
    single-variant renderer, which performs its normal identity and output
    safety validation before writing.
    """
    max_open_sessions = validate_max_open_sessions(max_open_sessions)
    listing = list_bundle_variants(
        bundle,
        renderplan_root=renderplan_root,
        output_root=output_root,
    )
    if listing.kind == "document" and chapters.strip().casefold() != "all":
        raise ValueError("--chapters must be 'all' for a document bundle")

    outcomes: list[BundleVariantRenderOutcome] = []
    scheduled_names: list[str] = []

    for variant in listing.variants:
        if variant.plan_status == "invalid" or variant.state == "invalid":
            outcomes.append(_initial_outcome(variant, status="invalid", selected=False))
        elif variant.state == "complete" and not force:
            outcomes.append(
                _initial_outcome(variant, status="skipped_complete", selected=False)
            )
        else:
            scheduled_names.append(variant.name)
            outcomes.append(
                _initial_outcome(variant, status="not_attempted", selected=True)
            )

    outcome_index = {outcome.name: index for index, outcome in enumerate(outcomes)}
    failed = False
    for name in scheduled_names:
        if fail_fast and failed:
            break
        context_setter = getattr(on_progress, "set_context", None)
        if callable(context_setter):
            context_setter(name)
        try:
            result = render_bundle(
                listing.bundle,
                variant=name,
                chapters=chapters,
                speech=speech,
                renderplan_root=renderplan_root,
                output_root=output_root,
                cache_dir=cache_dir,
                force=force,
                reset_stale=reset_stale,
                emphasis_policy=emphasis_policy,
                fail_fast=fail_fast,
                on_progress=on_progress,
                max_open_sessions=max_open_sessions,
            )
        except Exception as exc:  # noqa: BLE001 - Keep batch failures isolated.
            error_code, error_message = _exception_details(exc)
            error_summary = _exception_summary_fields(exc)
            outcomes[outcome_index[name]] = replace(
                outcomes[outcome_index[name]],
                status="failed",
                attempt_status="failed",
                exit_code=1,
                error_code=error_code,
                error_message=error_message,
                error_type=type(exc).__name__,
                **cast(Any, error_summary),
            )
            failed = True
            continue

        exit_code = result.exit_code
        (
            result_error_code,
            result_error_message,
            result_error_type,
            result_error_stage,
            result_error_root_cause_type,
            result_error_root_cause_message,
            result_render_diagnostics,
        ) = _result_error_details(result)
        if exit_code != 0:
            if result_error_code is None:
                result_error_code = "render.failed"
            if result_error_message is None:
                result_error_message = f"renderer exited with status {exit_code}"
            outcomes[outcome_index[name]] = replace(
                outcomes[outcome_index[name]],
                status="failed",
                attempt_status="failed",
                exit_code=exit_code,
                error_code=result_error_code,
                error_message=result_error_message,
                error_type=result_error_type,
                error_stage=result_error_stage,
                error_root_cause_type=result_error_root_cause_type,
                error_root_cause_message=result_error_root_cause_message,
                render_diagnostics=result_render_diagnostics,
                result=result,
            )
            failed = True
        else:
            outcomes[outcome_index[name]] = replace(
                outcomes[outcome_index[name]],
                status="partial",
                attempt_status="complete",
                exit_code=0,
                result=result,
                error_code=result_error_code,
                error_message=result_error_message,
                error_type=result_error_type,
                error_stage=result_error_stage,
                error_root_cause_type=result_error_root_cause_type,
                error_root_cause_message=result_error_root_cause_message,
                render_diagnostics=result_render_diagnostics,
            )

    final_listing_error: Exception | None = None
    final_by_name: dict[str, BundleVariant] = {}
    try:
        final_listing = list_bundle_variants(
            listing.bundle,
            renderplan_root=renderplan_root,
            output_root=output_root,
        )
        final_by_name = {variant.name: variant for variant in final_listing.variants}
    except Exception as exc:  # noqa: BLE001 - Preserve completed attempt details.
        final_listing_error = exc

    finalized: list[BundleVariantRenderOutcome] = []
    for outcome in outcomes:
        final_variant = final_by_name.get(outcome.name)
        final_state = final_variant.state if final_variant is not None else None
        pending = (
            final_variant.pending_composition
            if final_variant is not None
            and final_variant.pending_composition is not None
            else outcome.pending_composition
        )
        status = outcome.status
        if outcome.attempt_status == "complete":
            status = "complete" if final_state == "complete" else "partial"
        finalized.append(
            replace(
                outcome,
                status=status,
                final_state=final_state,
                pending_composition=pending,
                output_error=(
                    final_variant.output_error
                    if final_variant is not None
                    and final_variant.output_error is not None
                    else outcome.output_error
                ),
            )
        )

    summary = _summarize(len(listing.variants), len(scheduled_names), finalized)
    exit_code = (
        2
        if final_listing_error is not None
        else 1
        if summary.failed or summary.invalid or summary.not_attempted
        else 0
    )
    final_error_code, final_error_message = (
        _exception_details(final_listing_error)
        if final_listing_error is not None
        else (None, None)
    )
    return BundleBatchRenderResult(
        bundle=listing.bundle,
        kind=listing.kind,
        renderplan_root=listing.renderplan_root,
        output_root=listing.output_root,
        force=force,
        variants=tuple(finalized),
        summary=summary,
        exit_code=exit_code,
        error_code=final_error_code,
        error_message=final_error_message,
        error_type=(
            type(final_listing_error).__name__
            if final_listing_error is not None
            else None
        ),
        **cast(
            Any,
            (
                _exception_summary_fields(final_listing_error)
                if final_listing_error is not None
                else diagnostic_summary_fields(None)
            ),
        ),
    )


def _initial_outcome(
    variant: BundleVariant,
    *,
    status: BatchVariantStatus,
    selected: bool,
) -> BundleVariantRenderOutcome:
    error_code = variant.error_code
    error_message = variant.validation_error or variant.error_message
    if status == "invalid" and error_code is None:
        error_code = "plan.invalid"
    return BundleVariantRenderOutcome(
        name=variant.name,
        initial_state=variant.state,
        plan_status=variant.plan_status,
        selected=selected,
        status=status,
        final_state=None,
        error_code=error_code,
        error_message=error_message,
        validation_error=variant.validation_error,
        output_error=variant.output_error,
        pending_composition=variant.pending_composition,
    )


def _summarize(
    discovered: int,
    scheduled: int,
    outcomes: list[BundleVariantRenderOutcome],
) -> BundleBatchRenderSummary:
    return BundleBatchRenderSummary(
        discovered=discovered,
        scheduled=scheduled,
        attempted=sum(outcome.attempt_status is not None for outcome in outcomes),
        completed=sum(outcome.status == "complete" for outcome in outcomes),
        partial=sum(outcome.status == "partial" for outcome in outcomes),
        failed=sum(outcome.status == "failed" for outcome in outcomes),
        skipped_complete=sum(
            outcome.status == "skipped_complete" for outcome in outcomes
        ),
        invalid=sum(outcome.status == "invalid" for outcome in outcomes),
        not_attempted=sum(outcome.status == "not_attempted" for outcome in outcomes),
    )


def _exception_details(exc: Exception) -> tuple[str, str]:
    code = getattr(exc, "code", None)
    return (
        code if isinstance(code, str) and code else "render.error",
        str(exc) or type(exc).__name__,
    )


def _exception_summary_fields(exc: Exception) -> dict[str, str | None]:
    try:
        diagnostic = exception_diagnostic(exc)
    except Exception:  # noqa: BLE001 - reporting must not mask a batch error.
        return diagnostic_summary_fields(None)
    return diagnostic_summary_fields(diagnostic)


def _result_error_details(
    result: BundleRenderResult,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    Path | None,
]:
    value = result.value
    fields: dict[str, object] = {
        "error_code": getattr(value, "error_code", None),
        "error_message": getattr(value, "error_message", None),
        "error_type": getattr(value, "error_type", None),
        "error_stage": getattr(value, "error_stage", None),
        "error_root_cause_type": getattr(value, "error_root_cause_type", None),
        "error_root_cause_message": getattr(value, "error_root_cause_message", None),
        "render_diagnostics": getattr(value, "render_diagnostics", None),
    }
    for outcome in getattr(value, "outcomes", ()):
        if getattr(outcome, "status", None) != "failed":
            continue
        for name in (
            "error_code",
            "error_message",
            "error_type",
            "error_stage",
            "error_root_cause_type",
            "error_root_cause_message",
            "render_diagnostics",
        ):
            if fields[name] is None:
                fields[name] = getattr(outcome, name, None)
        if any(fields[name] is not None for name in fields):
            break

    text_fields = (
        "error_code",
        "error_message",
        "error_type",
        "error_stage",
        "error_root_cause_type",
        "error_root_cause_message",
    )
    values = tuple(
        value if isinstance(value, str) else None
        for value in (fields[name] for name in text_fields)
    )
    diagnostics = fields["render_diagnostics"]
    diagnostics_path = diagnostics if isinstance(diagnostics, Path) else None
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        diagnostics_path,
    )


__all__ = [
    "BatchAttemptStatus",
    "BatchVariantStatus",
    "BundleBatchRenderResult",
    "BundleBatchRenderSummary",
    "BundleVariantRenderOutcome",
    "render_bundle_batch",
]
