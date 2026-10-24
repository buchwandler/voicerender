"""Persisted RenderPlan preflight without waveform synthesis."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from renderplan import (
    PreflightCheck,
    PreflightReport,
    Producer,
    RenderPlan,
    ReplanRequest,
    SpeechItem,
    save_preflight,
)

from .engines.base import EngineSelection
from .engines.registry import get_adapter
from .errors import PlanError
from .events import RenderEvent, RenderEventCallback, RenderEventKind, emit_render_event
from .progress import ProgressTracker
from .renderplan_input import (
    EmphasisPolicy,
    MeasurementOutcome,
    PreflightDisposition,
    classify_measurement,
    engine_selection_from_binding,
    speech_request_conversion_from_renderplan,
)
from .session_pool import (
    EngineSessionPool,
    SessionOpenFailure,
    validate_max_open_sessions,
)


def _preflight_progress_event(
    kind: RenderEventKind,
    tracker: ProgressTracker,
    *,
    plan_id: str,
    segment_number: int | None = None,
    segment_id: str | None = None,
    status: str | None = None,
) -> RenderEvent:
    snapshot = tracker.snapshot()
    return RenderEvent(
        kind=kind,
        plan_id=plan_id,
        scope="preflight",
        phase="preflight",
        segment_number=segment_number,
        segment_id=segment_id,
        status=status,
        completed_count=snapshot.completed_count,
        total_count=snapshot.total_count,
        progress=snapshot.progress,
        rendered_count=snapshot.rendered_count,
        reused_count=snapshot.reused_count,
        failed_count=snapshot.failed_count,
        selected_count=snapshot.total_count,
        phase_completed_count=snapshot.completed_count,
        phase_total_count=snapshot.total_count,
        phase_progress=snapshot.progress,
        elapsed_seconds=snapshot.elapsed_seconds,
        eta_seconds=snapshot.eta_seconds,
        eta_confidence=snapshot.eta_confidence,
        progress_snapshot=snapshot,
    )


def preflight_render_plan(
    plan: RenderPlan,
    output: str | Path,
    *,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
    max_open_sessions: int = 1,
    on_event: RenderEventCallback | None = None,
) -> PreflightReport:
    """Measure each required speech occurrence, then validate and persist its report."""
    if not isinstance(plan, RenderPlan):
        raise PlanError("preflight input must be a RenderPlan")
    max_open_sessions = validate_max_open_sessions(max_open_sessions)
    checks: list[PreflightCheck] = []

    required_count = sum(item.preflight_required for item in plan.speech)
    progress_tracker = ProgressTracker(required_count)
    plan_id = plan.plan_id or plan.computed_id()
    emit_render_event(
        on_event,
        _preflight_progress_event(
            RenderEventKind.PREFLIGHT_STARTED,
            progress_tracker,
            plan_id=plan_id,
        ),
    )
    with EngineSessionPool(max_open_sessions=max_open_sessions) as sessions:
        for item in plan.speech:
            if not item.preflight_required:
                continue
            segment_id = item.segment_id or item.computed_id()
            emit_render_event(
                on_event,
                _preflight_progress_event(
                    RenderEventKind.PREFLIGHT_ITEM_STARTED,
                    progress_tracker,
                    plan_id=plan_id,
                    segment_number=item.number,
                    segment_id=segment_id,
                ),
            )
            try:
                conversion = speech_request_conversion_from_renderplan(
                    item, emphasis_policy=emphasis_policy
                )
                request = conversion.request
                selection = engine_selection_from_binding(
                    item.binding, language=item.language
                )
                adapter = get_adapter(selection.engine)
                session_key = _session_compatibility_key(adapter, selection)
            except Exception as exc:  # noqa: BLE001 - Preflight must turn every binding failure into a reject check.
                check = _error_check(item, segment_id, exc)
                checks.append(check)
                progress_tracker.advance()
                emit_render_event(
                    on_event,
                    _preflight_progress_event(
                        RenderEventKind.PREFLIGHT_ITEM_FINISHED,
                        progress_tracker,
                        plan_id=plan_id,
                        segment_number=item.number,
                        segment_id=segment_id,
                        status=check.status,
                    ),
                )
                continue

            cached = sessions.acquire(session_key, selection)
            if isinstance(cached, SessionOpenFailure):
                check = _error_check(item, segment_id, cached.error)
                checks.append(check)
                progress_tracker.advance()
                emit_render_event(
                    on_event,
                    _preflight_progress_event(
                        RenderEventKind.PREFLIGHT_ITEM_FINISHED,
                        progress_tracker,
                        plan_id=plan_id,
                        segment_number=item.number,
                        segment_id=segment_id,
                        status=check.status,
                    ),
                )
                continue

            try:
                measured = cached.measure(request)
            except Exception as exc:  # noqa: BLE001 - Adapter measurement failures require centralized classification.
                outcome = classify_measurement(exc)
            else:
                outcome = classify_measurement(measured)
            check = _measurement_check(
                item,
                segment_id,
                outcome,
                deferred_capability=conversion.deferred_capability,
                deferred_control_count=conversion.deferred_control_count,
            )
            checks.append(check)
            progress_tracker.advance()
            emit_render_event(
                on_event,
                _preflight_progress_event(
                    RenderEventKind.PREFLIGHT_ITEM_FINISHED,
                    progress_tracker,
                    plan_id=plan_id,
                    segment_number=item.number,
                    segment_id=segment_id,
                    status=check.status,
                ),
            )

        expected = {
            (item.number, item.segment_id or item.computed_id())
            for item in plan.speech
            if item.preflight_required
        }
        observed = {(check.segment_number, check.segment_id) for check in checks}
        if observed != expected or len(checks) != len(expected):
            raise PlanError(
                "preflight did not produce exactly one check per required speech occurrence"
            )

        has_error = any(check.status == "error" for check in checks)
        too_long = [check for check in checks if check.status == "too_long"]
        status: Literal["ok", "replan", "reject"]
        if has_error:
            status = "reject"
            replans: tuple[ReplanRequest, ...] = ()
        elif too_long:
            status = "replan"
            speech_by_key = {
                (item.number, item.segment_id or item.computed_id()): item
                for item in plan.speech
            }
            replans = tuple(
                ReplanRequest(
                    segment_number=check.segment_number,
                    segment_id=check.segment_id,
                    source=speech_by_key[
                        (check.segment_number, check.segment_id)
                    ].source,
                    current_level=speech_by_key[
                        (check.segment_number, check.segment_id)
                    ].planning_level,
                    action="split",
                    reason="capacity",
                    metric=check.metric,
                    measured=check.measured,
                    maximum=check.maximum,
                )
                for check in too_long
            )
        else:
            status = "ok"
            replans = ()

        report = PreflightReport(
            render_plan_id=plan.plan_id or plan.computed_id(),
            source_plan_id=plan.source_plan_id,
            render_profile_id=plan.render_profile_id,
            status=status,
            checks=tuple(checks),
            replan=replans,
            producer=Producer(name="voicerender", version=_voicerender_version()),
        ).with_identity()
        report.validate_against(plan)
        save_preflight(report, output)
        emit_render_event(
            on_event,
            _preflight_progress_event(
                RenderEventKind.PREFLIGHT_FINISHED,
                progress_tracker,
                plan_id=plan_id,
                status=report.status,
            ),
        )
    return report


def preflight_exit_code(report: PreflightReport) -> int:
    """Return orchestration exit status: ok=0, replan=2, reject=1."""
    return {"ok": 0, "replan": 2, "reject": 1}[report.status]


def preflight_summary(report: PreflightReport, output: str | Path) -> dict[str, Any]:
    """Build the compact JSON summary printed by the preflight CLI."""
    return {
        "status": report.status,
        "exit_code": preflight_exit_code(report),
        "report_id": report.report_id or report.computed_id(),
        "render_plan_id": report.render_plan_id,
        "source_plan_id": report.source_plan_id,
        "render_profile_id": report.render_profile_id,
        "output": str(output),
        "checks": {
            status: sum(check.status == status for check in report.checks)
            for status in ("ok", "too_long", "error")
        },
        "replan_requests": len(report.replan),
    }


def _measurement_check(
    item: SpeechItem,
    segment_id: str,
    outcome: MeasurementOutcome,
    *,
    deferred_capability: str | None = None,
    deferred_control_count: int = 0,
) -> PreflightCheck:
    measurement = outcome.measurement
    if outcome.disposition is PreflightDisposition.OK:
        status: Literal["ok", "too_long", "error"] = "ok"
    elif outcome.disposition is PreflightDisposition.TOO_LONG:
        status = "too_long"
    else:
        status = "error"
    detail = outcome.error
    if deferred_capability is not None and deferred_control_count:
        deferred_detail = (
            f"Deferred {deferred_control_count} emphasis control(s) to "
            f"{deferred_capability}; pending composition (not applied by VoiceRender)."
        )
        detail = f"{detail}; {deferred_detail}" if detail else deferred_detail
    return PreflightCheck(
        segment_number=item.number,
        segment_id=segment_id,
        status=status,
        metric=measurement.unit if measurement is not None else "unknown",
        measured=measurement.amount if measurement is not None else None,
        maximum=measurement.maximum if measurement is not None else None,
        source=(
            measurement.source
            if measurement is not None and measurement.source.strip()
            else "voicerender.preflight"
        ),
        detail=detail,
    )


def _error_check(
    item: SpeechItem,
    segment_id: str,
    error: Exception,
    *,
    source: str | None = None,
) -> PreflightCheck:
    return PreflightCheck(
        segment_number=item.number,
        segment_id=segment_id,
        status="error",
        metric="unknown",
        measured=None,
        maximum=None,
        source=source or _error_source(error),
        detail=str(error) or "RenderPlan preflight rejected this speech occurrence",
    )


def _session_compatibility_key(adapter: Any, selection: EngineSelection) -> str:
    """Key frontend reuse by adapter API and every synthesis-relevant selection fact."""
    probe = adapter.probe_api()
    identity = {
        "engine": adapter.id,
        "engine_version": adapter.version() or probe.distribution_version,
        "api_probe": asdict(probe),
        "target_id": selection.target_id,
        "language": selection.language,
        "voice": selection.voice,
        "speaker": selection.speaker,
        "options": dict(selection.options),
        "offline": selection.offline,
        "model_revision": selection.metadata.get("model_revision"),
        "engine_identity": dict(adapter.canonical_synthesis_identity(selection)),
    }
    try:
        return json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PlanError(
            f"Renderer session identity is not serializable: {exc}"
        ) from exc


def _error_source(error: Exception) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str) and code.strip():
        return code
    return "voicerender.preflight"


def _voicerender_version() -> str:
    try:
        return version("voicerender")
    except PackageNotFoundError:
        return "0+unknown"


__all__ = [
    "preflight_exit_code",
    "preflight_render_plan",
    "preflight_summary",
]
