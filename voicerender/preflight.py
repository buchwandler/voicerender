"""Persisted RenderPlan preflight without waveform synthesis."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from renderplan import (
    PreflightCheck,
    PreflightReport,
    Producer,
    RenderPlan,
    ReplanRequest,
    SpeechItem,
    save_preflight,
)

from .engines.base import EngineSelection, EngineSession
from .engines.registry import get_adapter, open_session
from .errors import PlanError
from .renderplan_input import (
    EmphasisPolicy,
    MeasurementOutcome,
    PreflightDisposition,
    classify_measurement,
    engine_selection_from_binding,
    speech_request_conversion_from_renderplan,
)


@dataclass(frozen=True, slots=True)
class _OpenFailure:
    source: str
    detail: str


def preflight_render_plan(
    plan: RenderPlan,
    output: str | Path,
    *,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
) -> PreflightReport:
    """Measure each required speech occurrence, then validate and persist its report."""
    if not isinstance(plan, RenderPlan):
        raise PlanError("preflight input must be a RenderPlan")
    checks: list[PreflightCheck] = []
    sessions: dict[str, EngineSession | _OpenFailure] = {}

    with ExitStack() as stack:
        for item in plan.speech:
            if not item.preflight_required:
                continue
            segment_id = item.segment_id or item.computed_id()
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
                checks.append(_error_check(item, segment_id, exc))
                continue

            cached = sessions.get(session_key)
            if cached is None:
                try:
                    cached = stack.enter_context(open_session(selection))
                except Exception as exc:  # noqa: BLE001 - Session startup failures are persisted as reject checks.
                    cached = _OpenFailure(
                        source=_error_source(exc),
                        detail=str(exc) or "Cannot open renderer session",
                    )
                sessions[session_key] = cached
            if isinstance(cached, _OpenFailure):
                checks.append(
                    _error_check(
                        item,
                        segment_id,
                        RuntimeError(cached.detail),
                        source=cached.source,
                    )
                )
                continue

            try:
                measured = cached.measure(request)
            except Exception as exc:  # noqa: BLE001 - Adapter measurement failures require centralized classification.
                outcome = classify_measurement(exc)
            else:
                outcome = classify_measurement(measured)
            checks.append(
                _measurement_check(
                    item,
                    segment_id,
                    outcome,
                    deferred_capability=conversion.deferred_capability,
                    deferred_control_count=conversion.deferred_control_count,
                )
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
    status = {
        PreflightDisposition.OK: "ok",
        PreflightDisposition.TOO_LONG: "too_long",
        PreflightDisposition.REJECT: "error",
    }[outcome.disposition]
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
