"""Exact execution of a preflight-approved shared RenderPlan."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from renderplan import (
    PreflightCheck,
    PreflightReport,
    Producer,
    RenderManifest,
    RenderPlan,
    ReplanRequest,
    SpeechArtifact,
    SpeechItem,
    load_plan,
    load_preflight,
    save_manifest,
    save_preflight,
)

from .engines.base import EngineSelection, EngineSession, SpeechRequest
from .engines.registry import get_adapter, open_session
from .errors import ArtifactError, PlanError
from .preflight import _session_compatibility_key
from .renderplan_input import (
    MeasurementOutcome,
    PreflightDisposition,
    classify_measurement,
    engine_selection_from_binding,
    speech_request_from_renderplan,
)
from .storage import atomic_json, atomic_wav, digest, verified_wav, wav_info

RENDER_MANIFEST_FILENAME = "render-manifest.toml"
_CACHE_SCHEMA = "voicerender.renderplan-cache.v1"


@dataclass(frozen=True, slots=True)
class RenderPlanResult:
    manifest: RenderManifest
    manifest_path: Path
    rendered_count: int
    reused_count: int
    error: str | None = None
    fresh_preflight: PreflightReport | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None or not self.manifest.complete


@dataclass(frozen=True, slots=True)
class _PreparedSpeech:
    item: SpeechItem
    request: SpeechRequest
    selection: EngineSelection
    segment_id: str
    cache_key: str
    session_key: str


@dataclass(frozen=True, slots=True)
class _OpenFailure:
    error: Exception


@dataclass(frozen=True, slots=True)
class _RenderFailure:
    message: str
    fresh_preflight: PreflightReport | None = None


def render_renderplan(
    plan: RenderPlan | str | Path,
    *,
    preflight: PreflightReport | str | Path,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    force: bool = False,
    recheck_report_path: str | Path | None = None,
) -> RenderPlanResult:
    """Render one WAV per SpeechItem, returning a complete or valid partial manifest."""
    parsed_plan = load_plan(plan) if isinstance(plan, (str, Path)) else plan
    if not isinstance(parsed_plan, RenderPlan):
        raise PlanError("render input must be a RenderPlan")
    report = (
        load_preflight(preflight) if isinstance(preflight, (str, Path)) else preflight
    )
    if not isinstance(report, PreflightReport):
        raise PlanError("render requires a shared RenderPlan preflight report")
    if report.status != "ok":
        raise PlanError(
            f"RenderPlan preflight status must be 'ok', got {report.status!r}"
        )
    try:
        report.validate_against(parsed_plan)
    except ValueError as exc:
        raise PlanError(
            f"RenderPlan preflight report does not match the plan: {exc}"
        ) from exc
    checked = {
        (check.segment_number, check.segment_id): check for check in report.checks
    }
    for item in parsed_plan.speech:
        if not item.preflight_required:
            continue
        key = (item.number, item.segment_id or item.computed_id())
        check = checked.get(key)
        if check is None or check.status != "ok":
            raise PlanError(
                f"Speech occurrence {key} does not have a matching OK preflight check"
            )

    prepared = tuple(_prepare_speech(item) for item in parsed_plan.speech)
    out = Path(output_dir).expanduser().resolve()
    cache = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else out.parent / ".voicerender-cache"
    )
    if cache == out or cache.is_relative_to(out) or out.is_relative_to(cache):
        raise ArtifactError("cache_dir must be outside the RenderPlan output directory")
    manifest_path = out / RENDER_MANIFEST_FILENAME
    report_id = parsed_plan.plan_id or parsed_plan.computed_id()
    producer = Producer(name="voicerender", version=_voicerender_version())

    artifacts: list[SpeechArtifact] = []
    rendered = reused = 0
    failure: _RenderFailure | None = None
    with _exclusive_render(out):
        cache.mkdir(parents=True, exist_ok=True)
        _persist_manifest(
            parsed_plan,
            manifest_path,
            report_id,
            artifacts,
            complete=False,
            producer=producer,
        )
        with ExitStack() as opened:
            sessions: dict[str, EngineSession | _OpenFailure] = {}
            for current in prepared:
                item = current.item
                target = out / item.wav
                try:
                    cached = None if force else _read_cache(cache, current.cache_key)
                    if cached is not None:
                        source, info = cached
                        _copy_verified(source, target, info)
                        reused += 1
                    else:
                        session = sessions.get(current.session_key)
                        if session is None:
                            try:
                                session = opened.enter_context(
                                    open_session(current.selection)
                                )
                            except Exception as exc:  # noqa: BLE001 - Renderer setup must become a partial-render result.
                                session = _OpenFailure(exc)
                            sessions[current.session_key] = session
                        if isinstance(session, _OpenFailure):
                            failure = _RenderFailure(
                                f"Cannot open renderer session: {session.error}",
                                fresh_preflight=(
                                    _rejected_report(
                                        parsed_plan,
                                        item,
                                        current.segment_id,
                                        session.error,
                                    )
                                    if item.preflight_required
                                    else None
                                ),
                            )
                            break

                        try:
                            measured = session.measure(current.request)
                        except Exception as exc:  # noqa: BLE001 - Capacity/runtime errors are centrally classified.
                            outcome = classify_measurement(exc)
                        else:
                            outcome = classify_measurement(measured)
                        if outcome.disposition is not PreflightDisposition.OK:
                            fresh = (
                                _failure_report(
                                    parsed_plan,
                                    item,
                                    current.segment_id,
                                    outcome,
                                )
                                if item.preflight_required
                                else None
                            )
                            if recheck_report_path is not None and fresh is not None:
                                save_preflight(fresh, recheck_report_path)
                            failure = _RenderFailure(
                                outcome.error
                                or "Immediate renderer remeasurement did not return an exact fit",
                                fresh_preflight=fresh,
                            )
                            break

                        rendered_speech = session.synthesize(current.request)
                        if rendered_speech.id != current.request.id:
                            raise ArtifactError(
                                "Renderer returned an artifact for a different speech request"
                            )
                        info = atomic_wav(
                            cache / f"{current.cache_key}.wav",
                            rendered_speech.audio,
                            rendered_speech.sample_rate,
                        )
                        atomic_json(
                            cache / f"{current.cache_key}.json",
                            {
                                "schema": _CACHE_SCHEMA,
                                "key": current.cache_key,
                                "wav": info,
                            },
                        )
                        if not verified_wav(cache / f"{current.cache_key}.wav", info):
                            raise ArtifactError(
                                "New cache WAV failed integrity validation"
                            )
                        _copy_verified(cache / f"{current.cache_key}.wav", target, info)
                        rendered += 1

                    final_info = wav_info(target)
                    if not verified_wav(target, final_info):
                        raise ArtifactError(
                            f"Materialized WAV failed integrity validation: {target}"
                        )
                    artifacts.append(
                        SpeechArtifact(
                            segment_number=item.number,
                            segment_id=current.segment_id,
                            wav=item.wav,
                            sha256=f"sha256:{final_info['sha256']}",
                            frames=int(final_info["frames"]),
                            sample_rate=int(final_info["sample_rate"]),
                            channels=int(final_info["channels"]),
                        )
                    )
                    _persist_manifest(
                        parsed_plan,
                        manifest_path,
                        report_id,
                        artifacts,
                        complete=False,
                        producer=producer,
                    )
                except Exception as exc:  # noqa: BLE001 - Artifact and backend failures persist a valid partial manifest.
                    failure = _RenderFailure(str(exc) or "Render failed")
                    break

        complete = failure is None and len(artifacts) == len(parsed_plan.speech)
        manifest = _persist_manifest(
            parsed_plan,
            manifest_path,
            report_id,
            artifacts,
            complete=complete,
            producer=producer,
        )
    return RenderPlanResult(
        manifest=manifest,
        manifest_path=manifest_path,
        rendered_count=rendered,
        reused_count=reused,
        error=failure.message if failure is not None else None,
        fresh_preflight=failure.fresh_preflight if failure is not None else None,
    )


def _prepare_speech(item: SpeechItem) -> _PreparedSpeech:
    try:
        request = speech_request_from_renderplan(item)
        selection = engine_selection_from_binding(item.binding, language=item.language)
        adapter = get_adapter(selection.engine)
        session_key = _session_compatibility_key(adapter, selection)
        identity = {
            "schema": _CACHE_SCHEMA,
            "segment_id": item.segment_id or item.computed_id(),
            "engine": adapter.id,
            "engine_version": adapter.version(),
            "api_probe": asdict(adapter.probe_api()),
            "engine_identity": dict(adapter.canonical_synthesis_identity(selection)),
            "language": selection.language,
            "voice": selection.voice,
            "speaker": selection.speaker,
            "options": dict(selection.options),
            "offline": selection.offline,
        }
        cache_key = digest(identity)
    except Exception as exc:
        raise PlanError(
            f"Cannot prepare speech occurrence {item.number} before rendering: {exc}"
        ) from exc
    return _PreparedSpeech(
        item=item,
        request=request,
        selection=selection,
        segment_id=item.segment_id or item.computed_id(),
        cache_key=cache_key,
        session_key=session_key,
    )


def _read_cache(cache_dir: Path, key: str) -> tuple[Path, dict[str, Any]] | None:
    wav = cache_dir / f"{key}.wav"
    sidecar = cache_dir / f"{key}.json"
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(data, dict)
        or data.get("schema") != _CACHE_SCHEMA
        or data.get("key") != key
    ):
        return None
    info = data.get("wav")
    if not isinstance(info, dict) or not verified_wav(wav, info):
        return None
    return wav, info


def _copy_verified(source: Path, target: Path, info: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        if not verified_wav(temporary, info):
            raise ArtifactError(f"WAV copy failed integrity validation: {source}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _persist_manifest(
    plan: RenderPlan,
    path: Path,
    plan_id: str,
    artifacts: list[SpeechArtifact],
    *,
    complete: bool,
    producer: Producer,
) -> RenderManifest:
    manifest = RenderManifest(
        render_plan_id=plan_id,
        speech=tuple(artifacts),
        complete=complete,
        producer=producer,
    ).with_identity()
    manifest.validate_against(plan)
    save_manifest(manifest, path)
    return manifest


def _failure_report(
    plan: RenderPlan,
    item: SpeechItem,
    segment_id: str,
    outcome: MeasurementOutcome,
) -> PreflightReport:
    measurement = outcome.measurement
    if outcome.disposition is PreflightDisposition.TOO_LONG:
        status = "too_long"
        report_status = "replan"
    else:
        status = "error"
        report_status = "reject"
    check = PreflightCheck(
        segment_number=item.number,
        segment_id=segment_id,
        status=status,
        metric=measurement.unit if measurement is not None else "unknown",
        measured=measurement.amount if measurement is not None else None,
        maximum=measurement.maximum if measurement is not None else None,
        source=(
            measurement.source
            if measurement is not None
            and isinstance(measurement.source, str)
            and measurement.source.strip()
            else "voicerender.render.remeasurement"
        ),
        detail=outcome.error,
    )
    requests: tuple[ReplanRequest, ...] = ()
    if report_status == "replan":
        requests = (
            ReplanRequest(
                segment_number=item.number,
                segment_id=segment_id,
                source=item.source,
                current_level=item.planning_level,
                action="split",
                reason="capacity",
                metric=check.metric,
                measured=check.measured,
                maximum=check.maximum,
            ),
        )
    report = PreflightReport(
        render_plan_id=plan.plan_id or plan.computed_id(),
        source_plan_id=plan.source_plan_id,
        render_profile_id=plan.render_profile_id,
        status=report_status,
        checks=(check,),
        replan=requests,
        producer=Producer(name="voicerender", version=_voicerender_version()),
    ).with_identity()
    report.validate_against(plan)
    return report


def _rejected_report(
    plan: RenderPlan,
    item: SpeechItem,
    segment_id: str,
    error: Exception,
) -> PreflightReport:
    raw_code = getattr(error, "code", None)
    source = (
        raw_code
        if isinstance(raw_code, str) and raw_code.strip()
        else "voicerender.render.session"
    )
    check = PreflightCheck(
        segment_number=item.number,
        segment_id=segment_id,
        status="error",
        metric="unknown",
        measured=None,
        maximum=None,
        source=source,
        detail=str(error) or "Renderer session could not be opened",
    )
    report = PreflightReport(
        render_plan_id=plan.plan_id or plan.computed_id(),
        source_plan_id=plan.source_plan_id,
        render_profile_id=plan.render_profile_id,
        status="reject",
        checks=(check,),
        producer=Producer(name="voicerender", version=_voicerender_version()),
    ).with_identity()
    report.validate_against(plan)
    return report


def _voicerender_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("voicerender")
    except PackageNotFoundError:
        return "0+unknown"


@contextmanager
def _exclusive_render(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".voicerender.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ArtifactError(f"Render directory is locked: {output_dir}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


__all__ = [
    "RENDER_MANIFEST_FILENAME",
    "RenderPlanResult",
    "render_renderplan",
]
