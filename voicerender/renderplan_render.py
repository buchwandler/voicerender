"""Exact execution of a preflight-approved shared RenderPlan."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Collection, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from renderplan import (
    PreflightCheck,
    PreflightReport,
    Producer,
    RenderManifest,
    RenderPlan,
    ReplanRequest,
    SpeechArtifact,
    SpeechItem,
    load_manifest,
    load_plan,
    load_preflight,
    save_manifest,
    save_preflight,
)

from .engines.base import EngineSelection, EngineSession, SpeechRequest
from .engines.registry import get_adapter, open_session
from .errors import ArtifactError, PlanError, UnsupportedSynthesisFeatureError
from .preflight import _session_compatibility_key
from .renderplan_input import (
    EMPHASIS_COMPOSITION_CAPABILITY,
    EmphasisPolicy,
    MeasurementOutcome,
    PreflightDisposition,
    SpeechRequestConversion,
    classify_measurement,
    engine_selection_from_binding,
    speech_request_conversion_from_renderplan,
)
from .storage import atomic_json, atomic_wav, digest, verified_wav, wav_info

RENDER_MANIFEST_FILENAME = "render-manifest.toml"
RENDER_DIAGNOSTICS_FILENAME = "render-diagnostics.json"
_CACHE_SCHEMA = "voicerender.renderplan-cache.v1"


@dataclass(frozen=True, slots=True)
class RenderPlanResult:
    manifest: RenderManifest
    manifest_path: Path
    planned_count: int
    selected_count: int
    available_count: int
    rendered_count: int
    reused_count: int
    selection_complete: bool
    error: str | None = None
    fresh_preflight: PreflightReport | None = None
    error_code: str | None = None
    execution_fingerprint: tuple[dict[str, Any], ...] = ()
    pending_composition: bool = False

    @property
    def failed(self) -> bool:
        return self.error is not None or not self.selection_complete


@dataclass(frozen=True, slots=True)
class _PreparedSpeech:
    item: SpeechItem
    request: SpeechRequest
    selection: EngineSelection
    segment_id: str
    cache_key: str
    session_key: str
    execution_fingerprint: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _OpenFailure:
    error: Exception


@dataclass(frozen=True, slots=True)
class _RenderFailure:
    message: str
    fresh_preflight: PreflightReport | None = None
    error_code: str = "render.failed"


def render_renderplan(
    plan: RenderPlan | str | Path,
    *,
    preflight: PreflightReport | str | Path,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    force: bool = False,
    reset_stale_output: bool = False,
    recheck_report_path: str | Path | None = None,
    speech_numbers: Collection[int] | None = None,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
) -> RenderPlanResult:
    """Render selected speech items, retaining verified artifacts from prior runs."""
    parsed_plan = load_plan(plan) if isinstance(plan, (str, Path)) else plan
    if not isinstance(parsed_plan, RenderPlan):
        raise PlanError("render input must be a RenderPlan")
    selected_items = _select_speech_items(parsed_plan, speech_numbers)
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

    try:
        policy = EmphasisPolicy(emphasis_policy)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "emphasis_policy must be 'reject' or 'defer-to-compose'"
        ) from exc
    emphasis_items = tuple(
        item
        for item in parsed_plan.speech
        if any(control.name == "emphasis" for control in item.controls)
    )
    if emphasis_items and policy is not EmphasisPolicy.DEFER_TO_COMPOSE:
        raise UnsupportedSynthesisFeatureError(
            "This RenderPlan contains emphasis controls; explicitly select "
            "emphasis_policy='defer-to-compose' and declare "
            f"{EMPHASIS_COMPOSITION_CAPABILITY}"
        )
    for item in emphasis_items:
        if not item.preflight_required:
            continue
        key = (item.number, item.segment_id or item.computed_id())
        check = checked.get(key)
        if check is None or EMPHASIS_COMPOSITION_CAPABILITY not in (check.detail or ""):
            raise PlanError(
                f"Preflight did not confirm deferred emphasis for speech occurrence {key}"
            )
    conversions = {
        (
            item.number,
            item.segment_id or item.computed_id(),
        ): speech_request_conversion_from_renderplan(item, emphasis_policy=policy)
        for item in selected_items
    }
    deferred_controls = [
        {
            "segment_number": item.number,
            "segment_id": item.segment_id or item.computed_id(),
            "capability": EMPHASIS_COMPOSITION_CAPABILITY,
            "count": sum(control.name == "emphasis" for control in item.controls),
        }
        for item in emphasis_items
        if policy is EmphasisPolicy.DEFER_TO_COMPOSE
    ]
    diagnostics = {
        "schema": "voicerender.render-diagnostics.v1",
        "render_plan_id": parsed_plan.plan_id or parsed_plan.computed_id(),
        "emphasis_policy": policy.value,
        "pending_composition": bool(deferred_controls),
        "deferred_controls": deferred_controls,
        "warnings": (
            [
                "Any WAVs produced contain unchanged speech; emphasis remains pending composition."
            ]
            if deferred_controls
            else []
        ),
    }
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

    artifact_map: dict[tuple[int, str], SpeechArtifact] = {}
    rendered = reused = 0
    failure: _RenderFailure | None = None
    execution_fingerprints: dict[tuple[int, str], dict[str, Any]] = {}
    with _exclusive_render(out):
        cache.mkdir(parents=True, exist_ok=True)
        had_preflight = (out / "preflight.toml").exists() or (
            out / "preflight.toml"
        ).is_symlink()
        artifact_map, stale_output_reset = _load_existing_artifacts(
            parsed_plan,
            out,
            manifest_path,
            report_id,
            force=force,
            reset_stale_output=reset_stale_output,
        )
        if stale_output_reset and reset_stale_output and had_preflight:
            save_preflight(report, out / "preflight.toml")
        atomic_json(out / RENDER_DIAGNOSTICS_FILENAME, diagnostics)
        selected_keys = {
            (item.number, item.segment_id or item.computed_id())
            for item in selected_items
        }
        if not force:
            reused = sum(key in artifact_map for key in selected_keys)
        pending_items = tuple(
            item
            for item in selected_items
            if force
            or (item.number, item.segment_id or item.computed_id()) not in artifact_map
        )
        prepared = tuple(
            _prepare_speech(
                item,
                emphasis_policy=policy,
                conversion=conversions[
                    (item.number, item.segment_id or item.computed_id())
                ],
            )
            for item in pending_items
        )
        current_artifacts = _ordered_artifacts(parsed_plan, artifact_map)
        _persist_manifest(
            parsed_plan,
            manifest_path,
            report_id,
            current_artifacts,
            complete=len(current_artifacts) == len(parsed_plan.speech),
            producer=producer,
        )
        with ExitStack() as opened:
            sessions: dict[str, EngineSession | _OpenFailure] = {}
            for current in prepared:
                item = current.item
                key = (item.number, current.segment_id)
                target = out / item.wav
                fingerprint = dict(current.execution_fingerprint)
                fingerprint["capacity"] = None
                fingerprint["execution_status"] = "not_measured"
                try:
                    cached = None if force else _read_cache(cache, current.cache_key)
                    if cached is not None:
                        source, info = cached
                        _copy_verified(source, target, info)
                        reused += 1
                        fingerprint["execution_status"] = "cache_reused"
                    else:
                        fingerprint["execution_status"] = "synthesized"
                        session = sessions.get(current.session_key)
                        if session is None:
                            try:
                                session = opened.enter_context(
                                    open_session(current.selection)
                                )
                            except Exception as exc:  # noqa: BLE001 - Renderer setup becomes a partial-render result.
                                session = _OpenFailure(exc)
                            sessions[current.session_key] = session
                        if isinstance(session, _OpenFailure):
                            fingerprint["execution_status"] = "session_open_failed"
                            execution_fingerprints[key] = (
                                _finalize_execution_fingerprint(fingerprint)
                            )
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
                                error_code=(
                                    "preflight.reject"
                                    if item.preflight_required
                                    else "engine.session_failed"
                                ),
                            )
                            break

                        try:
                            measured = session.measure(current.request)
                        except Exception as exc:  # noqa: BLE001 - Capacity/runtime errors are centrally classified.
                            outcome = classify_measurement(exc)
                        else:
                            outcome = classify_measurement(measured)
                        fingerprint["capacity"] = _measurement_fingerprint(outcome)
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
                            fingerprint["execution_status"] = "remeasurement_failed"
                            execution_fingerprints[key] = (
                                _finalize_execution_fingerprint(fingerprint)
                            )
                            failure = _RenderFailure(
                                outcome.error
                                or "Immediate renderer remeasurement did not return an exact fit",
                                fresh_preflight=fresh,
                                error_code=(
                                    "preflight.replan"
                                    if fresh is not None and fresh.status == "replan"
                                    else "preflight.reject"
                                    if fresh is not None
                                    else "engine.measurement_failed"
                                ),
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
                    artifact_map[key] = SpeechArtifact(
                        segment_number=item.number,
                        segment_id=current.segment_id,
                        wav=item.wav,
                        sha256=f"sha256:{final_info['sha256']}",
                        frames=int(final_info["frames"]),
                        sample_rate=int(final_info["sample_rate"]),
                        channels=int(final_info["channels"]),
                    )
                    current_artifacts = _ordered_artifacts(parsed_plan, artifact_map)
                    _persist_manifest(
                        parsed_plan,
                        manifest_path,
                        report_id,
                        current_artifacts,
                        complete=len(current_artifacts) == len(parsed_plan.speech),
                        producer=producer,
                    )
                    execution_fingerprints[key] = _finalize_execution_fingerprint(
                        fingerprint
                    )
                except Exception as exc:  # noqa: BLE001 - Persist a valid cumulative partial manifest.
                    error_code = getattr(exc, "code", None)
                    failure = _RenderFailure(
                        str(exc) or "Render failed",
                        error_code=(
                            error_code
                            if isinstance(error_code, str)
                            else "render.failed"
                        ),
                    )
                    fingerprint["execution_status"] = "render_failed"
                    execution_fingerprints[key] = _finalize_execution_fingerprint(
                        fingerprint
                    )
                    break

        if (
            failure is not None
            and failure.fresh_preflight is not None
            and recheck_report_path is not None
        ):
            try:
                save_preflight(failure.fresh_preflight, recheck_report_path)
            except Exception as exc:  # noqa: BLE001 - Keep the generated report available in the result.
                failure = _RenderFailure(
                    f"Cannot persist fresh preflight report {recheck_report_path}: {exc}",
                    fresh_preflight=failure.fresh_preflight,
                    error_code="preflight.report_write_failed",
                )
        current_artifacts = _ordered_artifacts(parsed_plan, artifact_map)
        manifest = _persist_manifest(
            parsed_plan,
            manifest_path,
            report_id,
            current_artifacts,
            complete=len(current_artifacts) == len(parsed_plan.speech),
            producer=producer,
        )

    selected_complete = all(
        (item.number, item.segment_id or item.computed_id()) in artifact_map
        for item in selected_items
    )
    return RenderPlanResult(
        manifest=manifest,
        manifest_path=manifest_path,
        planned_count=len(parsed_plan.speech),
        selected_count=len(selected_items),
        available_count=len(manifest.speech),
        rendered_count=rendered,
        reused_count=reused,
        selection_complete=selected_complete,
        error=failure.message if failure is not None else None,
        fresh_preflight=failure.fresh_preflight if failure is not None else None,
        error_code=failure.error_code if failure is not None else None,
        execution_fingerprint=tuple(execution_fingerprints.values()),
        pending_composition=bool(deferred_controls),
    )


def _select_speech_items(
    plan: RenderPlan, speech_numbers: Collection[int] | None
) -> tuple[SpeechItem, ...]:
    if speech_numbers is None:
        speech_items: tuple[SpeechItem, ...] = plan.speech
        return speech_items
    try:
        requested = tuple(speech_numbers)
    except TypeError as exc:
        raise PlanError(
            "speech_numbers must be a collection of positive integers"
        ) from exc
    if not requested:
        raise PlanError("speech_numbers selection must not be empty")
    if any(type(number) is not int or number <= 0 for number in requested):
        raise PlanError("speech_numbers must contain only positive integers")
    requested_set = set(requested)
    planned_numbers = {item.number for item in plan.speech}
    unknown = requested_set - planned_numbers
    if unknown:
        raise PlanError(
            f"speech_numbers contains unknown plan occurrence(s): {sorted(unknown)}"
        )
    return tuple(item for item in plan.speech if item.number in requested_set)


def _load_existing_artifacts(
    plan: RenderPlan,
    output_dir: Path,
    manifest_path: Path,
    plan_id: str,
    *,
    force: bool,
    reset_stale_output: bool,
) -> tuple[dict[tuple[int, str], SpeechArtifact], bool]:
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return {}, False
    if manifest_path.is_symlink():
        if not force:
            raise ArtifactError(
                "Existing render manifest must not be a symlink",
                code="artifact.invalid",
            )
        _reset_owned_artifacts(output_dir)
        return {}, True
    try:
        manifest = load_manifest(manifest_path)
        if manifest.render_plan_id != plan_id:
            if not (force or reset_stale_output):
                raise ArtifactError(
                    "Stale render output belongs to a different RenderPlan; "
                    "use --reset-stale or --force to reset it",
                    code="output.stale_plan",
                )
            _reset_owned_artifacts(output_dir)
            return {}, True
        manifest.validate_against(plan)
        expected = {
            (item.number, item.segment_id or item.computed_id()): item
            for item in plan.speech
        }
        verified: dict[tuple[int, str], SpeechArtifact] = {}
        for artifact in manifest.speech:
            key = (artifact.segment_number, artifact.segment_id)
            if key in verified or key not in expected:
                raise ArtifactError(
                    "Existing render manifest has duplicate or unknown speech entries"
                )
            item = expected[key]
            if artifact.wav != item.wav:
                raise ArtifactError(
                    "Existing render manifest WAV name does not match the plan"
                )
            metadata = {
                "sha256": artifact.sha256.removeprefix("sha256:"),
                "frames": artifact.frames,
                "sample_rate": artifact.sample_rate,
                "channels": artifact.channels,
            }
            if verified_wav(output_dir / artifact.wav, metadata):
                verified[key] = artifact
        return verified, False
    except ArtifactError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        if not force:
            raise ArtifactError(
                f"Existing render manifest is invalid: {exc}", code="artifact.invalid"
            ) from exc
        _reset_owned_artifacts(output_dir)
        return {}, True


def _reset_owned_artifacts(output_dir: Path) -> None:
    """Remove only VoiceRender's well-known chapter output files."""
    for child in output_dir.iterdir():
        owned = child.name in {
            RENDER_MANIFEST_FILENAME,
            RENDER_DIAGNOSTICS_FILENAME,
            "preflight.toml",
            "preflight-recheck.toml",
        } or bool(re.fullmatch(r"\d{6,}\.wav", child.name))
        if not owned:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.exists():
            raise ArtifactError(f"Cannot reset owned artifact directory: {child}")


def _ordered_artifacts(
    plan: RenderPlan, artifacts: dict[tuple[int, str], SpeechArtifact]
) -> list[SpeechArtifact]:
    return [
        artifacts[(item.number, item.segment_id or item.computed_id())]
        for item in plan.speech
        if (item.number, item.segment_id or item.computed_id()) in artifacts
    ]


def _prepare_speech(
    item: SpeechItem,
    *,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
    conversion: SpeechRequestConversion | None = None,
) -> _PreparedSpeech:
    try:
        if conversion is None:
            conversion = speech_request_conversion_from_renderplan(
                item, emphasis_policy=emphasis_policy
            )
        request = conversion.request
        selection = engine_selection_from_binding(item.binding, language=item.language)
        adapter = get_adapter(selection.engine)
        session_key = _session_compatibility_key(adapter, selection)
        adapter_version = adapter.version()
        api_probe = asdict(adapter.probe_api())
        synthesis_identity = dict(adapter.canonical_synthesis_identity(selection))
        identity = {
            "schema": _CACHE_SCHEMA,
            "segment_id": item.segment_id or item.computed_id(),
            "engine": adapter.id,
            "engine_version": adapter_version,
            "api_probe": api_probe,
            "engine_identity": synthesis_identity,
            "language": selection.language,
            "voice": selection.voice,
            "speaker": selection.speaker,
            "options": dict(selection.options),
            "offline": selection.offline,
        }
        execution_fingerprint = {
            "voicerender_version": _voicerender_version(),
            "adapter_id": adapter.id,
            "adapter_version": adapter_version,
            "api_probe_signature": digest(api_probe),
            "canonical_synthesis_identity": synthesis_identity,
            "model": selection.target_id,
            "language": selection.language,
            "voice": selection.voice,
            "speaker": selection.speaker,
            "options": dict(selection.options),
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
        execution_fingerprint=execution_fingerprint,
    )


def _measurement_fingerprint(outcome: MeasurementOutcome) -> dict[str, Any] | None:
    measurement = outcome.measurement
    if measurement is None:
        return None
    fingerprint = {
        "metric": measurement.unit,
        "measured": measurement.amount,
        "maximum": measurement.maximum,
        "source": measurement.source,
    }
    policy_revision = measurement.details.get("capacity_policy_revision")
    if isinstance(policy_revision, str) and policy_revision.strip():
        fingerprint["capacity_policy_revision"] = policy_revision.strip()
    return fingerprint


def _finalize_execution_fingerprint(fingerprint: dict[str, Any]) -> dict[str, Any]:
    result = dict(fingerprint)
    result["fingerprint_id"] = f"sha256:{digest(fingerprint)}"
    return result


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
    status: Literal["ok", "too_long", "error"]
    report_status: Literal["ok", "replan", "reject"]
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
    "RENDER_DIAGNOSTICS_FILENAME",
    "RENDER_MANIFEST_FILENAME",
    "RenderPlanResult",
    "render_renderplan",
]
