from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from renderplan import (
    PreflightCheck,
    PreflightReport,
    Producer,
    RenderManifest,
    RenderPlan,
    RenderPlanSet,
    RenderPlanSetChapter,
    RenderProfile,
    ReplanRequest,
    RoleProfile,
    SourceSpan,
    SpeechArtifact,
    SpeechControl,
    SpeechItem,
    load_manifest,
    load_plan,
    save_manifest,
    save_plan,
    save_preflight,
    save_profile,
    save_renderplan_set,
)

import voicerender.book as book_module
from voicerender import (
    BookRenderError,
    BookRenderRequest,
    BookRenderResult,
    ChapterRenderOutcome,
)
from voicerender.book import render_book
from voicerender.book_contract import load_renderplan_set
from voicerender.errors import ArtifactError
from voicerender.events import (
    RenderEvent,
    RenderEventKind,
    RenderScope,
    emit_render_event,
)
from voicerender.progress import ProgressTracker
from voicerender.renderplan_render import (
    RENDER_DIAGNOSTICS_FILENAME,
    RENDER_MANIFEST_FILENAME,
    RenderPlanResult,
)
from voicerender.storage import atomic_wav


def _profile() -> RenderProfile:
    return RenderProfile(
        roles={
            "narrator": RoleProfile(language="en-US", binding={"renderer": "fixture"})
        },
        default_role="narrator",
    ).with_identity()


def _plan(profile: RenderProfile, chapter_number: int, count: int) -> RenderPlan:
    profile_id = profile.profile_id or profile.computed_id()
    speech = tuple(
        SpeechItem(
            number=number,
            text=f"chapter {chapter_number} speech {number}",
            language="en-US",
            role="narrator",
            binding={"renderer": "fixture"},
            source=(
                SourceSpan(
                    locator=f"chapter-{chapter_number}.txt",
                    start=number,
                    end=number + 1,
                ),
            ),
        )
        for number in range(1, count + 1)
    )
    return RenderPlan(
        source_plan_id=f"sha256:{chapter_number:064d}",
        render_profile_id=profile_id,
        flow=speech,
    ).with_identity()


def _book(
    tmp_path: Path,
    chapters: tuple[tuple[int, int], ...] = ((7, 2), (5, 3), (8, 1)),
    emphasis_speech: int | None = None,
) -> tuple[Path, str]:
    workspace = tmp_path / "sample.ssmd-bundle"
    workspace.mkdir(parents=True)
    (workspace / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}),
        encoding="utf-8",
    )
    variant_name = "narration--"
    profile = _profile()
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant_name += fingerprint
    variant_dir = workspace / "renderplan" / variant_name
    variant_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")

    entries: list[RenderPlanSetChapter] = []
    for source_number, speech_count in chapters:
        chapter_id = f"chapter-{source_number:04d}"
        plan = _plan(profile, source_number, speech_count)
        if emphasis_speech is not None:
            plan = replace(
                plan,
                flow=tuple(
                    replace(
                        item,
                        segment_id=None,
                        controls=(
                            SpeechControl(start=0, end=3, name="emphasis", params={}),
                        ),
                    )
                    if item.number == emphasis_speech
                    else item
                    for item in plan.flow
                ),
                plan_id=None,
            ).with_identity()
        plan_name = f"{chapter_id}.renderplan.toml"
        save_plan(plan, variant_dir / plan_name)
        entries.append(
            RenderPlanSetChapter(
                id=chapter_id,
                source_number=source_number,
                source=f"../../chapters/{chapter_id}.md",
                utterplan=f"../../utterplan/{chapter_id}.toml",
                source_plan_id=plan.source_plan_id,
                renderplan=plan_name,
                render_plan_id=plan.plan_id or plan.computed_id(),
            )
        )
    renderplan_set = RenderPlanSet(
        variant="narration",
        profile_fingerprint=fingerprint,
        producer_version="0.1.0",
        manifest="../../book.json",
        utterplan_dir="../../utterplan",
        profile_id=profile_id,
        snapshot="renderprofile.toml",
        source_profile="../../narration.renderprofile.toml",
        chapters=tuple(entries),
    )
    save_renderplan_set(variant_dir / "renderplan-set.toml", renderplan_set)
    return workspace, variant_name


def _preflight(plan: RenderPlan, path: Path) -> PreflightReport:
    report = PreflightReport(
        render_plan_id=plan.plan_id or plan.computed_id(),
        source_plan_id=plan.source_plan_id,
        render_profile_id=plan.render_profile_id,
        status="ok",
        checks=tuple(
            PreflightCheck(
                segment_number=item.number,
                segment_id=item.segment_id or item.computed_id(),
                status="ok",
                metric="fixture",
                measured=1,
                maximum=10,
                source="test.fixture",
            )
            for item in plan.speech
            if item.preflight_required
        ),
        producer=Producer(name="voicerender", version="test"),
    ).with_identity()
    report.validate_against(plan)
    save_preflight(report, path)
    return report


def _emit_test_progress(
    callback: Any,
    kind: RenderEventKind,
    tracker: ProgressTracker,
    *,
    scope: RenderScope,
    **context: Any,
) -> None:
    snapshot = tracker.snapshot()
    emit_render_event(
        callback,
        RenderEvent(
            kind=kind,
            scope=scope,  # type: ignore[arg-type]
            phase=scope,
            completed_count=snapshot.completed_count,
            total_count=snapshot.total_count,
            progress=snapshot.progress,
            phase_completed_count=snapshot.completed_count,
            phase_total_count=snapshot.total_count,
            phase_progress=snapshot.progress,
            rendered_count=snapshot.rendered_count,
            reused_count=snapshot.reused_count,
            failed_count=snapshot.failed_count,
            selected_count=snapshot.total_count,
            elapsed_seconds=snapshot.elapsed_seconds,
            eta_seconds=snapshot.eta_seconds,
            eta_confidence=snapshot.eta_confidence,
            progress_snapshot=snapshot,
            **context,
        ),
    )


def _install_fake_execution(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    preflight_calls: list[str] = []
    render_calls: list[tuple[str, frozenset[int] | None]] = []

    policy_calls: list[str] = []

    def preflight(
        plan: RenderPlan,
        path: str | Path,
        *,
        emphasis_policy: book_module.EmphasisPolicy = book_module.EmphasisPolicy.REJECT,
        on_event: Any = None,
    ) -> PreflightReport:
        preflight_calls.append(plan.plan_id or plan.computed_id())
        policy_calls.append(f"preflight:{emphasis_policy.value}")
        report = _preflight(plan, Path(path))
        if on_event is not None:
            required = [item for item in plan.speech if item.preflight_required]
            tracker = ProgressTracker(len(required))
            plan_id = plan.plan_id or plan.computed_id()
            _emit_test_progress(
                on_event,
                RenderEventKind.PREFLIGHT_STARTED,
                tracker,
                scope="preflight",
                plan_id=plan_id,
            )
            checks = {
                (check.segment_number, check.segment_id): check
                for check in report.checks
            }
            for item in required:
                segment_id = item.segment_id or item.computed_id()
                _emit_test_progress(
                    on_event,
                    RenderEventKind.PREFLIGHT_ITEM_STARTED,
                    tracker,
                    scope="preflight",
                    plan_id=plan_id,
                    segment_number=item.number,
                    segment_id=segment_id,
                )
                tracker.advance()
                check = checks[(item.number, segment_id)]
                _emit_test_progress(
                    on_event,
                    RenderEventKind.PREFLIGHT_ITEM_FINISHED,
                    tracker,
                    scope="preflight",
                    plan_id=plan_id,
                    segment_number=item.number,
                    segment_id=segment_id,
                    status=check.status,
                )
            _emit_test_progress(
                on_event,
                RenderEventKind.PREFLIGHT_FINISHED,
                tracker,
                scope="preflight",
                plan_id=plan_id,
                status=report.status,
            )
        return report

    def render(
        plan: RenderPlan,
        *,
        preflight: PreflightReport,
        output_dir: str | Path,
        cache_dir: str | Path | None = None,
        force: bool = False,
        reset_stale_output: bool = False,
        recheck_report_path: str | Path | None = None,
        speech_numbers: set[int] | frozenset[int] | None = None,
        emphasis_policy: book_module.EmphasisPolicy = book_module.EmphasisPolicy.REJECT,
        on_event: Any = None,
    ) -> RenderPlanResult:
        del cache_dir, recheck_report_path, reset_stale_output
        policy_calls.append(f"render:{emphasis_policy.value}")
        selected = (
            tuple(plan.speech)
            if speech_numbers is None
            else tuple(item for item in plan.speech if item.number in speech_numbers)
        )
        render_calls.append(
            (
                plan.plan_id or plan.computed_id(),
                None if speech_numbers is None else frozenset(speech_numbers),
            )
        )
        tracker = ProgressTracker(len(selected))
        _emit_test_progress(
            on_event,
            RenderEventKind.RENDER_STARTED,
            tracker,
            scope="render",
            plan_id=plan.plan_id or plan.computed_id(),
        )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / RENDER_MANIFEST_FILENAME
        artifacts: dict[tuple[int, str], SpeechArtifact] = {}
        if manifest_path.is_file() and not force:
            prior = load_manifest(manifest_path)
            if prior.render_plan_id != (plan.plan_id or plan.computed_id()):
                raise ArtifactError("stale plan", code="output.stale_plan")
            prior.validate_against(plan)
            artifacts = {
                (item.segment_number, item.segment_id): item for item in prior.speech
            }
        rendered = reused = 0
        rendered_keys: set[tuple[int, str]] = set()
        reused_keys: set[tuple[int, str]] = set()
        for item in selected:
            key = (item.number, item.segment_id or item.computed_id())
            if key in artifacts and not force:
                reused += 1
                reused_keys.add(key)
                continue
            wav_path = output / item.wav
            info = atomic_wav(wav_path, [0.0, 0.1, -0.1], 24000)
            artifacts[key] = SpeechArtifact(
                segment_number=item.number,
                segment_id=key[1],
                wav=item.wav,
                sha256=f"sha256:{info['sha256']}",
                frames=int(info["frames"]),
                sample_rate=int(info["sample_rate"]),
                channels=int(info["channels"]),
            )
            rendered += 1
            rendered_keys.add(key)
        ordered = tuple(
            artifacts[(item.number, item.segment_id or item.computed_id())]
            for item in plan.speech
            if (item.number, item.segment_id or item.computed_id()) in artifacts
        )
        manifest = RenderManifest(
            render_plan_id=plan.plan_id or plan.computed_id(),
            speech=ordered,
            complete=len(ordered) == len(plan.speech),
            producer=Producer(name="voicerender", version="test"),
        ).with_identity()
        manifest.validate_against(plan)
        save_manifest(manifest, manifest_path)
        pending_composition = (
            emphasis_policy is book_module.EmphasisPolicy.DEFER_TO_COMPOSE
            and any(
                control.name == "emphasis"
                for item in plan.speech
                for control in item.controls
            )
        )
        if pending_composition:
            diagnostics = {
                "schema": "voicerender.render-diagnostics.v1",
                "render_plan_id": plan.plan_id or plan.computed_id(),
                "emphasis_policy": emphasis_policy.value,
                "pending_composition": True,
                "deferred_controls": [
                    {
                        "capability": book_module.EMPHASIS_COMPOSITION_CAPABILITY,
                        "segment_number": item.number,
                        "segment_id": item.segment_id or item.computed_id(),
                        "count": sum(
                            control.name == "emphasis" for control in item.controls
                        ),
                    }
                    for item in plan.speech
                    if any(control.name == "emphasis" for control in item.controls)
                ],
            }
            (output / RENDER_DIAGNOSTICS_FILENAME).write_text(
                json.dumps(diagnostics), encoding="utf-8"
            )
        selected_keys = {
            (item.number, item.segment_id or item.computed_id()) for item in selected
        }
        result = RenderPlanResult(
            manifest=manifest,
            manifest_path=manifest_path,
            planned_count=len(plan.speech),
            selected_count=len(selected),
            available_count=len(ordered),
            rendered_count=rendered,
            reused_count=reused,
            selection_complete=selected_keys.issubset(set(artifacts)),
            pending_composition=pending_composition,
        )
        for item in selected:
            key = (item.number, item.segment_id or item.computed_id())
            if key in rendered_keys:
                tracker.complete("rendered", synthesis_seconds=0.25)
                kind = RenderEventKind.SEGMENT_RENDERED
                synthesis_seconds = 0.25
            else:
                tracker.complete("reused")
                kind = RenderEventKind.SEGMENT_REUSED
                synthesis_seconds = None
            _emit_test_progress(
                on_event,
                kind,
                tracker,
                scope="render",
                plan_id=plan.plan_id or plan.computed_id(),
                segment_number=item.number,
                segment_id=key[1],
                engine=str(item.binding.get("renderer", "test")),
                status="rendered" if synthesis_seconds is not None else "reused",
                synthesis_seconds=synthesis_seconds,
            )
        _emit_test_progress(
            on_event,
            RenderEventKind.RENDER_COMPLETED,
            tracker,
            scope="render",
            plan_id=plan.plan_id or plan.computed_id(),
            status="completed",
        )
        return result

    monkeypatch.setattr(book_module, "preflight_render_plan", preflight)
    monkeypatch.setattr(book_module, "render_renderplan", render)
    return {
        "preflight": preflight_calls,
        "render": render_calls,
        "policy": policy_calls,
    }


def test_book_propagates_explicit_emphasis_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fake_execution(monkeypatch)
    assert render_book is book_module.render_book
    assert BookRenderError is book_module.BookRenderError
    assert BookRenderRequest is book_module.BookRenderRequest
    assert BookRenderResult is book_module.BookRenderResult
    assert ChapterRenderOutcome is book_module.ChapterRenderOutcome
    workspace, variant = _book(tmp_path)

    result = render_book(workspace, variant=variant, emphasis_policy="defer-to-compose")

    assert result.exit_code == 0
    assert len(calls["policy"]) == 6
    assert all(call.endswith(":defer-to-compose") for call in calls["policy"])


def test_book_reports_pending_composition_in_result_and_human_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path, chapters=((5, 1),), emphasis_speech=1)

    result = book_module.render_book(
        workspace, variant=variant, emphasis_policy="defer-to-compose"
    )
    assert result.pending_composition is True
    outcome = result.outcomes[0]
    assert outcome.pending_composition is True
    report = tomllib.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["pending_composition"] is True
    assert report["item"][0]["pending_composition"] is True

    diagnostics_path = outcome.output_dir / "render-diagnostics.json"
    legacy = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert legacy["schema"] == "voicerender.render-diagnostics.v1"
    assert "error_diagnostic" not in legacy
    second = book_module.render_book(
        workspace, variant=variant, emphasis_policy="defer-to-compose"
    )
    assert second.outcomes[0].status == "reused"
    assert len(calls["render"]) == 1

    additive = dict(legacy)
    additive["error_type"] = "EngineBackendError"
    additive["error_diagnostic"] = {
        "schema": "voicerender.error-diagnostic.v1",
        "type": "EngineBackendError",
        "message": "optional additive field",
    }
    diagnostics_path.write_text(json.dumps(additive), encoding="utf-8")
    third = book_module.render_book(
        workspace, variant=variant, emphasis_policy="defer-to-compose"
    )
    assert third.outcomes[0].status == "reused"
    assert len(calls["render"]) == 1

    from voicerender.bundle import BundleRenderResult
    from voicerender.cli import _print_bundle_result

    _print_bundle_result(BundleRenderResult(kind="book", value=result))
    assert "emphasis remains pending composition" in capsys.readouterr().out


def test_book_renders_in_set_order_and_idempotently_reuses_complete_chapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path)

    first = book_module.render_book(workspace, variant=variant)
    assert first.exit_code == 0
    assert first.workspace == workspace.resolve()
    assert first.variant == variant
    assert first.output_variant_dir == workspace / "voicerender" / variant
    assert (
        first.report_path == first.output_variant_dir / book_module.BOOK_REPORT_FILENAME
    )
    assert first.requested == 3
    assert first.written == sum(outcome.rendered for outcome in first.outcomes) == 6
    assert first.reused == sum(outcome.reused for outcome in first.outcomes) == 0
    assert first.failed == first.skipped == 0
    assert first.success is True
    assert [item.id for item in first.outcomes] == [
        "chapter-0007",
        "chapter-0005",
        "chapter-0008",
    ]
    assert [item.status for item in first.outcomes] == [
        "complete",
        "complete",
        "complete",
    ]
    assert first.full_chapters == 3
    assert len(calls["preflight"]) == 3
    assert len(calls["render"]) == 3
    for outcome in first.outcomes:
        assert outcome.manifest is not None and outcome.manifest.is_file()
        assert (outcome.output_dir / "000001.wav").is_file()
    report = tomllib.loads(first.report_path.read_text(encoding="utf-8"))
    assert report["format"] == "voicerender-bundle-book-report"
    assert report["variant"] == variant
    assert report["requested"] == report["succeeded"] == 3
    assert report["full_chapters"] == 3

    second = book_module.render_book(workspace, variant=variant)
    assert second.exit_code == 0
    assert [item.status for item in second.outcomes] == ["reused"] * 3
    assert second.written == 0
    assert second.reused == 6
    assert second.success is True
    assert len(calls["preflight"]) == 3
    assert len(calls["render"]) == 3


def test_book_chapter_and_speech_selectors_use_source_and_speech_numbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path)

    selected = book_module.render_book(
        workspace,
        variant=variant,
        chapters="5",
        speech="2-3",
    )
    assert [item.id for item in selected.outcomes] == ["chapter-0005"]
    assert selected.outcomes[0].status == "partial"
    assert selected.outcomes[0].planned_speech == 3
    assert selected.outcomes[0].selected_speech == 2
    assert selected.outcomes[0].available_speech == 2
    assert selected.partial_chapters == 1
    assert selected.exit_code == 0
    assert [
        item.wav for item in load_manifest(selected.outcomes[0].manifest).speech
    ] == [
        "000002.wav",
        "000003.wav",
    ]
    assert calls["render"] == [(selected.outcomes[0].render_plan_id, frozenset({2, 3}))]


def test_partial_windows_accumulate_and_later_full_selection_finishes_chapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path, chapters=((5, 3),))

    first = book_module.render_book(
        workspace, variant=variant, chapters="5", speech="1"
    )
    assert first.exit_code == 0
    assert first.outcomes[0].status == "partial"
    second = book_module.render_book(
        workspace, variant=variant, chapters="5", speech="3"
    )
    assert second.exit_code == 0
    assert second.outcomes[0].status == "partial"
    assert [
        item.segment_number
        for item in load_manifest(second.outcomes[0].manifest).speech
    ] == [1, 3]
    assert len(calls["preflight"]) == 1

    final = book_module.render_book(workspace, variant=variant, chapters="5")
    assert final.exit_code == 0
    assert final.outcomes[0].status == "complete"
    assert final.outcomes[0].manifest_complete
    assert final.outcomes[0].available_speech == 3
    assert len(calls["preflight"]) == 1


@pytest.mark.parametrize(
    ("status", "check_status", "error_code", "replan_required"),
    (
        ("replan", "too_long", "preflight.replan", True),
        ("reject", "error", "preflight.reject", False),
    ),
)
def test_book_preserves_initial_preflight_report_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: str,
    check_status: str,
    error_code: str,
    replan_required: bool,
) -> None:
    calls = _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path, chapters=((5, 1),))

    def initial_preflight(
        plan: RenderPlan,
        path: str | Path,
        *,
        emphasis_policy: book_module.EmphasisPolicy = book_module.EmphasisPolicy.REJECT,
    ) -> PreflightReport:
        del emphasis_policy
        speech = plan.speech[0]
        segment_id = speech.segment_id or speech.computed_id()
        check = PreflightCheck(
            segment_number=speech.number,
            segment_id=segment_id,
            status=check_status,
            metric="model_tokens",
            measured=101,
            maximum=100,
            source="test.initial_measurement",
        )
        requests = ()
        if status == "replan":
            requests = (
                ReplanRequest(
                    segment_number=speech.number,
                    segment_id=segment_id,
                    source=speech.source,
                    current_level=speech.planning_level,
                    action="split",
                    reason="capacity",
                    metric="model_tokens",
                    measured=101,
                    maximum=100,
                ),
            )
        report = PreflightReport(
            render_plan_id=plan.plan_id or plan.computed_id(),
            source_plan_id=plan.source_plan_id,
            render_profile_id=plan.render_profile_id,
            status=status,
            checks=(check,),
            replan=requests,
            producer=Producer(name="voicerender", version="test"),
        ).with_identity()
        save_preflight(report, path)
        return report

    monkeypatch.setattr(book_module, "preflight_render_plan", initial_preflight)
    result = book_module.render_book(workspace, variant=variant)
    outcome = result.outcomes[0]

    assert outcome.status == "failed"
    assert outcome.error_code == error_code
    assert outcome.replan_required is replan_required
    assert outcome.preflight_report == outcome.output_dir / "preflight.toml"
    assert outcome.preflight_report.is_file()
    assert calls["render"] == []

    item = result.to_dict()["items"][0]
    assert item["replan_required"] is replan_required
    assert item["preflight_report"] == "chapter-0005/preflight.toml"
    report = tomllib.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["item"][0]["replan_required"] is replan_required
    assert report["item"][0]["preflight_report"] == "chapter-0005/preflight.toml"

    from voicerender.bundle import BundleRenderResult
    from voicerender.cli import _print_bundle_result

    _print_bundle_result(BundleRenderResult(kind="book", value=result))
    label = "replan report" if replan_required else "preflight report"
    assert f"{label}: {outcome.preflight_report}" in capsys.readouterr().out


def test_book_persists_runtime_replan_report_for_exact_plan_occurrence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path, chapters=((5, 1),))
    original_render = book_module.render_renderplan

    def replan_during_render(
        plan: RenderPlan,
        *,
        preflight: PreflightReport,
        output_dir: str | Path,
        cache_dir: str | Path | None = None,
        force: bool = False,
        reset_stale_output: bool = False,
        recheck_report_path: str | Path | None = None,
        speech_numbers: set[int] | frozenset[int] | None = None,
        emphasis_policy: book_module.EmphasisPolicy = book_module.EmphasisPolicy.REJECT,
    ) -> RenderPlanResult:
        result = original_render(
            plan,
            preflight=preflight,
            output_dir=output_dir,
            cache_dir=cache_dir,
            force=force,
            reset_stale_output=reset_stale_output,
            recheck_report_path=recheck_report_path,
            speech_numbers=speech_numbers,
            emphasis_policy=emphasis_policy,
        )
        assert recheck_report_path is not None
        speech = plan.speech[0]
        check = PreflightCheck(
            segment_number=speech.number,
            segment_id=speech.segment_id or speech.computed_id(),
            status="too_long",
            metric="model_tokens",
            measured=101,
            maximum=100,
            source="test.runtime_measurement",
        )
        request = ReplanRequest(
            segment_number=speech.number,
            segment_id=speech.segment_id or speech.computed_id(),
            source=speech.source,
            current_level=speech.planning_level,
            action="split",
            reason="capacity",
            metric="model_tokens",
            measured=101,
            maximum=100,
        )
        fresh = PreflightReport(
            render_plan_id=plan.plan_id or plan.computed_id(),
            source_plan_id=plan.source_plan_id,
            render_profile_id=plan.render_profile_id,
            status="replan",
            checks=(check,),
            replan=(request,),
            producer=Producer(name="voicerender", version="test"),
        ).with_identity()
        fresh.validate_against(plan)
        save_preflight(fresh, recheck_report_path)
        return replace(
            result,
            error="fresh capacity check rejected this occurrence",
            fresh_preflight=fresh,
            error_code="preflight.replan",
        )

    monkeypatch.setattr(book_module, "render_renderplan", replan_during_render)
    rendered = book_module.render_book(workspace, variant=variant)

    outcome = rendered.outcomes[0]
    assert outcome.status == "failed"
    assert outcome.error_code == "preflight.replan"
    assert outcome.recheck_report is not None
    assert outcome.recheck_report.name == "preflight-recheck.toml"
    assert outcome.preflight_report == outcome.recheck_report
    assert outcome.preflight_report.name == "preflight-recheck.toml"
    assert outcome.replan_required is True
    fresh = book_module.load_preflight(outcome.recheck_report)
    assert fresh.status == "replan"
    assert fresh.render_plan_id == outcome.render_plan_id
    expected_plan = load_plan(outcome.renderplan)
    assert [(check.segment_number, check.segment_id) for check in fresh.checks] == [
        (
            expected_plan.speech[0].number,
            expected_plan.speech[0].segment_id or expected_plan.speech[0].computed_id(),
        )
    ]
    report = tomllib.loads(rendered.report_path.read_text(encoding="utf-8"))
    assert report["item"][0]["error_code"] == "preflight.replan"
    assert report["item"][0]["replan_required"] is True
    assert report["item"][0]["preflight_report"] == (
        "chapter-0005/preflight-recheck.toml"
    )
    assert report["item"][0]["recheck_report"] == (
        "chapter-0005/preflight-recheck.toml"
    )
    payload_item = rendered.to_dict()["items"][0]
    assert payload_item["replan_required"] is True
    assert payload_item["preflight_report"] == "chapter-0005/preflight-recheck.toml"
    assert report["item"][0]["recheck_report"] == (
        "chapter-0005/preflight-recheck.toml"
    )


def test_invalid_selection_and_identity_fail_before_preflight_or_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path)

    with pytest.raises(book_module.BookRenderError) as unknown:
        book_module.render_book(workspace, variant=variant, chapters="99")
    assert unknown.value.code == "selection.invalid"
    with pytest.raises(
        book_module.BookRenderError, match="exactly one selected chapter"
    ):
        book_module.render_book(
            workspace, variant=variant, chapters="5,7", speech="1-2"
        )
    assert calls["preflight"] == []
    assert calls["render"] == []

    set_path = workspace / "renderplan" / variant / "renderplan-set.toml"
    renderplan_set = load_renderplan_set(set_path)
    bad_chapter = replace(
        renderplan_set.chapters[0], render_plan_id="sha256:" + "f" * 64
    )
    save_renderplan_set(
        set_path,
        replace(renderplan_set, chapters=(bad_chapter, *renderplan_set.chapters[1:])),
    )
    with pytest.raises(book_module.BookRenderError) as mismatch:
        book_module.render_book(workspace, variant=variant)
    assert mismatch.value.code == "plan.identity_mismatch"
    assert calls["render"] == []


def test_malformed_set_variant_path_and_symlink_escape_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path, chapters=((5, 1),))
    variant_dir = workspace / "renderplan" / variant
    set_path = variant_dir / "renderplan-set.toml"

    with pytest.raises(book_module.BookRenderError) as wrong_name:
        book_module.render_book(workspace, variant="narration")
    assert wrong_name.value.code == "variant.not_found"

    set_path.write_text("format = [", encoding="utf-8")
    with pytest.raises(book_module.BookRenderError) as malformed:
        book_module.render_book(workspace, variant=variant)
    assert malformed.value.code == "index.read_failed"

    workspace2, variant2 = _book(tmp_path / "second", chapters=((5, 1),))
    variant_dir2 = workspace2 / "renderplan" / variant2
    set_path2 = variant_dir2 / "renderplan-set.toml"
    renderplan_set2 = load_renderplan_set(set_path2)
    entry = renderplan_set2.chapters[0]
    outside = tmp_path / "outside.renderplan.toml"
    outside.write_text(
        (variant_dir2 / entry.renderplan).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (variant_dir2 / entry.renderplan).unlink()
    (variant_dir2 / entry.renderplan).symlink_to(outside)
    with pytest.raises(book_module.BookRenderError) as escape:
        book_module.render_book(workspace2, variant=variant2)
    assert escape.value.code == "plan.missing"


def test_fail_fast_skips_remaining_chapters_but_default_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path)
    original = book_module.render_renderplan
    failed_once = False

    def fail_first(*args: Any, **kwargs: Any) -> RenderPlanResult:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise ArtifactError("intentional failure", code="artifact.invalid")
        return original(*args, **kwargs)

    monkeypatch.setattr(book_module, "render_renderplan", fail_first)
    fast_events: list[RenderEvent] = []
    fast = book_module.render_book(
        workspace,
        variant=variant,
        fail_fast=True,
        on_event=fast_events.append,
    )
    assert [item.status for item in fast.outcomes] == ["failed", "skipped", "skipped"]
    assert fast.failed == 1 and fast.skipped == 2
    assert fast.outcomes[0].error_code == "artifact.invalid"
    skipped_events = [
        event for event in fast_events if event.kind is RenderEventKind.CHAPTER_SKIPPED
    ]
    assert len(skipped_events) == 2
    assert fast_events[-1].kind is RenderEventKind.BOOK_COMPLETED
    assert fast_events[-1].progress is not None and fast_events[-1].progress < 1.0
    assert tomllib.loads(fast.report_path.read_text(encoding="utf-8"))["skipped"] == 2

    # The default mode continues with each later selected chapter.
    failed_once = False
    continued = book_module.render_book(workspace, variant=variant, force=True)
    assert [item.status for item in continued.outcomes] == [
        "failed",
        "complete",
        "complete",
    ]
    assert continued.failed == 1 and continued.succeeded == 2
    assert calls["preflight"]


def test_render_book_cli_json_qc_then_human_completion_and_json_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path, chapters=((5, 3),))
    output_root = tmp_path / "custom-render"
    cache_dir = tmp_path / "custom-cache"
    from voicerender.cli import main

    partial_code = main(
        [
            "bundle",
            "render",
            str(workspace),
            "--variant",
            variant,
            "--chapters",
            "5",
            "--speech",
            "1-2",
            "--renderplan-root",
            str(workspace / "renderplan"),
            "--output-root",
            str(output_root),
            "--cache-dir",
            str(cache_dir),
            "--json",
        ]
    )
    partial_payload = json.loads(capsys.readouterr().out)
    partial_result = partial_payload["result"]
    assert partial_payload["format"] == "voicerender-bundle-result"
    assert partial_payload["kind"] == "book"
    assert partial_code == 0
    assert partial_result["partial_chapters"] == 1
    assert partial_result["items"][0]["status"] == "partial"
    assert partial_result["items"][0]["manifest_complete"] is False
    assert (output_root / variant / "chapter-0005" / "000001.wav").is_file()
    assert cache_dir.is_dir()

    full_code = main(
        [
            "bundle",
            "render",
            str(workspace),
            "--variant",
            variant,
            "--renderplan-root",
            str(workspace / "renderplan"),
            "--output-root",
            str(output_root),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    human_output = capsys.readouterr().out
    assert full_code == 0
    assert "chapter state: complete" in human_output
    assert "VoiceRender summary" in human_output
    assert "variant:   " + variant in human_output
    assert len(calls["preflight"]) == 1
    assert len(calls["render"]) == 2

    error_code = main(
        [
            "bundle",
            "render",
            str(workspace),
            "--variant",
            variant,
            "--chapters",
            "99",
            "--json",
        ]
    )
    error_payload = json.loads(capsys.readouterr().out)
    assert error_code == 2
    assert error_payload["code"] == "selection.invalid"
    assert error_payload["exit_code"] == 2
    assert len(calls["preflight"]) == 1


def test_public_api_and_render_book_cli_have_behavioral_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_execution(monkeypatch)
    api_workspace, api_variant = _book(
        tmp_path / "api", chapters=((5, 3), (7, 1)), emphasis_speech=2
    )
    cli_workspace, cli_variant = _book(
        tmp_path / "cli", chapters=((5, 3), (7, 1)), emphasis_speech=2
    )
    assert api_variant == cli_variant

    import voicerender.cli as cli_module
    from voicerender import render_bundle
    from voicerender.cli import main

    assert cli_module.render_bundle is render_bundle

    def run_cli(*, force: bool) -> tuple[int, dict[str, Any]]:
        args = [
            "bundle",
            "render",
            str(cli_workspace),
            "--variant",
            cli_variant,
            "--chapters",
            "5",
            "--speech",
            "2-3",
            "--emphasis-policy",
            "defer-to-compose",
            "--json",
        ]
        if force:
            args.append("--force")
        code = main(args)
        return code, json.loads(capsys.readouterr().out)

    def assert_parity(
        api_result: book_module.BookRenderResult, payload: dict[str, Any]
    ) -> None:
        assert payload["format"] == "voicerender-bundle-result"
        assert payload["kind"] == "book"
        payload = payload["result"]
        assert payload["variant"] == api_result.variant == api_variant
        assert payload["pending_composition"] is api_result.pending_composition is True
        assert payload["emphasis_policy"] == "defer-to-compose"
        assert payload["selection"] == {"chapters": "5", "speech": "2-3"}
        assert len(api_result.outcomes) == len(payload["items"]) == 1
        outcome = api_result.outcomes[0]
        item = payload["items"][0]
        assert outcome.id == item["id"] == "chapter-0005"
        assert outcome.render_plan_id == item["render_plan_id"]
        assert outcome.status == item["status"]
        assert outcome.selected_speech == item["selected_speech"] == 2
        assert outcome.pending_composition is item["pending_composition"] is True
        assert api_result.written == item["rendered"]
        assert api_result.reused == item["reused"]

        api_report = tomllib.loads(api_result.report_path.read_text(encoding="utf-8"))
        cli_report_path = (
            cli_workspace
            / "voicerender"
            / cli_variant
            / book_module.BOOK_REPORT_FILENAME
        )
        cli_report = tomllib.loads(cli_report_path.read_text(encoding="utf-8"))
        assert api_report == cli_report

        api_manifest = load_manifest(outcome.manifest)
        cli_manifest = load_manifest(
            cli_workspace / "voicerender" / cli_variant / item["manifest"]
        )
        assert api_manifest.manifest_id == cli_manifest.manifest_id
        assert api_manifest.render_plan_id == cli_manifest.render_plan_id
        assert [artifact.segment_number for artifact in api_manifest.speech] == [2, 3]
        assert [artifact.segment_number for artifact in cli_manifest.speech] == [2, 3]

    for force, expected_status, expected_written, expected_reused in (
        (False, "partial", 2, 0),
        (False, "reused", 0, 2),
        (True, "partial", 2, 0),
    ):
        api_result = render_bundle(
            api_workspace,
            variant=api_variant,
            chapters="5",
            speech="2-3",
            force=force,
            emphasis_policy="defer-to-compose",
        ).value
        cli_code, cli_payload = run_cli(force=force)
        assert cli_code == api_result.exit_code == 0
        assert api_result.outcomes[0].status == expected_status
        assert api_result.written == expected_written
        assert api_result.reused == expected_reused
        assert_parity(api_result, cli_payload)


def test_book_emits_global_speech_progress_and_preflight_phase_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path)
    events: list[RenderEvent] = []

    result = render_book(workspace, variant=variant, on_event=events.append)

    assert result.elapsed_seconds >= 0
    assert all(outcome.elapsed_seconds >= 0 for outcome in result.outcomes)
    assert events[0].kind is RenderEventKind.BOOK_STARTED
    assert events[0].total_count == 6
    assert events[-1].kind is RenderEventKind.BOOK_COMPLETED
    assert all(event.total_count == 6 for event in events)

    preflight_items = [
        event
        for event in events
        if event.kind is RenderEventKind.PREFLIGHT_ITEM_FINISHED
    ]
    assert preflight_items
    assert preflight_items[0].scope == "book"
    assert preflight_items[0].chapter_id is not None
    assert preflight_items[0].phase_total_count == 2
    assert preflight_items[0].phase_progress == 0.5
    assert preflight_items[0].progress == 0.0

    rendered = [
        event for event in events if event.kind is RenderEventKind.SEGMENT_RENDERED
    ]
    assert len(rendered) == 6
    assert [event.completed_count for event in rendered] == [1, 2, 3, 4, 5, 6]
    assert [event.progress for event in rendered] == [
        1 / 6,
        2 / 6,
        3 / 6,
        4 / 6,
        5 / 6,
        1.0,
    ]
    assert rendered[0].eta_seconds is not None
    assert rendered[0].eta_confidence == "low"

    first_chapter_finish = next(
        index
        for index, event in enumerate(events)
        if event.kind is RenderEventKind.CHAPTER_FINISHED
    )
    assert any(
        event.kind is RenderEventKind.SEGMENT_RENDERED
        for event in events[:first_chapter_finish]
    )


def test_reused_chapters_advance_book_progress_immediately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path)
    render_book(workspace, variant=variant)
    events: list[RenderEvent] = []

    result = render_book(workspace, variant=variant, on_event=events.append)

    assert all(outcome.status == "reused" for outcome in result.outcomes)
    reused_chapters = [
        event for event in events if event.kind is RenderEventKind.CHAPTER_REUSED
    ]
    assert len(reused_chapters) == 3
    assert [event.progress for event in reused_chapters] == [2 / 6, 5 / 6, 1.0]
    assert [event.eta_seconds for event in reused_chapters] == [None, None, 0.0]
    assert events[-1].kind is RenderEventKind.BOOK_COMPLETED
    assert events[-1].progress == 1.0


def test_estimate_render_book_is_workload_only_and_does_not_create_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voicerender import estimate_render_book

    _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path)
    output_root = tmp_path / "estimated-output"

    initial = estimate_render_book(workspace, variant=variant, output_root=output_root)
    assert initial.selected_count == 6
    assert initial.known_output_reuse_count == 0
    assert initial.pending_count == 6
    assert initial.estimated_seconds is None
    assert initial.confidence == "unavailable"
    assert initial.source == "workload-only"
    assert not output_root.exists()

    render_book(workspace, variant=variant, output_root=output_root)
    before_files = {
        path.relative_to(output_root): path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    complete = estimate_render_book(workspace, variant=variant, output_root=output_root)
    partial = estimate_render_book(
        workspace,
        variant=variant,
        chapters="5",
        speech="2-3",
        output_root=output_root,
    )

    assert (complete.selected_count, complete.known_output_reuse_count) == (6, 6)
    assert (complete.pending_count, complete.estimated_seconds) == (0, None)
    assert (partial.selected_count, partial.known_output_reuse_count) == (2, 2)
    after_files = {
        path.relative_to(output_root): path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    assert after_files == before_files


def test_book_workspace_and_public_apis_accept_root_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voicerender import estimate_render_book
    from voicerender.book_layout import resolve_workspace

    workspace, _ = _book(tmp_path)
    child = workspace / "child"
    child.mkdir()
    for working_directory, path in (
        (workspace, "."),
        (workspace, "./"),
        (child, ".."),
    ):
        monkeypatch.chdir(working_directory)
        assert resolve_workspace(path) == workspace.resolve()
        for render_api in (render_book, estimate_render_book):
            with pytest.raises(
                BookRenderError, match="variant directory does not exist"
            ):
                render_api(path, variant="missing")

    monkeypatch.chdir(workspace / "renderplan")
    with pytest.raises(BookRenderError, match="canonical .ssmd-bundle"):
        resolve_workspace(".")

    link = tmp_path / "link.ssmd-bundle"
    try:
        link.symlink_to(workspace, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    with pytest.raises(BookRenderError, match="root must not be a symlink"):
        resolve_workspace(link)


def test_book_report_and_events_propagate_error_diagnostic_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path, chapters=((5, 1),))
    original_render = book_module.render_renderplan
    diagnostic = {
        "schema": "voicerender.error-diagnostic.v1",
        "type": "EngineBackendError",
        "module": "voicerender.errors",
        "message": "backend failed",
        "details": {"native_stage": "inference"},
        "cause": {
            "type": "RuntimeError",
            "module": "builtins",
            "message": "provider execution failed",
        },
    }

    def render_with_diagnostic(*args: Any, **kwargs: Any) -> Any:
        result = original_render(*args, **kwargs)
        diagnostics_path = Path(kwargs["output_dir"]) / "render-diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostic), encoding="utf-8")
        return replace(
            result,
            error="backend failed",
            error_code="synthesis.backend_error",
            error_type="EngineBackendError",
            error_diagnostic=diagnostic,
            diagnostics_path=diagnostics_path,
            error_segment_number=1,
            error_segment_id="sha256:segment",
            error_engine="kokoro",
        )

    monkeypatch.setattr(book_module, "render_renderplan", render_with_diagnostic)
    events: list[RenderEvent] = []
    result = book_module.render_book(workspace, variant=variant, on_event=events.append)
    outcome = result.outcomes[0]

    assert outcome.status == "failed"
    assert (
        outcome.render_diagnostics is not None and outcome.render_diagnostics.is_file()
    )
    assert outcome.error_type == "EngineBackendError"
    assert outcome.error_stage == "inference"
    assert outcome.error_root_cause_type == "RuntimeError"
    assert outcome.error_root_cause_message == "provider execution failed"
    item = result.to_dict()["items"][0]
    assert item["render_diagnostics"] == "chapter-0005/render-diagnostics.json"
    assert item["error_root_cause_message"] == "provider execution failed"
    report = tomllib.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["item"][0]["render_diagnostics"] == item["render_diagnostics"]
    chapter_event = next(
        event for event in events if event.kind is RenderEventKind.CHAPTER_FINISHED
    )
    assert chapter_event.error_type == "EngineBackendError"
    assert chapter_event.error_stage == "inference"
    assert chapter_event.error_root_cause_message == "provider execution failed"
    assert chapter_event.render_diagnostics == str(outcome.render_diagnostics)

    from voicerender.bundle import BundleRenderResult
    from voicerender.cli import _print_bundle_result

    _print_bundle_result(BundleRenderResult(kind="book", value=result))
    human_output = capsys.readouterr().out
    assert "root cause RuntimeError: provider execution failed" in human_output
    assert str(outcome.render_diagnostics) in human_output
