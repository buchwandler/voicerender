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
from voicerender.book_contract import load_renderplan_set
from voicerender.errors import ArtifactError
from voicerender.renderplan_render import RENDER_MANIFEST_FILENAME, RenderPlanResult
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
    workspace = tmp_path / "sample.ssmdbook"
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


def _install_fake_execution(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    preflight_calls: list[str] = []
    render_calls: list[tuple[str, frozenset[int] | None]] = []

    policy_calls: list[str] = []

    def preflight(
        plan: RenderPlan,
        path: str | Path,
        *,
        emphasis_policy: book_module.EmphasisPolicy = book_module.EmphasisPolicy.REJECT,
    ) -> PreflightReport:
        preflight_calls.append(plan.plan_id or plan.computed_id())
        policy_calls.append(f"preflight:{emphasis_policy.value}")
        return _preflight(plan, Path(path))

    def render(
        plan: RenderPlan,
        *,
        preflight: PreflightReport,
        output_dir: str | Path,
        cache_dir: str | Path | None = None,
        force: bool = False,
        recheck_report_path: str | Path | None = None,
        speech_numbers: set[int] | frozenset[int] | None = None,
        emphasis_policy: book_module.EmphasisPolicy = book_module.EmphasisPolicy.REJECT,
    ) -> RenderPlanResult:
        del cache_dir, recheck_report_path
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
        for item in selected:
            key = (item.number, item.segment_id or item.computed_id())
            if key in artifacts and not force:
                reused += 1
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
        selected_keys = {
            (item.number, item.segment_id or item.computed_id()) for item in selected
        }
        return RenderPlanResult(
            manifest=manifest,
            manifest_path=manifest_path,
            planned_count=len(plan.speech),
            selected_count=len(selected),
            available_count=len(ordered),
            rendered_count=rendered,
            reused_count=reused,
            selection_complete=selected_keys.issubset(set(artifacts)),
            pending_composition=(
                emphasis_policy is book_module.EmphasisPolicy.DEFER_TO_COMPOSE
                and any(
                    control.name == "emphasis"
                    for item in plan.speech
                    for control in item.controls
                )
            ),
        )

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
    workspace, variant = _book(tmp_path)

    result = book_module.render_book(
        workspace, variant=variant, emphasis_policy="defer-to-compose"
    )

    assert result.exit_code == 0
    assert len(calls["policy"]) == 6
    assert all(call.endswith(":defer-to-compose") for call in calls["policy"])


def test_book_reports_pending_composition_in_result_and_human_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_execution(monkeypatch)
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

    from voicerender.cli import _print_book_result

    _print_book_result(result)
    assert "emphasis remains pending composition" in capsys.readouterr().out


def test_book_renders_in_set_order_and_idempotently_reuses_complete_chapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fake_execution(monkeypatch)
    workspace, variant = _book(tmp_path)

    first = book_module.render_book(workspace, variant=variant)
    assert first.exit_code == 0
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
    assert report["format"] == "voicerender-book-report"
    assert report["variant"] == variant
    assert report["requested"] == report["succeeded"] == 3
    assert report["full_chapters"] == 3

    second = book_module.render_book(workspace, variant=variant)
    assert second.exit_code == 0
    assert [item.status for item in second.outcomes] == ["reused"] * 3
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
    fast = book_module.render_book(workspace, variant=variant, fail_fast=True)
    assert [item.status for item in fast.outcomes] == ["failed", "skipped", "skipped"]
    assert fast.failed == 1 and fast.skipped == 2
    assert fast.outcomes[0].error_code == "artifact.invalid"
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
            "render-book",
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
    assert partial_code == 0
    assert partial_payload["partial_chapters"] == 1
    assert partial_payload["items"][0]["status"] == "partial"
    assert partial_payload["items"][0]["manifest_complete"] is False
    assert (output_root / variant / "chapter-0005" / "000001.wav").is_file()
    assert cache_dir.is_dir()

    full_code = main(
        [
            "render-book",
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
    assert "variant:     " + variant in human_output
    assert len(calls["preflight"]) == 1
    assert len(calls["render"]) == 2

    error_code = main(
        [
            "render-book",
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
