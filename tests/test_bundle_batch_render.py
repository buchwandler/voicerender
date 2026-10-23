"""Deterministic state-aware bundle batch rendering tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

import voicerender.bundle as bundle_module
import voicerender.bundle_batch as batch_module
from voicerender.bundle import BundleRenderResult
from voicerender.bundle_contract import BundleContractError, BundleKind
from voicerender.events import RenderEvent, RenderEventKind
from voicerender.variants import (
    BundleVariant,
    BundleVariantsResult,
    PlanStatus,
    RenderStatus,
    VariantState,
)


def _variant(
    name: str,
    state: VariantState,
    *,
    plan_status: PlanStatus = "valid",
    kind: BundleKind = "book",
    pending_composition: bool | None = None,
    validation_error: str | None = None,
) -> BundleVariant:
    render_status: RenderStatus = state if state != "invalid" else "unknown"
    return BundleVariant(
        name=name,
        profile_id="sha256:profile",
        plan_status=plan_status,
        validation_error=validation_error,
        render_status=render_status,
        state=state,
        unit_kind="chapters" if kind == "book" else "documents",
        completed_units=0,
        total_units=1,
        planned_speech=2,
        available_speech=0,
        pending_composition=pending_composition,
        report_path=None,
    )


def _listing(
    root: Path,
    variants: tuple[BundleVariant, ...],
    *,
    kind: BundleKind = "book",
) -> BundleVariantsResult:
    return BundleVariantsResult(
        bundle=root,
        kind=kind,
        renderplan_root=root / "renderplan",
        output_root=root / "voicerender",
        variants=variants,
    )


def _result(
    exit_code: int = 0,
    *,
    kind: BundleKind = "book",
    error_code: str | None = None,
    error_message: str | None = None,
    error_type: str | None = None,
    error_stage: str | None = None,
    error_root_cause_type: str | None = None,
    error_root_cause_message: str | None = None,
    render_diagnostics: Path | None = None,
) -> BundleRenderResult:
    value = SimpleNamespace(
        exit_code=exit_code,
        error_code=error_code,
        error_message=error_message,
        error_type=error_type,
        error_stage=error_stage,
        error_root_cause_type=error_root_cause_type,
        error_root_cause_message=error_root_cause_message,
        render_diagnostics=render_diagnostics,
        to_dict=lambda: {"exit_code": exit_code},
    )
    return BundleRenderResult(kind=kind, value=cast(Any, value))


def _install_listings(
    monkeypatch: pytest.MonkeyPatch,
    *listings: BundleVariantsResult,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    remaining = iter(listings)

    def fake_listing(
        bundle: str | Path,
        *,
        renderplan_root: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> BundleVariantsResult:
        calls.append(
            {
                "bundle": bundle,
                "renderplan_root": renderplan_root,
                "output_root": output_root,
            }
        )
        return next(remaining)

    monkeypatch.setattr(batch_module, "list_bundle_variants", fake_listing)
    return calls


def test_default_selection_retries_noncomplete_and_keeps_pending_composition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "story.ssmd-bundle"
    initial = _listing(
        root,
        (
            _variant("a", "not_rendered"),
            _variant("b", "partial"),
            _variant("c", "failed"),
            _variant("d", "unknown"),
            _variant("e", "complete", pending_composition=True),
        ),
    )
    final = _listing(
        root,
        (
            _variant("a", "complete"),
            _variant("b", "partial"),
            _variant("c", "complete"),
            _variant("d", "unknown"),
            _variant("e", "complete", pending_composition=True),
        ),
    )
    _install_listings(monkeypatch, initial, final)
    attempted: list[str] = []

    def fake_render(
        bundle: Path, *, variant: str, **_kwargs: Any
    ) -> BundleRenderResult:
        assert bundle == root
        attempted.append(variant)
        return _result()

    monkeypatch.setattr(batch_module, "render_bundle", fake_render)

    result = batch_module.render_bundle_batch(root)

    assert attempted == ["a", "b", "c", "d"]
    assert [outcome.name for outcome in result.variants] == ["a", "b", "c", "d", "e"]
    assert result.summary.to_dict() == {
        "discovered": 5,
        "scheduled": 4,
        "attempted": 4,
        "completed": 2,
        "partial": 2,
        "failed": 0,
        "skipped_complete": 1,
        "invalid": 0,
        "not_attempted": 0,
    }
    by_name = {outcome.name: outcome for outcome in result.variants}
    assert by_name["b"].attempt_status == "complete"
    assert by_name["b"].final_state == "partial"
    assert by_name["b"].status == "partial"
    assert by_name["e"].status == "skipped_complete"
    assert by_name["e"].pending_composition is True
    assert result.exit_code == 0


def test_force_schedules_complete_valid_variants_and_forwards_all_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "story.ssmd-bundle"
    initial = _listing(
        root,
        (
            _variant("done", "complete"),
            _variant(
                "bad",
                "invalid",
                plan_status="invalid",
                validation_error="bad plan",
            ),
        ),
    )
    final = _listing(
        root,
        (
            _variant("done", "complete"),
            _variant(
                "bad",
                "invalid",
                plan_status="invalid",
                validation_error="bad plan",
            ),
        ),
    )
    discovery_calls = _install_listings(monkeypatch, initial, final)
    calls: list[tuple[Path, str, dict[str, Any]]] = []
    progress = lambda _event: None
    options = {
        "chapters": "5-8",
        "speech": "1-20",
        "renderplan_root": Path("custom-plans"),
        "output_root": Path("custom-output"),
        "cache_dir": Path("custom-cache"),
        "force": True,
        "reset_stale": True,
        "emphasis_policy": "defer-to-compose",
        "fail_fast": True,
        "on_progress": progress,
    }

    def fake_render(bundle: Path, *, variant: str, **kwargs: Any) -> BundleRenderResult:
        calls.append((bundle, variant, kwargs))
        return _result()

    monkeypatch.setattr(batch_module, "render_bundle", fake_render)

    result = batch_module.render_bundle_batch(root, **options)

    assert [name for _, name, _ in calls] == ["done"]
    assert calls[0] == (
        root,
        "done",
        {
            "chapters": "5-8",
            "speech": "1-20",
            "renderplan_root": Path("custom-plans"),
            "output_root": Path("custom-output"),
            "cache_dir": Path("custom-cache"),
            "force": True,
            "reset_stale": True,
            "emphasis_policy": "defer-to-compose",
            "fail_fast": True,
            "on_progress": progress,
        },
    )
    assert discovery_calls == [
        {
            "bundle": root,
            "renderplan_root": Path("custom-plans"),
            "output_root": Path("custom-output"),
        },
        {
            "bundle": root,
            "renderplan_root": Path("custom-plans"),
            "output_root": Path("custom-output"),
        },
    ]
    assert result.summary.scheduled == result.summary.attempted == 1
    assert result.summary.invalid == 1
    assert result.exit_code == 1


def test_no_variants_and_all_complete_are_successful_noops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "empty.ssmd-bundle"
    empty = _listing(root, ())
    empty_final = _listing(root, ())
    complete = _listing(root, (_variant("done", "complete"),))
    complete_final = _listing(root, (_variant("done", "complete"),))
    _install_listings(monkeypatch, empty, empty_final, complete, complete_final)
    monkeypatch.setattr(
        batch_module,
        "render_bundle",
        lambda *_args, **_kwargs: pytest.fail("no-op batch invoked renderer"),
    )

    empty_result = batch_module.render_bundle_batch(root)
    complete_result = batch_module.render_bundle_batch(root)

    assert empty_result.exit_code == 0
    assert empty_result.summary.discovered == 0
    assert empty_result.no_work is True
    assert complete_result.exit_code == 0
    assert complete_result.summary.skipped_complete == 1
    assert complete_result.no_work is True


def test_invalid_variant_is_never_rendered_and_causes_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "story.ssmd-bundle"
    invalid = _variant(
        "invalid", "invalid", plan_status="invalid", validation_error="invalid plan"
    )
    initial = _listing(root, (invalid, _variant("valid", "not_rendered")))
    final = _listing(root, (invalid, _variant("valid", "complete")))
    _install_listings(monkeypatch, initial, final)
    attempted: list[str] = []
    monkeypatch.setattr(
        batch_module,
        "render_bundle",
        lambda _bundle, *, variant, **_kwargs: attempted.append(variant) or _result(),
    )

    result = batch_module.render_bundle_batch(root)

    assert attempted == ["valid"]
    invalid_outcome = result.variants[0]
    assert invalid_outcome.status == "invalid"
    assert invalid_outcome.error_message == "invalid plan"
    assert result.summary.invalid == 1
    assert result.exit_code == 1


@pytest.mark.parametrize(
    ("fail_fast", "expected_attempts", "expected_not_attempted"),
    [(False, ["a", "b", "c"], 0), (True, ["a"], 2)],
)
def test_failures_continue_or_mark_remaining_variants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_fast: bool,
    expected_attempts: list[str],
    expected_not_attempted: int,
) -> None:
    root = tmp_path / "story.ssmd-bundle"
    variants = tuple(_variant(name, "not_rendered") for name in ("a", "b", "c"))
    _install_listings(
        monkeypatch,
        _listing(root, variants),
        _listing(root, variants),
    )
    attempted: list[str] = []

    def fake_render(
        _bundle: Path, *, variant: str, **_kwargs: Any
    ) -> BundleRenderResult:
        attempted.append(variant)
        if variant == "a":
            return _result(
                1, error_code="engine.unavailable", error_message="engine missing"
            )
        return _result()

    monkeypatch.setattr(batch_module, "render_bundle", fake_render)

    result = batch_module.render_bundle_batch(root, fail_fast=fail_fast)

    assert attempted == expected_attempts
    assert result.summary.failed == 1
    assert result.summary.not_attempted == expected_not_attempted
    assert result.variants[0].error_code == "engine.unavailable"
    assert result.variants[0].error_message == "engine missing"
    assert result.exit_code == 1


@pytest.mark.parametrize("kind", ["book", "document"])
def test_batch_keeps_existing_book_and_document_renderer_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: Literal["book", "document"],
) -> None:
    root = tmp_path / f"{kind}.ssmd-bundle"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ssmdconvert.book"
                if kind == "book"
                else "ssmdconvert.document",
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    initial = _listing(
        root, (_variant("variant", "not_rendered", kind=kind),), kind=kind
    )
    final = _listing(root, (_variant("variant", "complete", kind=kind),), kind=kind)
    _install_listings(monkeypatch, initial, final)
    calls: list[str] = []

    def fake_book(_bundle: Path, **_kwargs: Any) -> SimpleNamespace:
        calls.append("book")
        return SimpleNamespace(exit_code=0)

    def fake_document(_bundle: Path, **_kwargs: Any) -> SimpleNamespace:
        calls.append("document")
        return SimpleNamespace(exit_code=0)

    monkeypatch.setattr(bundle_module, "render_book", fake_book)
    monkeypatch.setattr(bundle_module, "render_document", fake_document)

    result = batch_module.render_bundle_batch(root)

    assert result.kind == kind
    assert calls == [kind]
    assert result.variants[0].final_state == "complete"
    assert result.exit_code == 0


def test_document_batch_rejects_chapter_subset_before_rendering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "document.ssmd-bundle"
    listing = _listing(
        root,
        (_variant("variant", "not_rendered", kind="document"),),
        kind="document",
    )
    _install_listings(monkeypatch, listing)
    monkeypatch.setattr(
        batch_module,
        "render_bundle",
        lambda *_args, **_kwargs: pytest.fail("invalid chapter selection rendered"),
    )

    with pytest.raises(ValueError, match="--chapters must be 'all'"):
        batch_module.render_bundle_batch(root, chapters="5")


def test_final_inspection_failure_preserves_attempt_and_returns_root_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "story.ssmd-bundle"
    initial = _listing(root, (_variant("a", "not_rendered"),))
    calls = 0

    def fake_listing(*_args: Any, **_kwargs: Any) -> BundleVariantsResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return initial
        raise BundleContractError("bundle changed during render")

    monkeypatch.setattr(batch_module, "list_bundle_variants", fake_listing)
    monkeypatch.setattr(
        batch_module, "render_bundle", lambda *_args, **_kwargs: _result()
    )

    result = batch_module.render_bundle_batch(root)

    assert result.variants[0].attempt_status == "complete"
    assert result.variants[0].final_state is None
    assert result.error_code == "bundle.invalid"
    assert result.error_message == "bundle changed during render"
    assert result.exit_code == 2


def test_batch_result_uses_versioned_json_format() -> None:
    result = batch_module.BundleBatchRenderResult(
        bundle=Path("story.ssmd-bundle"),
        kind="book",
        renderplan_root=Path("story.ssmd-bundle/renderplan"),
        output_root=Path("story.ssmd-bundle/voicerender"),
        force=False,
        variants=(),
        summary=batch_module.BundleBatchRenderSummary(
            discovered=0,
            scheduled=0,
            attempted=0,
            completed=0,
            partial=0,
            failed=0,
            skipped_complete=0,
            invalid=0,
            not_attempted=0,
        ),
        exit_code=0,
    )

    payload = result.to_dict()

    assert payload["format"] == "voicerender-bundle-batch-result"
    assert payload["version"] == 1
    assert payload["summary"]["discovered"] == 0


def test_cli_batch_json_routing_and_option_forwarding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from voicerender import cli

    payload = {
        "format": "voicerender-bundle-batch-result",
        "version": 1,
        "kind": "book",
        "summary": {"discovered": 1, "attempted": 1},
        "variants": [],
        "exit_code": 0,
    }
    result = SimpleNamespace(exit_code=0, to_dict=lambda: payload)
    calls: dict[str, Any] = {}

    def fake_batch(bundle: Path, **kwargs: Any) -> SimpleNamespace:
        calls["bundle"] = bundle
        calls["kwargs"] = kwargs
        return result

    monkeypatch.setattr(cli, "render_bundle_batch", fake_batch)

    exit_code = cli.main(
        [
            "bundle",
            "render",
            "story.ssmd-bundle",
            "--chapters",
            "5-8",
            "--speech",
            "1-20",
            "--renderplan-root",
            "plans",
            "--output-root",
            "output",
            "--cache-dir",
            "cache",
            "--force",
            "--reset-stale",
            "--emphasis-policy",
            "defer-to-compose",
            "--fail-fast",
            "--json",
            "--no-progress",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == payload
    assert captured.err == ""
    assert calls == {
        "bundle": Path("story.ssmd-bundle"),
        "kwargs": {
            "chapters": "5-8",
            "speech": "1-20",
            "renderplan_root": Path("plans"),
            "output_root": Path("output"),
            "cache_dir": Path("cache"),
            "force": True,
            "reset_stale": True,
            "emphasis_policy": "defer-to-compose",
            "fail_fast": True,
            "on_progress": None,
        },
    }


def test_batch_root_error_uses_batch_json_format(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from voicerender import cli

    def fail_batch(*_args: Any, **_kwargs: Any) -> Any:
        raise BundleContractError("not a canonical bundle")

    monkeypatch.setattr(cli, "render_bundle_batch", fail_batch)

    exit_code = cli.main(
        ["bundle", "render", "not-a-bundle", "--json", "--no-progress"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["format"] == "voicerender-bundle-batch-result"
    assert payload["version"] == 1
    assert payload["error_code"] == "bundle.invalid"
    assert payload["exit_code"] == 2
    assert payload["variants"] == []
    assert captured.err == ""


def test_batch_progress_identifies_variants_and_keeps_json_stdout_clean(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from voicerender import cli

    root = tmp_path / "story.ssmd-bundle"
    initial_variants = (_variant("a", "not_rendered"), _variant("b", "not_rendered"))
    final_variants = (_variant("a", "complete"), _variant("b", "complete"))
    _install_listings(
        monkeypatch,
        _listing(root, initial_variants),
        _listing(root, final_variants),
    )
    monkeypatch.setattr(
        batch_module,
        "render_bundle",
        lambda _bundle, *, on_progress, **_kwargs: (
            on_progress(
                RenderEvent(
                    kind=RenderEventKind.RENDER_STARTED,
                    progress=0.0,
                    completed_count=0,
                    total_count=2,
                )
            )
            or _result()
        ),
    )

    exit_code = cli.main(["bundle", "render", str(root), "--json", "--progress"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["format"] == "voicerender-bundle-batch-result"
    assert captured.out.count("\n{") == 0
    assert "[a] starting" in captured.err
    assert "[a] progress" in captured.err
    assert "[b] starting" in captured.err
    assert "[b] progress" in captured.err


def test_batch_human_output_reports_noop_and_pending_composition(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from voicerender import cli

    root = tmp_path / "story.ssmd-bundle"
    complete = (_variant("done", "complete", pending_composition=True),)
    _install_listings(
        monkeypatch,
        _listing(root, complete),
        _listing(root, complete),
    )
    monkeypatch.setattr(
        batch_module,
        "render_bundle",
        lambda *_args, **_kwargs: pytest.fail("complete variant was rendered"),
    )

    exit_code = cli.main(["bundle", "render", str(root), "--no-progress"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "mode:        batch" in output
    assert "skipped:     1 complete" in output
    assert "All valid variants are already complete; nothing to do." in output
    assert "WARNING: audio composition is pending" in output


def test_real_discovery_empty_root_is_read_only_and_successful(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "empty.ssmd-bundle"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        batch_module,
        "render_bundle",
        lambda *_args, **_kwargs: pytest.fail("empty bundle invoked renderer"),
    )

    result = batch_module.render_bundle_batch(root)

    assert result.exit_code == 0
    assert result.summary.discovered == 0
    assert not (root / "renderplan").exists()
    assert not (root / "voicerender").exists()


def test_invalid_bundle_root_fails_before_any_variant_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "not-a-bundle"
    root.mkdir()
    monkeypatch.setattr(
        batch_module,
        "render_bundle",
        lambda *_args, **_kwargs: pytest.fail("invalid bundle invoked renderer"),
    )

    with pytest.raises(BundleContractError, match="canonical .ssmd-bundle"):
        batch_module.render_bundle_batch(root)


def test_batch_failure_details_are_visible_in_human_and_json_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from voicerender import cli

    root = tmp_path / "story.ssmd-bundle"
    initial = _listing(root, (_variant("voice", "not_rendered"),))
    final = _listing(root, (_variant("voice", "failed"),))
    _install_listings(monkeypatch, initial, final, initial, final)
    monkeypatch.setattr(
        batch_module,
        "render_bundle",
        lambda *_args, **_kwargs: _result(
            1,
            error_code="engine.unavailable",
            error_message="engine unavailable",
            error_type="EngineBackendError",
            error_stage="inference",
            error_root_cause_type="RuntimeError",
            error_root_cause_message="provider execution failed",
            render_diagnostics=root / "voice" / "render-diagnostics.json",
        ),
    )

    json_exit = cli.main(["bundle", "render", str(root), "--json", "--no-progress"])
    json_output = capsys.readouterr().out
    payload = json.loads(json_output)
    human_exit = cli.main(["bundle", "render", str(root), "--no-progress"])
    human_output = capsys.readouterr().out

    assert json_exit == human_exit == 1
    assert payload["variants"][0]["error_code"] == "engine.unavailable"
    assert payload["variants"][0]["error_message"] == "engine unavailable"
    assert "engine.unavailable: engine unavailable" in human_output
    assert "root cause RuntimeError: provider execution failed" in human_output
    assert "sidecar " in human_output and "render-diagnostics.json" in human_output
    assert payload["variants"][0]["error_type"] == "EngineBackendError"
    assert payload["variants"][0]["error_stage"] == "inference"
    assert payload["variants"][0]["error_root_cause_type"] == "RuntimeError"
    assert (
        payload["variants"][0]["error_root_cause_message"]
        == "provider execution failed"
    )
    assert payload["variants"][0]["render_diagnostics"].endswith(
        "voice/render-diagnostics.json"
    )
