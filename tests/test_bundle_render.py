"""Tests for the unified bundle rendering facade."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import voicerender.bundle as bundle_module


def _bundle(root: Path, bundle_format: str) -> Path:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"format": bundle_format, "schema_version": 1}), encoding="utf-8"
    )
    return root


def _result(exit_code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        exit_code=exit_code,
        to_dict=lambda: {"status": "complete", "exit_code": exit_code},
    )


def test_render_bundle_dispatches_book_options_and_wraps_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle", "ssmdconvert.book")
    calls: dict[str, Any] = {}
    rendered = _result(exit_code=1)
    progress = lambda _event: None

    def fake_render_book(path: Path, **kwargs: Any) -> SimpleNamespace:
        calls["path"] = path
        calls["kwargs"] = kwargs
        return rendered

    monkeypatch.setattr(bundle_module, "render_book", fake_render_book)
    monkeypatch.setattr(
        bundle_module,
        "render_document",
        lambda *_args, **_kwargs: pytest.fail("book bundle dispatched as document"),
    )

    result = bundle_module.render_bundle(
        root,
        variant="narration--abc",
        chapters="2-4",
        speech="1,3",
        renderplan_root="renderplans",
        output_root="outputs",
        cache_dir="cache",
        force=True,
        reset_stale=True,
        emphasis_policy="defer-to-compose",
        fail_fast=True,
        on_progress=progress,
    )

    assert result.kind == "book"
    assert result.value is rendered
    assert result.exit_code == 1
    assert result.to_dict() == {
        "format": "voicerender-bundle-result",
        "version": 1,
        "kind": "book",
        "result": {"status": "complete", "exit_code": 1},
    }
    assert calls == {
        "path": root.resolve(),
        "kwargs": {
            "variant": "narration--abc",
            "chapters": "2-4",
            "speech": "1,3",
            "renderplan_root": "renderplans",
            "output_root": "outputs",
            "cache_dir": "cache",
            "force": True,
            "reset_stale_output": True,
            "fail_fast": True,
            "emphasis_policy": bundle_module.EmphasisPolicy.DEFER_TO_COMPOSE,
            "on_event": progress,
        },
    }


def test_render_bundle_dispatches_document_and_ignores_fail_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _bundle(tmp_path / "document.ssmd-bundle", "ssmdconvert.document")
    calls: dict[str, Any] = {}
    rendered = _result()
    progress = lambda _event: None

    def fake_render_document(path: Path, **kwargs: Any) -> SimpleNamespace:
        calls["path"] = path
        calls["kwargs"] = kwargs
        return rendered

    monkeypatch.setattr(
        bundle_module,
        "render_book",
        lambda *_args, **_kwargs: pytest.fail("document bundle dispatched as book"),
    )
    monkeypatch.setattr(bundle_module, "render_document", fake_render_document)

    result = bundle_module.render_bundle(
        root,
        variant="narration--abc",
        chapters=" ALL ",
        speech="3-5",
        reset_stale=True,
        fail_fast=True,
        on_progress=progress,
    )

    assert result.kind == "document"
    assert result.value is rendered
    assert calls == {
        "path": root.resolve(),
        "kwargs": {
            "variant": "narration--abc",
            "speech": "3-5",
            "renderplan_root": None,
            "output_root": None,
            "cache_dir": None,
            "force": False,
            "reset_stale_output": True,
            "emphasis_policy": bundle_module.EmphasisPolicy.REJECT,
            "on_event": progress,
        },
    }


def test_document_bundle_rejects_chapter_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _bundle(tmp_path / "document.ssmd-bundle", "ssmdconvert.document")
    monkeypatch.setattr(
        bundle_module,
        "render_document",
        lambda *_args, **_kwargs: pytest.fail("invalid selection reached renderer"),
    )

    with pytest.raises(ValueError, match="--chapters must be 'all'"):
        bundle_module.render_bundle(root, variant="narration--abc", chapters="1")


def test_bundle_render_parser_and_removed_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from voicerender.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["bundle", "render", "sample.ssmd-bundle", "--variant", "v"]
    )
    assert args.command == "bundle"
    assert args.bundle_command == "render"
    assert args.chapters == args.speech == "all"
    for removed in ("render" + "-book", "render" + "-document"):
        with pytest.raises(SystemExit):
            parser.parse_args([removed, "sample.ssmd-bundle", "--variant", "v"])
    with pytest.raises(SystemExit) as help_result:
        parser.parse_args(["bundle", "render", "--help"])
    assert help_result.value.code == 0
    help_output = capsys.readouterr().out
    assert "--variant" in help_output
    assert "--chapters" in help_output


def test_bundle_cli_emits_neutral_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from voicerender import cli

    result = SimpleNamespace(
        exit_code=0,
        to_dict=lambda: {
            "format": "voicerender-bundle-result",
            "version": 1,
            "kind": "book",
            "result": {"selected": 2},
        },
    )
    calls: dict[str, Any] = {}

    def fake_render_bundle(bundle: Path, **kwargs: Any) -> SimpleNamespace:
        calls["bundle"] = bundle
        calls["kwargs"] = kwargs
        return result

    monkeypatch.setattr(cli, "render_bundle", fake_render_bundle)
    exit_code = cli.main(
        [
            "bundle",
            "render",
            "sample.ssmd-bundle",
            "--variant",
            "narration--abc",
            "--chapters",
            "2-3",
            "--fail-fast",
            "--json",
            "--no-progress",
        ]
    )

    assert exit_code == 0
    assert calls["bundle"] == Path("sample.ssmd-bundle")
    assert calls["kwargs"]["chapters"] == "2-3"
    assert calls["kwargs"]["fail_fast"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload == result.to_dict()


def test_bundle_cli_prints_neutral_human_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from voicerender import cli

    request = SimpleNamespace(
        workspace=Path("story.ssmd-bundle"),
        variant_name="narration--abc",
        output_dir=Path("voice/narration--abc/document-0001"),
        speech_selector="all",
        report_path=Path("voice/report.toml"),
        preflight_report=None,
        error_code=None,
        pending_composition=False,
    )
    value = SimpleNamespace(
        request=request,
        status="complete",
        report_path=Path("voice/report.toml"),
        rendered=1,
        reused=0,
        error_code=None,
        error_message=None,
        preflight_report=None,
        pending_composition=False,
    )
    result = SimpleNamespace(kind="document", value=value, exit_code=0)
    monkeypatch.setattr(cli, "render_bundle", lambda *_args, **_kwargs: result)

    assert (
        cli.main(
            [
                "bundle",
                "render",
                "story.ssmd-bundle",
                "--variant",
                "narration--abc",
                "--no-progress",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "VoiceRender bundle render" in output
    assert "kind:      document" in output
    assert "workspace: story.ssmd-bundle" in output
    assert "selected:  1" in output
