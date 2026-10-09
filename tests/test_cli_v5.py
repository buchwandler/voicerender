from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

from voicerender import cli, render_v5
from voicerender.inputs.utterplan_v5 import load_utterplan_v5
from voicerender.manifest import MANIFEST_SCHEMA


def _result():
    return SimpleNamespace(
        as_dict=lambda: {"rendered": 1, "reused": 0, "failed": 0, "selected": 1},
        failed_count=0,
        manifest_path="renders/manifest.json",
    )


def test_example_plan_is_valid_utterplan_v5():
    plan = load_utterplan_v5("examples/plans/chapter.utterplan.toml")
    assert plan.plan_id.startswith("sha256:")
    assert len(plan.segments) == 1
    assert plan.segments[0].text == "Hello from UtterPlan v5."


def test_cli_voice_flag_dispatches_direct_v5_and_keeps_legacy_alias(
    monkeypatch, tmp_path
):
    calls = []

    def direct(plan, **kwargs):
        calls.append(("v5", plan, kwargs))
        return _result()

    def legacy(plan, **kwargs):
        calls.append(("legacy", plan, kwargs))
        return _result()

    monkeypatch.setattr(render_v5, "render_utterplan_v5", direct)
    monkeypatch.setattr(cli, "render_plan", legacy)

    assert (
        cli.main(
            [
                "render",
                "chapter.utterplan.toml",
                "--voice",
                "voice.toml",
                "--out-dir",
                str(tmp_path / "v5"),
                "--select",
                "f000000.s000",
            ]
        )
        == 0
    )
    assert calls[-1][0] == "v5"
    assert calls[-1][2]["profile"] == cli.Path("voice.toml")
    assert calls[-1][2]["select"] == ["f000000.s000"]

    assert (
        cli.main(
            [
                "render",
                "legacy.voiceplan.toml",
                "--profile",
                "legacy-profile.toml",
                "--out-dir",
                str(tmp_path / "legacy"),
            ]
        )
        == 0
    )
    assert calls[-1][0] == "legacy"


def test_cli_inspect_verifies_manifest_v2(monkeypatch, tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": MANIFEST_SCHEMA}), encoding="utf-8")
    verified = {
        "schema": MANIFEST_SCHEMA,
        "manifest_id": "sha256:abc",
        "complete": True,
        "segments": [{"source": {"locator": "f000000.s000"}, "status": "rendered"}],
    }
    calls = []

    def read_manifest(path, *, verify_audio):
        calls.append((path, verify_audio))
        return verified

    monkeypatch.setattr(cli, "read_manifest", read_manifest)
    assert cli.main(["inspect", str(manifest_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == MANIFEST_SCHEMA
    assert payload["valid_wavs"] == 1
    assert calls[0][1] is True


def test_core_import_does_not_load_audiocompose():
    code = (
        "import sys, voicerender, voicerender.render_v5; "
        "assert not any(name == 'audiocompose' or "
        "name.startswith('audiocompose.') for name in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
