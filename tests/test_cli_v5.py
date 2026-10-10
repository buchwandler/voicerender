from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from voicerender import (
    EngineApiProbe,
    EngineCapabilities,
    EngineSelection,
    RenderedSpeech,
    RequestMeasure,
    SpeechRequest,
    cli,
    render_v5,
)
from voicerender.engines.registry import _ADAPTERS
from voicerender.inputs.utterplan_v5 import load_utterplan_v5
from voicerender.manifest import MANIFEST_SCHEMA


def _result():
    return SimpleNamespace(
        as_dict=lambda: {"rendered": 1, "reused": 0, "failed": 0, "selected": 1},
        failed_count=0,
        manifest_path="renders/manifest.json",
    )


class _FakeSession:
    def measure(self, request: SpeechRequest) -> RequestMeasure:
        return RequestMeasure(
            fits=True,
            amount=len(request.text),
            maximum=len(request.text),
            unit="unknown",
            source="cli-test",
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        return RenderedSpeech(
            id=request.id,
            audio=np.full(24, 0.1, dtype=np.float32),
            sample_rate=16_000,
        )


class _FakeAdapter:
    id = "cli-fake"
    package_name = "cli-fake"

    def __init__(self):
        self.open_count = 0

    def probe_api(self) -> EngineApiProbe:
        return EngineApiProbe(
            engine=self.id,
            package=self.package_name,
            compatible=True,
            status="ready",
            distribution_version="1.0",
            expected_api_version=1,
            api_version=1,
            contract_source="explicit",
        )

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(id=self.id, supports_request_measurement=True)

    def version(self) -> str:
        return "1.0"

    def resolve(self, request):
        return (
            EngineSelection(
                engine=self.id,
                target_id=request.target_id,
                language=request.language,
                voice=request.voice,
                speaker=request.speaker,
                options=request.options,
            ),
            (),
        )

    def validate_selection(self, selection):
        return ()

    def canonical_synthesis_identity(self, selection):
        return {"engine": self.id, "target_id": selection.target_id}

    @contextmanager
    def open(self, selection):
        self.open_count += 1
        yield _FakeSession()


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


def test_cli_standalone_validate_render_inspect_with_fake_engine(
    monkeypatch, tmp_path, capsys
):
    adapter = _FakeAdapter()
    monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
    plan = Path(__file__).resolve().parents[1] / "examples/plans/chapter.utterplan.toml"
    profile = tmp_path / "voice.toml"
    profile.write_text(
        'schema_version = 1\nname = "cli-profile"\n'
        '[defaults]\nrole = "narrator"\n'
        '[roles.narrator]\nengine = "cli-fake"\ntarget_id = "model-1"\n',
        encoding="utf-8",
    )

    assert cli.main(["validate", str(plan), "--voice", str(profile), "--json"]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["valid"] is True
    assert validation["engines"][0]["id"] == adapter.id
    assert adapter.open_count == 0

    output = tmp_path / "render"
    assert (
        cli.main(
            ["render", str(plan), "--voice", str(profile), "--out-dir", str(output)]
        )
        == 0
    )
    assert adapter.open_count == 1
    capsys.readouterr()

    manifest_path = output / "render-manifest.json"
    assert cli.main(["inspect", str(manifest_path), "--verify", "--json"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["complete"] is True
    assert inspection["valid_wavs"] == 1


def test_audiojob_command_is_not_part_of_voice_render_cli():
    import pytest

    with pytest.raises(SystemExit):
        cli._parser().parse_args(["audiojob"])
