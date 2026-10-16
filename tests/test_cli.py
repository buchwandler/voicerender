from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_primary_help_is_renderplan_centered_and_has_no_legacy_render_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    root_help = subprocess.run(
        [sys.executable, "-m", "voicerender", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.casefold()
    render_help = subprocess.run(
        [sys.executable, "-m", "voicerender", "render", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.casefold()
    book_help = subprocess.run(
        [sys.executable, "-m", "voicerender", "render-book", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.casefold()

    assert "preflight" in root_help
    assert "renderplan" in root_help
    assert "render-book" in root_help
    assert "utterplan" not in root_help
    assert "--preflight" in render_help
    assert "--out-dir" in render_help
    assert "--voice" not in render_help
    assert "--take" not in render_help
    assert "--profile" not in render_help
    assert "--variant" in book_help
    assert "--chapters" in book_help
    assert "--speech" in book_help
    assert "utterplan" not in book_help


def test_package_imports_when_legacy_planner_package_is_unavailable() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class RejectLegacyPlanner(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0].casefold() == 'utterplan':
            raise AssertionError('unexpected legacy planner import: ' + fullname)
        return None

sys.meta_path.insert(0, RejectLegacyPlanner())
import voicerender
import voicerender.book
import voicerender.cli
assert callable(voicerender.preflight_render_plan)
assert callable(voicerender.render_renderplan)
args = voicerender.cli.build_parser().parse_args(['render-book', 'book', '--variant', 'variant'])
assert args.command == 'render-book'
assert not any(name.split('.', 1)[0].casefold() == 'utterplan' for name in sys.modules)
assert not any(name.startswith('voicerender.inputs.utterplan') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_declared_renderplan_floor_exposes_book_set_contract() -> None:
    from renderplan import RenderPlanSet

    import voicerender.book

    assert voicerender.book is not None
    assert RenderPlanSet is not None


def test_direct_v5_consumer_files_are_retired() -> None:
    root = Path(__file__).resolve().parents[1]
    retired = (
        root / "voicerender" / "render_v5.py",
        root / "voicerender" / "manifest.py",
        root / "voicerender" / "handoff.py",
        root / "voicerender" / "inputs" / "utterplan_v5.py",
        root / "tests" / "test_render_v5.py",
        root / "tests" / "test_utterplan_v5.py",
        root / "tests" / "test_cli_v5.py",
        root / "tests" / "test_manifest.py",
        root / "tests" / "fixtures" / "utterplan_v5_contract.toml",
        root / "docs" / "utterplan-v5-integration.md",
        root / "examples" / "plans" / "chapter.utterplan.toml",
        root / "examples" / "voices" / "voice.toml",
        root / "voicerender" / "schemas" / "render-manifest-v2.schema.json",
    )
    assert not [path for path in retired if path.exists()]
    project_metadata = (root / "pyproject.toml").read_text(encoding="utf-8").casefold()
    assert "utterplan" not in project_metadata


def test_shared_example_plan_and_profile_are_identity_matched() -> None:
    from renderplan import load_plan, load_profile

    root = Path(__file__).resolve().parents[1]
    profile = load_profile(root / "examples/voices/narration.renderprofile.toml")
    plan = load_plan(root / "examples/plans/chapter.renderplan.toml")

    assert plan.render_profile_id == profile.profile_id
    assert plan.speech[0].binding == profile.roles["narrator"].binding
    assert plan.speech[0].text == "A small shared RenderPlan example."


def test_cli_emphasis_policy_defaults_to_reject_and_can_defer() -> None:
    from voicerender.cli import build_parser

    parser = build_parser()
    commands = (
        ["preflight", "plan.toml", "--output", "preflight.toml"],
        [
            "render",
            "plan.toml",
            "--preflight",
            "preflight.toml",
            "--out-dir",
            "rendered",
        ],
        ["render-book", "book", "--variant", "narration"],
    )
    for command in commands:
        assert parser.parse_args(command).emphasis_policy == "reject"
        assert (
            parser.parse_args(
                [*command, "--emphasis-policy", "defer-to-compose"]
            ).emphasis_policy
            == "defer-to-compose"
        )


def test_profile_create_reference_arguments_are_mutually_exclusive() -> None:
    import pytest

    from voicerender.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "profile",
                "create",
                "--engine",
                "sopro",
                "--target",
                "target-v1",
                "--language",
                "en-US",
                "--voice",
                "voice-a",
                "--reference-wav",
                "speaker.wav",
                "--output",
                "profile.toml",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "profile",
                "create",
                "--engine",
                "qwen3",
                "--target",
                "base-0.6b",
                "--language",
                "en-US",
                "--prepared-voice",
                "speaker.qvoice",
                "--voice-source",
                "source.toml",
                "--output",
                "profile.toml",
            ]
        )


def test_cli_profile_create_reference_wav_persists_computed_pin(
    monkeypatch, tmp_path: Path
) -> None:
    import hashlib
    from types import SimpleNamespace

    from voicerender import cli

    reference = tmp_path / "speaker.wav"
    audio = b"speaker reference"
    reference.write_bytes(audio)
    captured = {}
    monkeypatch.setattr(
        cli,
        "create_render_profile",
        lambda **kwargs: (
            captured.update(kwargs) or SimpleNamespace(profile_id="profile-1")
        ),
    )
    monkeypatch.setattr(cli, "save_render_profile", lambda profile, path: path)

    result = cli.main(
        [
            "profile",
            "create",
            "--engine",
            "sopro",
            "--target",
            "target-v1",
            "--language",
            "en-US",
            "--reference-wav",
            str(reference),
            "--output",
            str(tmp_path / "speaker.renderprofile.toml"),
            "--json",
        ]
    )

    assert result == 0
    assert captured["voice_source"] == {
        "kind": "reference",
        "value": str(reference.resolve()),
        "sha256": hashlib.sha256(audio).hexdigest(),
    }


def test_cli_profile_create_voicefolio_ref_uses_full_pinned_source(
    monkeypatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from voicerender import cli

    source = {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "value": "catalog:voice-id",
        "sha256": "a" * 64,
        "source_revision": "revision-1",
        "source_repository": "voice-repo",
        "source_path": "voices/reference.wav",
    }
    captured = {}
    monkeypatch.setattr(cli, "managed_reference_source", lambda ref: source)
    monkeypatch.setattr(
        cli,
        "create_render_profile",
        lambda **kwargs: (
            captured.update(kwargs) or SimpleNamespace(profile_id="profile-2")
        ),
    )
    monkeypatch.setattr(cli, "save_render_profile", lambda profile, path: path)

    result = cli.main(
        [
            "profile",
            "create",
            "--engine",
            "sopro",
            "--target",
            "target-v1",
            "--language",
            "en-US",
            "--voicefolio-ref",
            source["value"],
            "--output",
            str(tmp_path / "managed.renderprofile.toml"),
            "--json",
        ]
    )

    assert result == 0
    assert captured["voice_source"] == source


def _qwen_origin() -> dict[str, object]:
    return {
        "schema": "voicerender.voice-origin.qwen3-design.v1",
        "generator": "qwen3",
        "target_id": "voice-design-1.7b",
        "model_revision": "design-revision",
        "qwen3synth_version": "0.1.0",
        "onnxvoice_version": "0.1.0",
        "prompt": "Warm, measured voice.",
        "reference_text": "  This exact sample.  ",
        "language": "en",
        "quality": "int4",
        "provider": "cpu",
        "temperature": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_new_tokens": 128,
        "seed": 42,
    }


def _write_qwen_source(tmp_path: Path):
    from voicerender.voice_source_codec import save_voice_source
    from voicerender.voice_sources import local_reference_source

    reference = tmp_path / "designed.wav"
    reference.write_bytes(b"pinned reference")
    source = local_reference_source(reference)
    source["origin"] = _qwen_origin()
    sidecar = tmp_path / "designed.voice-source.toml"
    save_voice_source(source, sidecar)
    return source, reference, sidecar


def test_cli_profile_create_preserves_source_sidecar_and_rejects_output_collision(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from types import SimpleNamespace

    from voicerender import cli
    from voicerender.voice_source_codec import load_voice_source

    source, _, sidecar = _write_qwen_source(tmp_path)
    captured = {}
    monkeypatch.setattr(
        cli,
        "create_render_profile",
        lambda **kwargs: (
            captured.update(kwargs) or SimpleNamespace(profile_id="profile-3")
        ),
    )
    monkeypatch.setattr(cli, "save_render_profile", lambda profile, path: path)
    monkeypatch.setattr(
        cli,
        "design_qwen3_reference",
        lambda *args, **kwargs: pytest.fail(
            "profile creation must not generate a voice"
        ),
    )

    result = cli.main(
        [
            "profile",
            "create",
            "--engine",
            "pocket",
            "--target",
            "pocket-test",
            "--language",
            "en-US",
            "--voice-source",
            str(sidecar),
            "--output",
            str(tmp_path / "pocket.renderprofile.toml"),
            "--json",
        ]
    )
    assert result == 0
    assert captured["voice_source"] == load_voice_source(sidecar) == source

    result = cli.main(
        [
            "profile",
            "create",
            "--engine",
            "pocket",
            "--target",
            "pocket-test",
            "--language",
            "en-US",
            "--voice-source",
            str(sidecar),
            "--output",
            str(sidecar),
        ]
    )
    assert result == 1
    assert "must not overwrite" in capsys.readouterr().err


def test_cli_prepared_voice_profile_inspects_without_opening_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from voicerender import cli

    prepared = tmp_path / "speaker.qvoice"
    prepared.write_bytes(b"qvoice bytes")
    info = SimpleNamespace(
        model_id="wavekat/base-0.6b",
        model_revision="base-revision",
        layout="qwen3-wavekat-v1",
    )
    fake_module = SimpleNamespace(
        inspect_voice_profile=lambda path: info,
        Qwen3Runtime=SimpleNamespace(
            from_installed=lambda *args, **kwargs: pytest.fail(
                "profile creation must not open a runtime"
            )
        ),
    )
    monkeypatch.setattr(cli, "_qwen3_module", lambda: fake_module)
    captured = {}
    monkeypatch.setattr(
        cli,
        "create_render_profile",
        lambda **kwargs: (
            captured.update(kwargs) or SimpleNamespace(profile_id="qwen-profile")
        ),
    )
    monkeypatch.setattr(cli, "save_render_profile", lambda profile, path: path)

    result = cli.main(
        [
            "profile",
            "create",
            "--engine",
            "qwen3",
            "--target",
            "base-0.6b",
            "--language",
            "en-US",
            "--prepared-voice",
            str(prepared),
            "--output",
            str(tmp_path / "qwen.renderprofile.toml"),
        ]
    )

    assert result == 0
    assert captured["voice_source"] == {
        "kind": "prepared",
        "format": "qwen3.qvoice.v2",
        "value": str(prepared.resolve()),
        "sha256": __import__("hashlib").sha256(b"qvoice bytes").hexdigest(),
        "model_id": info.model_id,
        "model_revision": info.model_revision,
        "layout": info.layout,
    }


def test_cli_prepare_qwen_reuses_exact_origin_transcript(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from types import ModuleType, SimpleNamespace

    from voicerender import cli

    _, _, sidecar = _write_qwen_source(tmp_path)
    events = []

    class FakePrepared:
        model_id = "wavekat/base-0.6b"
        model_revision = "base-revision"
        layout = "qwen3-wavekat-v1"

        def save(self, path: Path) -> None:
            path.write_bytes(b"prepared qvoice")

    class FakeRuntime:
        @classmethod
        def from_installed(cls, target_id: str, **kwargs):
            events.append(("open", target_id, kwargs))
            return cls()

        def prepare_voice(self, reference):
            events.append(("prepare", reference))
            return FakePrepared()

        def close(self) -> None:
            events.append(("close", None))

    module = ModuleType("qwen3synth")
    module.__version__ = "0.1.0"
    module.Qwen3Runtime = FakeRuntime
    module.ReferenceVoice = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setattr(cli, "_qwen3_module", lambda: module)
    output = tmp_path / "prepared.qvoice"

    result = cli.main(
        [
            "voice",
            "prepare-qwen",
            "--voice-source",
            str(sidecar),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert result == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert (
        payload["sha256"]
        == __import__("hashlib").sha256(b"prepared qvoice").hexdigest()
    )
    assert [event[0] for event in events] == ["open", "prepare", "close"]
    reference = events[1][1]
    assert reference.transcript == "  This exact sample.  "
    assert reference.language == "en"
    assert reference.source_id


def test_cli_prepare_qwen_rejects_arbitrary_wav_without_transcript(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from voicerender import cli

    reference = tmp_path / "arbitrary.wav"
    reference.write_bytes(b"audio without a transcript")
    monkeypatch.setattr(
        cli,
        "_qwen3_module",
        lambda: pytest.fail("runtime must not open without an exact transcript"),
    )

    result = cli.main(
        [
            "voice",
            "prepare-qwen",
            "--reference",
            str(reference),
            "--output",
            str(tmp_path / "prepared.qvoice"),
        ]
    )

    assert result == 1
    assert "requires --transcript" in capsys.readouterr().err


def test_cli_voice_inspect_handles_source_sidecar_and_qvoice(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from types import SimpleNamespace

    from voicerender import cli

    source, _, sidecar = _write_qwen_source(tmp_path)
    result = cli.main(["voice", "inspect", str(sidecar), "--json"])
    assert result == 0
    source_payload = __import__("json").loads(capsys.readouterr().out)
    assert source_payload["voice_source"] == source

    profile = tmp_path / "speaker.qvoice"
    profile.write_bytes(b"qvoice bytes")
    info = SimpleNamespace(
        profile_sha256="c" * 64,
        model_id="wavekat/base-0.6b",
        model_revision="base-revision",
        layout="qwen3-wavekat-v1",
        language="en",
        reference_samples=24_000,
    )
    monkeypatch.setattr(
        cli,
        "_qwen3_module",
        lambda: SimpleNamespace(inspect_voice_profile=lambda path: info),
    )
    result = cli.main(["voice", "inspect", str(profile), "--json"])
    assert result == 0
    profile_payload = __import__("json").loads(capsys.readouterr().out)
    assert profile_payload["kind"] == "prepared"
    assert profile_payload["sha256"] == info.profile_sha256
    assert profile_payload["model_id"] == info.model_id


def test_cli_voice_design_preserves_recipe_text(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from types import SimpleNamespace

    from voicerender import cli

    captured = {}
    output = tmp_path / "designed.wav"
    monkeypatch.setattr(
        cli,
        "design_qwen3_reference",
        lambda recipe, path, **kwargs: (
            captured.update(recipe=recipe, path=path, options=kwargs)
            or SimpleNamespace(
                path=path,
                sha256="a" * 64,
                source={"kind": "reference"},
                model_id="wavekat/voice-design-1.7b",
                runtime_identity={"engine": "qwen3synth"},
            )
        ),
    )

    result = cli.main(
        [
            "voice",
            "design",
            "--prompt",
            " Warm voice. ",
            "--text",
            "  Exact reference text.  ",
            "--output",
            str(output),
            "--json",
        ]
    )

    assert result == 0
    assert captured["recipe"].prompt == " Warm voice. "
    assert captured["recipe"].reference_text == "  Exact reference text.  "
    assert captured["path"] == output
    assert "runtime_identity" in __import__("json").loads(capsys.readouterr().out)
