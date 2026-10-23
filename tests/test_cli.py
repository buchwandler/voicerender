from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_primary_help_is_renderplan_centered_and_exposes_bundle_render() -> None:
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
    bundle_help = subprocess.run(
        [sys.executable, "-m", "voicerender", "bundle", "render", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.casefold()
    variants_help = subprocess.run(
        [sys.executable, "-m", "voicerender", "bundle", "variants", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.casefold()
    bundle_help = " ".join(bundle_help.split())
    variants_help = " ".join(variants_help.split())

    assert "preflight" in root_help
    assert "renderplan" in root_help
    assert "bundle" in root_help
    assert "utterplan" not in root_help
    assert "--preflight" in render_help
    assert "--out-dir" in render_help
    assert "--voice" not in render_help
    assert "--take" not in render_help
    assert "--profile" not in render_help
    assert "--variant" in bundle_help
    assert "--chapters" in bundle_help
    assert "--speech" in bundle_help
    assert "utterplan" not in bundle_help

    assert "--renderplan-root" in variants_help
    assert "--output-root" in variants_help
    assert "--json" in variants_help

    assert "path resolving to a canonical name.ssmd-bundle directory" in bundle_help
    assert "path resolving to a canonical name.ssmd-bundle directory" in variants_help
    assert "positional path must resolve to the bundle root" in variants_help
    assert (
        "--renderplan-root only overrides where variant directories are read from"
        in variants_help
    )


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
assert callable(voicerender.list_bundle_variants)
assert voicerender.BundleVariant is not None
assert voicerender.BundleVariantsResult is not None
args = voicerender.cli.build_parser().parse_args(['bundle', 'render', 'book.ssmd-bundle', '--variant', 'variant'])
assert args.command == 'bundle' and args.bundle_command == 'render'
args = voicerender.cli.build_parser().parse_args(['bundle', 'variants', 'book.ssmd-bundle', '--json'])
assert args.command == 'bundle' and args.bundle_command == 'variants'
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
        ["bundle", "render", "book.ssmd-bundle", "--variant", "narration"],
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
            "--transcript",
            "Exact reference words.",
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

    assert captured["transcript"] == "Exact reference words."


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

    collision_source = tmp_path / "designed.voice-source.renderprofile.toml"
    collision_source.write_bytes(sidecar.read_bytes())

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
            str(collision_source),
            "--output",
            str(collision_source),
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


def test_cli_progress_printer_formats_tty_and_cleans_up_line() -> None:
    import io

    from voicerender.cli import _ProgressPrinter
    from voicerender.events import RenderEvent, RenderEventKind

    stream = io.StringIO()
    printer = _ProgressPrinter(stream, is_tty=True, clock=lambda: 0.0)
    printer(
        RenderEvent(
            kind=RenderEventKind.RENDER_STARTED,
            scope="render",
            phase="render",
            completed_count=118,
            total_count=319,
            progress=0.37,
            rendered_count=91,
            reused_count=27,
            elapsed_seconds=761,
            eta_seconds=1295,
        )
    )
    printer.finish()

    output = stream.getvalue()
    assert output.startswith("\r[ 37.0%] 118/319 speech")
    assert "91 rendered, 27 reused" in output
    assert "elapsed 12:41 | ETA 21:35" in output
    assert output.endswith("\n")
    assert "\x1b" not in output


def test_cli_progress_printer_throttles_non_tty_with_injected_clock() -> None:
    import io

    from voicerender.cli import _ProgressPrinter
    from voicerender.events import RenderEvent, RenderEventKind

    now = [0.0]
    stream = io.StringIO()
    printer = _ProgressPrinter(stream, is_tty=False, clock=lambda: now[0])

    def event(kind: RenderEventKind, progress: float) -> RenderEvent:
        completed = int(progress * 100)
        return RenderEvent(
            kind=kind,
            scope="render",
            phase="render",
            completed_count=completed,
            total_count=100,
            progress=progress,
            elapsed_seconds=now[0],
        )

    printer(event(RenderEventKind.RENDER_STARTED, 0.0))
    now[0] = 1.0
    printer(event(RenderEventKind.SEGMENT_RENDERED, 0.02))
    now[0] = 2.0
    printer(event(RenderEventKind.SEGMENT_RENDERED, 0.05))
    now[0] = 32.0
    printer(event(RenderEventKind.SEGMENT_STARTED, 0.05))
    now[0] = 33.0
    printer(event(RenderEventKind.SEGMENT_FAILED, 0.05))

    output = stream.getvalue()
    assert len(output.splitlines()) == 4
    assert "progress 5.0% 5/100" in output
    assert "failed_speech_item" in output
    assert "\r" not in output and "\x1b" not in output


def test_cli_progress_auto_force_disable_and_json_modes() -> None:
    import io
    from types import SimpleNamespace

    from voicerender.cli import _progress_callback, build_parser

    class TestStream(io.StringIO):
        def __init__(self, *, tty: bool) -> None:
            super().__init__()
            self.tty = tty

        def isatty(self) -> bool:
            return self.tty

    parser = build_parser()
    render_args = ["render", "plan", "--preflight", "report", "--out-dir", "out"]
    assert parser.parse_args(render_args).progress is None
    assert parser.parse_args([*render_args, "--no-progress"]).progress is False

    tty = TestStream(tty=True)
    assert _progress_callback(SimpleNamespace(progress=None, json=False), stream=tty)
    assert (
        _progress_callback(SimpleNamespace(progress=None, json=True), stream=tty)
        is None
    )
    assert (
        _progress_callback(SimpleNamespace(progress=False, json=False), stream=tty)
        is None
    )

    non_tty = TestStream(tty=False)
    assert _progress_callback(SimpleNamespace(progress=True, json=True), stream=non_tty)
    assert (
        _progress_callback(SimpleNamespace(progress=None, json=False), stream=non_tty)
        is None
    )


def test_preflight_cli_forced_progress_stays_on_stderr_with_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    from types import SimpleNamespace

    from voicerender import cli
    from voicerender.events import RenderEvent, RenderEventKind

    class NonTtyStream(io.StringIO):
        def isatty(self) -> bool:
            return False

    stderr = NonTtyStream()
    monkeypatch.setattr(cli.sys, "stderr", stderr)
    monkeypatch.setattr(cli, "load_plan", lambda _path: object())

    def fake_preflight(_plan, _output, *, on_event, **_kwargs):
        assert on_event is not None
        on_event(
            RenderEvent(
                kind=RenderEventKind.PREFLIGHT_STARTED,
                scope="preflight",
                phase="preflight",
                completed_count=0,
                total_count=2,
                progress=0.0,
                phase_completed_count=0,
                phase_total_count=2,
                phase_progress=0.0,
                elapsed_seconds=0.0,
            )
        )
        on_event(
            RenderEvent(
                kind=RenderEventKind.PREFLIGHT_ITEM_FINISHED,
                scope="preflight",
                phase="preflight",
                completed_count=1,
                total_count=2,
                progress=0.5,
                phase_completed_count=1,
                phase_total_count=2,
                phase_progress=0.5,
                elapsed_seconds=2.0,
            )
        )
        return SimpleNamespace(
            checks=[],
            replan=[],
            status="ok",
            render_plan_id="plan",
            render_profile_id="profile",
        )

    monkeypatch.setattr(cli, "preflight_render_plan", fake_preflight)
    output = tmp_path / "preflight.toml"
    assert (
        cli.main(
            ["preflight", "plan.toml", "--output", str(output), "--json", "--progress"]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    payload = __import__("json").loads(stdout)
    assert payload["status"] == "ok"
    assert "preflight progress 0.0%" in stderr.getvalue()
    assert "preflight progress 50.0%" in stderr.getvalue()
    assert "\r" not in stderr.getvalue() and "\x1b" not in stderr.getvalue()


def test_preflight_cli_json_auto_and_disabled_progress_are_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    from types import SimpleNamespace

    from voicerender import cli
    from voicerender.events import RenderEvent, RenderEventKind

    class NonTtyStream(io.StringIO):
        def isatty(self) -> bool:
            return False

    stderr = NonTtyStream()
    monkeypatch.setattr(cli.sys, "stderr", stderr)
    monkeypatch.setattr(cli, "load_plan", lambda _path: object())
    callbacks: list[bool] = []

    def fake_preflight(_plan, _output, *, on_event, **_kwargs):
        callbacks.append(on_event is not None)
        if on_event is not None:
            on_event(
                RenderEvent(
                    kind=RenderEventKind.PREFLIGHT_STARTED,
                    scope="preflight",
                    phase="preflight",
                    completed_count=0,
                    total_count=1,
                    progress=0.0,
                    phase_completed_count=0,
                    phase_total_count=1,
                    phase_progress=0.0,
                    elapsed_seconds=0.0,
                )
            )
        return SimpleNamespace(
            checks=[],
            replan=[],
            status="ok",
            render_plan_id="plan",
            render_profile_id="profile",
        )

    monkeypatch.setattr(cli, "preflight_render_plan", fake_preflight)
    base = [
        "preflight",
        "plan.toml",
        "--output",
        str(tmp_path / "preflight.toml"),
        "--json",
    ]
    assert cli.main(base) == 0
    assert __import__("json").loads(capsys.readouterr().out)["status"] == "ok"
    assert stderr.getvalue() == ""

    assert cli.main([*base, "--no-progress"]) == 0
    assert __import__("json").loads(capsys.readouterr().out)["status"] == "ok"
    assert stderr.getvalue() == ""
    assert callbacks == [False, False]


@pytest.mark.parametrize("command", ["render", "bundle"])
def test_render_command_progress_callbacks_keep_json_stdout_clean(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import io
    from types import SimpleNamespace

    from voicerender import cli
    from voicerender.events import RenderEvent, RenderEventKind, RenderScope

    class NonTtyStream(io.StringIO):
        def isatty(self) -> bool:
            return False

    stderr = NonTtyStream()
    monkeypatch.setattr(cli.sys, "stderr", stderr)

    def event_callback(on_event, scope: RenderScope) -> None:
        assert on_event is not None
        on_event(
            RenderEvent(
                kind=RenderEventKind.SEGMENT_RENDERED,
                scope=scope,
                phase="render",
                completed_count=1,
                total_count=2,
                progress=0.5,
                rendered_count=1,
                reused_count=0,
                elapsed_seconds=1.0,
            )
        )

    if command == "render":

        def fake_render(*_args, on_event, **_kwargs):
            event_callback(on_event, "render")
            return SimpleNamespace(
                failed=False,
                pending_composition=False,
                rendered_count=1,
                reused_count=0,
            )

        monkeypatch.setattr(cli, "render_renderplan", fake_render)
        monkeypatch.setattr(cli, "_render_payload", lambda _result: {"speech_count": 2})
        args = [
            "render",
            "plan.toml",
            "--preflight",
            "preflight.toml",
            "--out-dir",
            str(tmp_path / "rendered"),
            "--json",
            "--progress",
        ]
    else:

        def fake_render_bundle(*_args, on_progress, **_kwargs):
            event_callback(on_progress, "book")
            return SimpleNamespace(to_dict=lambda: {"items": []}, exit_code=0)

        monkeypatch.setattr(cli, "render_bundle", fake_render_bundle)
        args = [
            "bundle",
            "render",
            str(tmp_path / "book.ssmd-bundle"),
            "--variant",
            "narration",
            "--json",
            "--progress",
        ]

    assert cli.main(args) == 0
    stdout = capsys.readouterr().out
    assert isinstance(__import__("json").loads(stdout), dict)
    assert "progress 50.0%" in stderr.getvalue()
    assert "\\r" not in stderr.getvalue() and "\\x1b" not in stderr.getvalue()


@pytest.mark.parametrize(
    ("supplied", "expected", "json_output"),
    [
        ("profile1", "profile1.renderprofile.toml", True),
        (
            "profile1.renderprofile.toml",
            "profile1.renderprofile.toml",
            False,
        ),
        ("profile1.toml", "profile1.renderprofile.toml", True),
        ("profiles/profile1", "profiles/profile1.renderprofile.toml", False),
    ],
)
def test_cli_profile_create_normalizes_output_path(
    monkeypatch, tmp_path: Path, capsys, supplied: str, expected: str, json_output: bool
) -> None:
    import json
    from types import SimpleNamespace

    from voicerender import cli

    destinations: list[Path] = []
    monkeypatch.setattr(
        cli,
        "create_render_profile",
        lambda **_kwargs: SimpleNamespace(profile_id="profile-1"),
    )
    monkeypatch.setattr(
        cli,
        "save_render_profile",
        lambda _profile, path: destinations.append(path) or path,
    )

    args = [
        "profile",
        "create",
        "--engine",
        "kokoro",
        "--target",
        "v1.0",
        "--voice",
        "af_heart",
        "--language",
        "en-us",
        "-o",
        supplied,
    ]
    if json_output:
        args.append("--json")

    assert cli.main(args) == 0
    destination = Path(expected)
    assert destinations == [destination]
    stdout = capsys.readouterr().out
    if json_output:
        assert json.loads(stdout)["path"] == str(destination)
    else:
        assert str(destination) in stdout


@pytest.mark.parametrize(
    "gain",
    [None, 0, 1.5, -1.5],
    ids=["unset", "zero", "positive", "negative"],
)
def test_cli_kokoro_profile_create_round_trips_voice_level_gain(
    monkeypatch, tmp_path: Path, capsys, gain: float | None
) -> None:
    import json
    from types import SimpleNamespace

    import tomlkit

    from voicerender import cli, models
    from voicerender.engines.api_probe import EngineApiProbe
    from voicerender.engines.registry import get_adapter
    from voicerender.renderprofile import load_render_profile

    adapter = get_adapter("kokoro")
    probe = EngineApiProbe(
        engine="kokoro",
        package="pykokoro",
        compatible=True,
        status="ready",
        distribution_version="0.10.5",
        expected_api_version=1,
        api_version=1,
        contract_source="explicit",
    )
    monkeypatch.setattr(adapter, "probe_api", lambda: probe)
    model = SimpleNamespace(
        id="v1.0",
        languages=("en-us",),
        voices=("am_michael",),
        qualities=("fp32",),
        metadata={"frontend": "kokoro"},
        sample_rate=24000,
        max_tokens=512,
        source="test",
        distribution_id="v1.0-test",
    )
    monkeypatch.setattr(models, "get_model_info", lambda *_args, **_kwargs: (model, ()))
    monkeypatch.chdir(tmp_path)

    args = [
        "profile",
        "create",
        "--engine",
        "kokoro",
        "--target",
        "v1.0",
        "--voice",
        "am_michael",
        "--language",
        "en-us",
        "-o",
        "profile1",
    ]
    if gain is not None:
        args.extend(
            [
                "--option",
                f"voice_level={json.dumps({'mode': 'off', 'gain_db': gain})}",
            ]
        )

    assert cli.main(args) == 0
    output = tmp_path / "profile1.renderprofile.toml"
    assert output.is_file()
    assert "profile1.renderprofile.toml" in capsys.readouterr().out
    document = tomlkit.parse(output.read_text(encoding="utf-8"))
    binding = document["roles"]["narrator"]["binding"]
    assert binding["renderer"] == "kokoro"
    assert binding["target_id"] == "v1.0"
    assert binding["voice"] == "am_michael"
    assert binding["options"]["voice_level"]["mode"] == "off"
    if gain is None:
        assert "gain_db" not in binding["options"]["voice_level"]
    else:
        assert binding["options"]["voice_level"]["gain_db"] == float(gain)

    profile = load_render_profile(output)
    role = profile.roles["narrator"]
    assert role.language == "en-us"
    assert role.binding["renderer"] == "kokoro"
    assert role.binding["target_id"] == "v1.0"
    assert role.binding["voice"] == "am_michael"
    loaded_voice_level = role.binding["options"]["voice_level"]
    assert loaded_voice_level["mode"] == "off"
    if gain is None:
        assert "gain_db" not in loaded_voice_level
    else:
        assert loaded_voice_level["gain_db"] == float(gain)


def test_profile_output_normalizer_is_idempotent_and_case_insensitive() -> None:
    from voicerender.cli import normalize_renderprofile_output_path

    normalized = normalize_renderprofile_output_path(Path("profiles/Profile1"))
    assert normalized == Path("profiles/Profile1.renderprofile.toml")
    assert normalize_renderprofile_output_path(normalized) == normalized
    uppercase = Path("PROFILE.RENDERPROFILE.TOML")
    assert normalize_renderprofile_output_path(uppercase) == uppercase


def test_cli_profile_create_prevents_overwriting_voice_source_after_normalization(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from voicerender import cli

    source = tmp_path / "profile.renderprofile.toml"
    monkeypatch.setattr(
        cli,
        "create_render_profile",
        lambda **_kwargs: pytest.fail(
            "profile creation must reject the collision first"
        ),
    )
    assert (
        cli.main(
            [
                "profile",
                "create",
                "--engine",
                "kokoro",
                "--target",
                "v1.0",
                "--reference-wav",
                str(source),
                "--language",
                "en-us",
                "-o",
                str(tmp_path / "profile"),
            ]
        )
        == 1
    )
    assert "must not overwrite a voice-source input" in capsys.readouterr().err


def test_bundle_variants_cli_json_human_and_root_overrides(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from voicerender import cli
    from voicerender.variants import BundleVariant, BundleVariantsResult

    bundle = tmp_path / "story.ssmd-bundle"
    renderplans = tmp_path / "plans"
    output = tmp_path / "output"
    variant = BundleVariant(
        name="narration--a1b2c3d4e5f6",
        profile_id="sha256:profile",
        plan_status="valid",
        validation_error=None,
        render_status="partial",
        state="partial",
        unit_kind="chapters",
        completed_units=1,
        total_units=2,
        planned_speech=13,
        available_speech=5,
        pending_composition=True,
        report_path=None,
    )
    listing = BundleVariantsResult(
        bundle=bundle,
        kind="book",
        renderplan_root=renderplans,
        output_root=output,
        variants=(variant,),
    )
    calls: list[tuple[Path, dict[str, Path | None]]] = []

    def fake_list(path, *, renderplan_root=None, output_root=None):
        calls.append(
            (path, {"renderplan_root": renderplan_root, "output_root": output_root})
        )
        return listing

    monkeypatch.setattr(cli, "list_bundle_variants", fake_list)
    assert (
        cli.main(
            [
                "bundle",
                "variants",
                str(bundle),
                "--renderplan-root",
                str(renderplans),
                "--output-root",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["format"] == "voicerender-bundle-variants"
    assert payload["version"] == 1
    assert payload["variants"][0]["name"] == variant.name
    assert calls[-1] == (
        bundle,
        {"renderplan_root": renderplans, "output_root": output},
    )

    assert cli.main(["bundle", "variants", str(bundle)]) == 0
    human = capsys.readouterr().out
    assert variant.name in human
    assert "valid" in human and "partial" in human
    assert "1/2 chapters" in human
    assert "pending" in human


def test_bundle_variants_empty_state_and_bundle_errors_are_json(
    tmp_path: Path, capsys
) -> None:
    import json

    from voicerender import cli

    bundle = tmp_path / "empty.ssmd-bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        '{"format":"ssmdconvert.book","schema_version":1}', encoding="utf-8"
    )
    empty = cli.main(["bundle", "variants", str(bundle)])
    assert empty == 0
    assert "No RenderPlan variants found" in capsys.readouterr().out

    missing = tmp_path / "missing.ssmd-bundle"
    assert cli.main(["bundle", "variants", str(missing), "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "voicerender-bundle-variants"
    assert payload["version"] == 1
    assert payload["error"]
    assert payload["exit_code"] == 2


def test_bundle_variants_accepts_current_directory_bundle_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from voicerender import cli

    bundle = tmp_path / "story.ssmd-bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        '{"format":"ssmdconvert.book","schema_version":1}', encoding="utf-8"
    )
    monkeypatch.chdir(bundle)

    assert cli.main(["bundle", "variants", "."]) == 0
    assert "No RenderPlan variants found" in capsys.readouterr().out

    from voicerender import render_bundle
    from voicerender.book_contract import BookRenderError

    with pytest.raises(BookRenderError) as caught:
        render_bundle(".", variant="missing")
    assert caught.value.code == "variant.not_found"
