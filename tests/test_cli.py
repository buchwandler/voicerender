from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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

    assert "preflight" in root_help
    assert "renderplan" in root_help
    assert "utterplan" not in root_help
    assert "--preflight" in render_help
    assert "--out-dir" in render_help
    assert "--voice" not in render_help
    assert "--take" not in render_help
    assert "--profile" not in render_help


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
assert callable(voicerender.preflight_render_plan)
assert callable(voicerender.render_renderplan)
assert not any(name.startswith('voicerender.inputs.utterplan') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


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
