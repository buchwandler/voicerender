from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_voice_render_imports_without_audiosig() -> None:
    package = Path(__file__).parents[1] / "voicerender"
    assert not (package / "audio_boundaries.py").exists()
    assert all(
        "trim_trail" not in source.read_text(encoding="utf-8")
        for source in package.rglob("*.py")
    )

    script = """
import builtins

original_import = builtins.__import__
def reject_audiosig(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "audiosig" or name.startswith("audiosig."):
        raise ModuleNotFoundError("audiosig is unavailable")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_audiosig
import voicerender
import voicerender.bundle_render_inspect
import voicerender.render
import voicerender.renderplan_render
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=package.parent,
        check=True,
        capture_output=True,
        text=True,
    )
