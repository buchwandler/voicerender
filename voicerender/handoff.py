"""Public, narrow reader and verifier for VoiceRender render handoffs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .inputs.utterplan_v5 import UtterPlanV5, load_utterplan_v5
from .manifest import (
    read_manifest,
    validate_manifest_document,
    verify_manifest_audio,
)
from .manifest import (
    validate_manifest_source as _validate_manifest_source,
)


def read_render_manifest(
    path: str | Path, *, verify_audio: bool = True
) -> dict[str, Any]:
    """Read and validate a manifest, optionally verifying every speech WAV."""
    return read_manifest(path, verify_audio=verify_audio)


def verify_render_manifest(path: str | Path) -> dict[str, Any]:
    """Verify manifest identity, structure, bundle containment, and speech WAVs."""
    return read_manifest(path, verify_audio=True)


def load_source_plan(path: str | Path) -> UtterPlanV5:
    """Load a strict UtterPlan v5 source for a downstream handoff join."""
    return load_utterplan_v5(path)


def validate_manifest_source(
    plan: UtterPlanV5 | str | Path,
    manifest: dict[str, Any] | str | Path,
) -> None:
    """Verify the manifest is bound to the exact source plan and selected operations."""
    source_plan = load_source_plan(plan) if isinstance(plan, (str, Path)) else plan
    if not isinstance(source_plan, UtterPlanV5):
        raise TypeError("plan must be an UtterPlanV5 value or path")
    if isinstance(manifest, (str, Path)):
        document = read_render_manifest(manifest)
    else:
        document = validate_manifest_document(manifest)
    _validate_manifest_source(document, source_plan)


def verify_render_manifest_audio(path: str | Path, document: dict[str, Any]) -> None:
    """Verify the speech artifacts in an already parsed manifest document."""
    verify_manifest_audio(path, document)


__all__ = [
    "load_source_plan",
    "read_render_manifest",
    "validate_manifest_source",
    "verify_render_manifest",
    "verify_render_manifest_audio",
]
