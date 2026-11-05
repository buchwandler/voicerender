"""Read-only, selection-aware inspection of persisted bundle render outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from renderplan import RenderPlan, SpeechArtifact, SpeechItem, load_manifest

from .audio_boundaries import BOUNDARY_POLICY_ID
from .book import (
    RECHECK_PREFLIGHT_FILENAME,
    BookRenderError,
    _resolve_request,
)
from .bundle_contract import BundleKind, load_bundle_identity
from .document import _resolve_document_request
from .renderplan_render import (
    RENDER_DIAGNOSTICS_FILENAME,
    RENDER_MANIFEST_FILENAME,
)
from .storage import verified_wav

RenderOccurrenceState = Literal["reusable", "missing", "stale", "corrupt"]
BundleRenderState = Literal["complete", "partial", "not_started", "stale", "corrupt"]


@dataclass(frozen=True, slots=True)
class RenderOccurrenceInspection:
    """Persisted state for one selected chapter/speech occurrence."""

    chapter_id: str
    number: int
    segment_id: str
    wav_path: Path
    state: RenderOccurrenceState
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "number": self.number,
            "segment_id": self.segment_id,
            "wav_path": str(self.wav_path),
            "state": self.state,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class BundleRenderInspection:
    """Selection-specific render readiness, without changing workspace files."""

    kind: BundleKind
    bundle: Path
    variant: str
    render_profile_id: str
    selected_chapters: tuple[str, ...]
    occurrences: tuple[RenderOccurrenceInspection, ...]
    force_required: bool
    reset_safe: bool
    reset_detail: str | None = None

    @property
    def replacement_safe(self) -> bool:
        """Whether VoiceRender certifies force/reset replacement of selected outputs."""
        return self.reset_safe

    @property
    def selected_occurrences(self) -> tuple[RenderOccurrenceInspection, ...]:
        return self.occurrences

    @property
    def rendered_occurrences(self) -> tuple[RenderOccurrenceInspection, ...]:
        return tuple(item for item in self.occurrences if item.state == "reusable")

    @property
    def reusable_occurrences(self) -> tuple[RenderOccurrenceInspection, ...]:
        return self.rendered_occurrences

    @property
    def missing_occurrences(self) -> tuple[RenderOccurrenceInspection, ...]:
        return tuple(item for item in self.occurrences if item.state == "missing")

    @property
    def stale_occurrences(self) -> tuple[RenderOccurrenceInspection, ...]:
        return tuple(item for item in self.occurrences if item.state == "stale")

    @property
    def corrupt_occurrences(self) -> tuple[RenderOccurrenceInspection, ...]:
        return tuple(item for item in self.occurrences if item.state == "corrupt")

    @property
    def state(self) -> BundleRenderState:
        states = {item.state for item in self.occurrences}
        if "stale" in states:
            return "stale"
        if "corrupt" in states:
            return "corrupt"
        if not states or states == {"reusable"}:
            return "complete"
        if states == {"missing"}:
            return "not_started"
        return "partial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "voicerender-bundle-render-inspection",
            "version": 1,
            "kind": self.kind,
            "bundle": str(self.bundle),
            "variant": self.variant,
            "render_profile_id": self.render_profile_id,
            "selected_chapters": list(self.selected_chapters),
            "state": self.state,
            "selected_count": len(self.occurrences),
            "rendered_count": len(self.rendered_occurrences),
            "missing_count": len(self.missing_occurrences),
            "stale_count": len(self.stale_occurrences),
            "corrupt_count": len(self.corrupt_occurrences),
            "force_required": self.force_required,
            "reset_safe": self.reset_safe,
            "reset_detail": self.reset_detail,
            "replacement_safe": self.replacement_safe,
            "occurrences": [item.to_dict() for item in self.occurrences],
        }


def inspect_bundle_render(
    bundle: str | Path,
    *,
    variant: str,
    chapters: str = "all",
    speech: str = "all",
    renderplan_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> BundleRenderInspection:
    """Inspect selected bundle output artifacts without preflight or filesystem writes."""
    identity = load_bundle_identity(bundle)
    if identity.kind == "book":
        request = _resolve_request(
            identity.root,
            variant=variant,
            chapter_selector=chapters,
            speech_selector=speech,
            renderplan_root=renderplan_root,
            output_root=output_root,
            cache_dir=None,
            force=False,
            reset_stale_output=False,
            fail_fast=False,
            emphasis_policy="reject",
            inspect_output=True,
        )
        chapter_plans = tuple(
            (
                chapter.entry.id,
                chapter.plan,
                chapter.output_dir,
                chapter.speech_numbers,
            )
            for chapter in request._loaded_chapters
        )
        profile_id = request.render_profile_id
    else:
        if chapters.strip().casefold() != "all":
            raise BookRenderError(
                "--chapters must be 'all' for a document bundle",
                code="selection.invalid",
            )
        request = _resolve_document_request(
            identity.root,
            variant=variant,
            speech=speech,
            renderplan_root=renderplan_root,
            output_root=output_root,
            cache_dir=None,
            force=False,
            reset_stale_output=False,
            emphasis_policy="reject",
        )
        chapter_plans = (
            (
                request.document_id,
                request.plan,
                request.output_dir,
                request.speech_numbers,
            ),
        )
        profile_id = request.render_profile_id

    occurrences: list[RenderOccurrenceInspection] = []
    force_required = False
    reset_detail: str | None = None
    for chapter_id, plan, output_dir, selected_numbers in chapter_plans:
        chapter_occurrences, chapter_force, chapter_reset_detail = _inspect_plan_output(
            chapter_id,
            plan,
            output_dir,
            selected_numbers,
        )
        occurrences.extend(chapter_occurrences)
        force_required = force_required or chapter_force
        if chapter_reset_detail is not None:
            reset_detail = chapter_reset_detail
    selected_chapters = tuple(chapter_id for chapter_id, *_ in chapter_plans)
    return BundleRenderInspection(
        kind=identity.kind,
        bundle=identity.root,
        variant=variant,
        render_profile_id=profile_id,
        selected_chapters=selected_chapters,
        occurrences=tuple(occurrences),
        force_required=force_required,
        reset_safe=all(
            _reset_is_safe(output_dir) for _, _, output_dir, _ in chapter_plans
        ),
        reset_detail=reset_detail,
    )


def _inspect_plan_output(
    chapter_id: str,
    plan: RenderPlan,
    output_dir: Path,
    selected_numbers: frozenset[int] | None,
) -> tuple[tuple[RenderOccurrenceInspection, ...], bool, str | None]:
    selected_items = tuple(
        item
        for item in plan.speech
        if selected_numbers is None or item.number in selected_numbers
    )
    expected = {
        (item.number, item.segment_id or item.computed_id()): item
        for item in plan.speech
    }
    if not selected_items:
        return (), False, None

    manifest_path = output_dir / RENDER_MANIFEST_FILENAME
    if manifest_path.is_symlink():
        return (
            _uniform_occurrences(
                chapter_id,
                selected_items,
                output_dir,
                "corrupt",
                "manifest is a symlink",
            ),
            True,
            _reset_safety_detail(output_dir),
        )
    if not manifest_path.exists():
        return (
            _uniform_occurrences(
                chapter_id,
                selected_items,
                output_dir,
                "missing",
                "render manifest is absent",
            ),
            False,
            _reset_safety_detail(output_dir),
        )
    if not manifest_path.is_file():
        return (
            _uniform_occurrences(
                chapter_id,
                selected_items,
                output_dir,
                "corrupt",
                "manifest is not a regular file",
            ),
            True,
            _reset_safety_detail(output_dir),
        )

    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError, TypeError) as exc:
        return (
            _uniform_occurrences(
                chapter_id,
                selected_items,
                output_dir,
                "corrupt",
                f"invalid manifest: {exc}",
            ),
            True,
            _reset_safety_detail(output_dir),
        )

    plan_id = plan.plan_id or plan.computed_id()
    if manifest.render_plan_id != plan_id:
        return (
            _uniform_occurrences(
                chapter_id,
                selected_items,
                output_dir,
                "stale",
                "manifest belongs to a different RenderPlan",
            ),
            True,
            _reset_safety_detail(output_dir),
        )
    try:
        manifest.validate_against(plan)
    except (ValueError, TypeError) as exc:
        return (
            _uniform_occurrences(
                chapter_id,
                selected_items,
                output_dir,
                "corrupt",
                f"invalid manifest: {exc}",
            ),
            True,
            _reset_safety_detail(output_dir),
        )

    if not _current_audio_policy(output_dir):
        return (
            _uniform_occurrences(
                chapter_id,
                selected_items,
                output_dir,
                "stale",
                "render output uses a different or unknown audio-boundary policy",
            ),
            True,
            _reset_safety_detail(output_dir),
        )

    artifacts: dict[tuple[int, str], SpeechArtifact] = {}
    for artifact in manifest.speech:
        key = (artifact.segment_number, artifact.segment_id)
        if key in artifacts or key not in expected:
            return (
                _uniform_occurrences(
                    chapter_id,
                    selected_items,
                    output_dir,
                    "corrupt",
                    "manifest contains duplicate or unknown speech occurrences",
                ),
                True,
                _reset_safety_detail(output_dir),
            )
        if artifact.wav != expected[key].wav:
            return (
                _uniform_occurrences(
                    chapter_id,
                    selected_items,
                    output_dir,
                    "corrupt",
                    "manifest WAV name does not match the RenderPlan",
                ),
                True,
                _reset_safety_detail(output_dir),
            )
        artifacts[key] = artifact

    found: list[RenderOccurrenceInspection] = []
    for item in selected_items:
        segment_id = item.segment_id or item.computed_id()
        key = (item.number, segment_id)
        artifact = artifacts.get(key)
        wav_path = output_dir / item.wav
        if artifact is None:
            state: RenderOccurrenceState = "missing"
            detail = "no verified manifest artifact is recorded"
        else:
            metadata = {
                "sha256": artifact.sha256.removeprefix("sha256:"),
                "frames": artifact.frames,
                "sample_rate": artifact.sample_rate,
                "channels": artifact.channels,
            }
            if verified_wav(wav_path, metadata):
                state = "reusable"
                detail = None
            elif wav_path.exists() or wav_path.is_symlink():
                state = "corrupt"
                detail = "WAV artifact fails manifest integrity verification"
            else:
                state = "missing"
                detail = "manifest artifact WAV is absent"
        found.append(
            RenderOccurrenceInspection(
                chapter_id,
                item.number,
                segment_id,
                wav_path,
                state,
                detail,
            )
        )
    return tuple(found), False, _reset_safety_detail(output_dir)


def _uniform_occurrences(
    chapter_id: str,
    items: tuple[SpeechItem, ...],
    output_dir: Path,
    state: RenderOccurrenceState,
    detail: str,
) -> tuple[RenderOccurrenceInspection, ...]:
    return tuple(
        RenderOccurrenceInspection(
            chapter_id,
            item.number,
            item.segment_id or item.computed_id(),
            output_dir / item.wav,
            state,
            detail,
        )
        for item in items
    )


def _current_audio_policy(output_dir: Path) -> bool:
    path = output_dir / RENDER_DIAGNOSTICS_FILENAME
    if path.is_symlink() or not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and data.get("audio_boundary_policy") == BOUNDARY_POLICY_ID
    )


def _reset_is_safe(output_dir: Path) -> bool:
    if not output_dir.exists():
        return True
    if not output_dir.is_dir() or output_dir.is_symlink():
        return False
    owned_names = {
        RENDER_MANIFEST_FILENAME,
        RENDER_DIAGNOSTICS_FILENAME,
        "preflight.toml",
        RECHECK_PREFLIGHT_FILENAME,
    }
    try:
        for child in output_dir.iterdir():
            is_owned = child.name in owned_names or re.fullmatch(
                r"\d{6,}\.wav", child.name
            )
            if is_owned and not child.is_symlink() and child.is_dir():
                return False
    except OSError:
        return False
    return True


def _reset_safety_detail(output_dir: Path) -> str | None:
    if _reset_is_safe(output_dir):
        return None
    return "reset/force would encounter an unsafe owned output path"


__all__ = [
    "BundleRenderInspection",
    "RenderOccurrenceInspection",
    "inspect_bundle_render",
]
