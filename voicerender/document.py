"""Single-document SSMDDoc rendering on the existing RenderPlan execution path."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any, Literal

import tomlkit
from renderplan import RenderPlan, load_plan
from renderplan import codec as renderplan_codec

from .book import _get_preflight
from .document_contract import DocumentWorkspaceError, load_document_workspace
from .errors import ArtifactError, DocumentRenderError, PlanError
from .events import RenderEventCallback
from .renderplan_input import EmphasisPolicy
from .renderplan_render import RENDER_MANIFEST_FILENAME, render_renderplan

DOCUMENT_RENDERPLAN_FILENAME = "document.renderplan.toml"


@dataclass(frozen=True, slots=True)
class DocumentRenderRequest:
    workspace: Path
    document_id: str
    variant_name: str
    renderplan_variant_dir: Path
    profile_path: Path
    renderplan_path: Path
    output_variant_dir: Path
    output_dir: Path
    cache_dir: Path
    speech_numbers: frozenset[int] | None
    speech_selector: str
    force: bool
    reset_stale_output: bool
    emphasis_policy: EmphasisPolicy
    render_profile_id: str
    plan: RenderPlan


@dataclass(frozen=True, slots=True)
class DocumentRenderResult:
    request: DocumentRenderRequest
    report_path: Path
    preflight_report: Path | None
    recheck_report: Path | None
    manifest: Path | None
    status: Literal["complete", "partial", "failed", "reused"]
    planned_speech: int
    selected_speech: int
    available_speech: int
    rendered: int
    reused: int
    manifest_complete: bool
    error_code: str | None = None
    error_message: str | None = None
    replan_required: bool = False
    pending_composition: bool = False
    elapsed_seconds: float = 0.0
    error_segment_number: int | None = None
    error_segment_id: str | None = None
    error_engine: str | None = None

    @property
    def exit_code(self) -> int:
        return 1 if self.status == "failed" else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": str(self.request.workspace),
            "document_id": self.request.document_id,
            "variant": self.request.variant_name,
            "render_profile_id": self.request.render_profile_id,
            "renderplan": str(self.request.renderplan_path),
            "output": str(self.request.output_dir),
            "cache": str(self.request.cache_dir),
            "speech_selector": self.request.speech_selector,
            "status": self.status,
            "planned_speech": self.planned_speech,
            "selected_speech": self.selected_speech,
            "available_speech": self.available_speech,
            "rendered": self.rendered,
            "reused": self.reused,
            "manifest_complete": self.manifest_complete,
            "preflight_report": str(self.preflight_report)
            if self.preflight_report
            else None,
            "recheck_report": str(self.recheck_report) if self.recheck_report else None,
            "manifest": str(self.manifest) if self.manifest else None,
            "report": str(self.report_path),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "replan_required": self.replan_required,
            "pending_composition": self.pending_composition,
            "elapsed_seconds": self.elapsed_seconds,
            "error_segment_number": self.error_segment_number,
            "error_segment_id": self.error_segment_id,
            "error_engine": self.error_engine,
            "exit_code": self.exit_code,
        }


def _resolve_document_request(
    workspace: str | Path,
    *,
    variant: str,
    speech: str = "all",
    renderplan_root: str | Path | None = None,
    output_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    reset_stale_output: bool = False,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
) -> DocumentRenderRequest:
    index = load_document_workspace(workspace)
    if (
        not isinstance(variant, str)
        or not variant
        or variant in {".", ".."}
        or "/" in variant
        or "\\" in variant
        or "\x00" in variant
        or Path(variant).name != variant
    ):
        raise DocumentWorkspaceError("variant must be an exact safe directory basename")
    rp_root = (
        Path(renderplan_root).expanduser()
        if renderplan_root is not None
        else index.root / "renderplan"
    )
    variant_dir = rp_root / variant
    rp_root = rp_root.resolve(strict=False)
    if not variant_dir.is_dir():
        raise DocumentRenderError(
            f"RenderPlan variant not found: {variant_dir}", code="variant.not_found"
        )
    variant_dir = variant_dir.resolve(strict=True)
    if not _within(variant_dir, rp_root):
        raise DocumentWorkspaceError("RenderPlan variant escapes the selected root")
    if not variant_dir.is_dir():
        raise DocumentRenderError(
            f"RenderPlan variant not found: {variant_dir}", code="variant.not_found"
        )
    snapshot = variant_dir / "renderprofile.toml"
    snapshot = _contained_regular(snapshot, variant_dir, "profile snapshot")
    if not snapshot.is_file():
        raise DocumentRenderError(
            f"profile snapshot missing: {snapshot}", code="profile.snapshot_missing"
        )
    profile = renderplan_codec.load_profile(snapshot)
    profile_id = profile.profile_id or profile.computed_id()
    if not variant.endswith("--" + profile_id.removeprefix("sha256:")[:12]):
        raise DocumentRenderError(
            "variant fingerprint does not match profile snapshot",
            code="variant.identity_mismatch",
        )
    plan_path = variant_dir / DOCUMENT_RENDERPLAN_FILENAME
    plan_path = _contained_regular(plan_path, variant_dir, "RenderPlan")
    plan = load_plan(plan_path)
    if not isinstance(plan, RenderPlan) or plan.render_profile_id != profile_id:
        raise DocumentRenderError(
            "RenderPlan profile identity does not match profile snapshot",
            code="plan.identity_mismatch",
        )
    if speech.strip().lower() == "all":
        selected = None
    else:
        numbers: set[int] = set()
        try:
            for part in speech.split(","):
                match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
                if not match:
                    raise ValueError
                start, end = int(match.group(1)), int(match.group(2) or match.group(1))
                if start < 1 or end < start or end > len(plan.speech):
                    raise ValueError
                numbers.update(range(start, end + 1))
        except ValueError as exc:
            raise DocumentRenderError(
                f"invalid speech selector: {speech}", code="selection.invalid"
            ) from exc
        selected = frozenset(numbers)
    out_root = (
        Path(output_root).expanduser()
        if output_root is not None
        else index.root / "voicerender"
    )
    out_root = out_root.resolve(strict=False)
    out_variant = (out_root / variant).resolve(strict=False)
    if not _within(out_variant, out_root):
        raise DocumentWorkspaceError("output variant escapes the selected root")
    output = out_variant / index.document_id
    cache = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else out_variant / ".voicerender-cache"
    )
    output = output.resolve(strict=False)
    cache = cache.resolve(strict=False)
    if not _within(output, out_variant) or not _within(cache, out_variant):
        raise DocumentWorkspaceError(
            "output and cache paths must be contained by the output variant"
        )
    if output.is_relative_to(cache) or cache.is_relative_to(output):
        raise DocumentWorkspaceError(
            "cache and document output paths must be siblings, not nested"
        )
    if output.resolve(strict=False).is_relative_to(variant_dir.resolve(strict=False)):
        raise DocumentWorkspaceError(
            "VoiceRender output must not be inside the RenderPlan variant"
        )
    return DocumentRenderRequest(
        index.root,
        index.document_id,
        variant,
        variant_dir,
        snapshot,
        plan_path,
        out_variant,
        output,
        cache,
        selected,
        speech,
        force,
        reset_stale_output,
        EmphasisPolicy(emphasis_policy),
        profile_id,
        plan,
    )


def render_document(
    workspace: str | Path,
    *,
    variant: str,
    speech: str = "all",
    renderplan_root: str | Path | None = None,
    output_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    reset_stale_output: bool = False,
    emphasis_policy: EmphasisPolicy | str = EmphasisPolicy.REJECT,
    on_event: RenderEventCallback | None = None,
) -> DocumentRenderResult:
    start = monotonic()
    request = _resolve_document_request(
        workspace,
        variant=variant,
        speech=speech,
        renderplan_root=renderplan_root,
        output_root=output_root,
        cache_dir=cache_dir,
        force=force,
        reset_stale_output=reset_stale_output,
        emphasis_policy=emphasis_policy,
    )
    report_path = request.output_variant_dir / "voicerender-report.toml"
    preflight_path = request.output_dir / "preflight.toml"
    recheck_path = request.output_dir / "preflight-recheck.toml"
    manifest_path = request.output_dir / RENDER_MANIFEST_FILENAME
    try:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        preflight = _get_preflight(
            request.plan,
            preflight_path,
            emphasis_policy=request.emphasis_policy,
            on_event=on_event,
        )
        if preflight.status != "ok":
            code = (
                "preflight.replan"
                if preflight.status == "replan"
                else "preflight.reject"
            )
            result = DocumentRenderResult(
                request,
                report_path,
                preflight_path,
                None,
                None,
                "failed",
                len(request.plan.speech),
                len(request.plan.speech)
                if request.speech_numbers is None
                else len(request.speech_numbers),
                0,
                0,
                0,
                False,
                code,
                f"RenderPlan preflight status is {preflight.status!r}",
                preflight.status == "replan",
            )
        else:
            rendered = render_renderplan(
                request.plan,
                preflight=preflight,
                output_dir=request.output_dir,
                cache_dir=request.cache_dir,
                force=request.force,
                reset_stale_output=request.reset_stale_output,
                recheck_report_path=recheck_path,
                speech_numbers=request.speech_numbers,
                emphasis_policy=request.emphasis_policy,
                on_event=on_event,
            )
            result = DocumentRenderResult(
                request,
                report_path,
                preflight_path,
                recheck_path if rendered.fresh_preflight else None,
                rendered.manifest_path,
                "failed"
                if rendered.failed
                else (
                    "reused"
                    if rendered.rendered_count == 0
                    and rendered.reused_count > 0
                    and rendered.manifest.complete
                    else ("complete" if rendered.manifest.complete else "partial")
                ),
                rendered.planned_count,
                rendered.selected_count,
                rendered.available_count,
                rendered.rendered_count,
                rendered.reused_count,
                rendered.manifest.complete,
                rendered.error_code,
                rendered.error,
                False,
                rendered.pending_composition,
                error_segment_number=getattr(rendered, "error_segment_number", None),
                error_segment_id=getattr(rendered, "error_segment_id", None),
                error_engine=getattr(rendered, "error_engine", None),
            )
    except (ArtifactError, PlanError, OSError, ValueError, TypeError) as exc:
        code = getattr(exc, "code", None) or "render.failed"
        result = DocumentRenderResult(
            request,
            report_path,
            preflight_path if preflight_path.exists() else None,
            None,
            manifest_path if manifest_path.exists() else None,
            "failed",
            len(request.plan.speech),
            len(request.plan.speech)
            if request.speech_numbers is None
            else len(request.speech_numbers),
            0,
            0,
            0,
            False,
            code,
            str(exc),
        )
    result = replace(result, elapsed_seconds=monotonic() - start)
    _write_report(result)
    return result


def _write_report(result: DocumentRenderResult) -> None:
    req = result.request
    data = {
        "format": "voicerender-document-report",
        "version": 1,
        "workspace": str(req.workspace),
        "document_id": req.document_id,
        "variant": req.variant_name,
        "render_profile_id": req.render_profile_id,
        "renderplan": str(req.renderplan_path),
        "render_plan_id": req.plan.plan_id,
        "source_plan_id": req.plan.source_plan_id,
        "output": str(req.output_dir),
        "cache": str(req.cache_dir),
        "speech_selector": req.speech_selector,
        "planned_speech": result.planned_speech,
        "selected_speech": result.selected_speech,
        "available_speech": result.available_speech,
        "rendered": result.rendered,
        "reused": result.reused,
        "status": result.status,
        "manifest_complete": result.manifest_complete,
        "pending_composition": result.pending_composition,
        "replan_required": result.replan_required,
        "elapsed_seconds": result.elapsed_seconds,
    }
    for key, val in (
        ("preflight_report", result.preflight_report),
        ("recheck_report", result.recheck_report),
        ("manifest", result.manifest),
        ("error_code", result.error_code),
        ("error_message", result.error_message),
        ("error_segment_number", result.error_segment_number),
        ("error_segment_id", result.error_segment_id),
        ("error_engine", result.error_engine),
    ):
        if val is not None:
            data[key] = str(val) if isinstance(val, Path) else val
    _atomic_write_text(result.report_path, tomlkit.dumps(data))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _contained_regular(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DocumentRenderError(
            f"{label} missing or unreadable: {path}",
            code="profile.snapshot_missing"
            if label == "profile snapshot"
            else "plan.missing",
        ) from exc
    if not _within(resolved, root.resolve(strict=True)) or not resolved.is_file():
        raise DocumentWorkspaceError(
            f"{label} must be a contained regular file: {path}"
        )
    return resolved


__all__ = [
    "DocumentRenderRequest",
    "DocumentRenderResult",
    "_resolve_document_request",
    "render_document",
]
