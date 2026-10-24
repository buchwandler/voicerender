"""Unified public rendering facade for SSMD bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .book import BookRenderResult, render_book
from .bundle_contract import BundleKind, load_bundle_identity
from .document import DocumentRenderResult, render_document
from .events import RenderEventCallback
from .renderplan_input import EmphasisPolicy
from .session_pool import validate_max_open_sessions


@dataclass(frozen=True, slots=True)
class BundleRenderResult:
    """Kind-tagged result returned by :func:`render_bundle`."""

    kind: BundleKind
    value: BookRenderResult | DocumentRenderResult

    @property
    def exit_code(self) -> int:
        return self.value.exit_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "voicerender-bundle-result",
            "version": 1,
            "kind": self.kind,
            "result": self.value.to_dict(),
        }


def render_bundle(
    bundle: str | Path,
    *,
    variant: str,
    chapters: str = "all",
    speech: str = "all",
    renderplan_root: str | Path | None = None,
    output_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    reset_stale: bool = False,
    emphasis_policy: Literal["reject", "defer-to-compose"] = "reject",
    fail_fast: bool = False,
    max_open_sessions: int = 1,
    on_progress: RenderEventCallback | None = None,
) -> BundleRenderResult:
    """Render one canonical bundle root, dispatching by its manifest identity."""
    max_open_sessions = validate_max_open_sessions(max_open_sessions)
    identity = load_bundle_identity(bundle)
    if identity.kind == "book":
        book_result = render_book(
            identity.root,
            variant=variant,
            chapters=chapters,
            speech=speech,
            renderplan_root=renderplan_root,
            output_root=output_root,
            cache_dir=cache_dir,
            force=force,
            reset_stale_output=reset_stale,
            fail_fast=fail_fast,
            emphasis_policy=EmphasisPolicy(emphasis_policy),
            on_event=on_progress,
            max_open_sessions=max_open_sessions,
        )
        return BundleRenderResult(kind=identity.kind, value=book_result)

    if chapters.strip().casefold() != "all":
        raise ValueError("--chapters must be 'all' for a document bundle")
    document_result = render_document(
        identity.root,
        variant=variant,
        speech=speech,
        renderplan_root=renderplan_root,
        output_root=output_root,
        cache_dir=cache_dir,
        force=force,
        reset_stale_output=reset_stale,
        emphasis_policy=EmphasisPolicy(emphasis_policy),
        on_event=on_progress,
        max_open_sessions=max_open_sessions,
    )
    return BundleRenderResult(kind=identity.kind, value=document_result)


__all__ = ["BundleRenderResult", "render_bundle"]
