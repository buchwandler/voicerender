"""Minimal, dependency-free document-bundle identity validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import DocumentRenderError

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
DOCUMENT_FORMAT = "ssmdconvert.document"


class DocumentWorkspaceError(DocumentRenderError):
    """The input is not a safe, valid document bundle."""


@dataclass(frozen=True, slots=True)
class DocumentWorkspaceIndex:
    root: Path
    manifest_path: Path
    document_id: str
    document_path: Path


def load_document_workspace(value: str | Path) -> DocumentWorkspaceIndex:
    """Validate document-specific identity fields within a canonical bundle root."""
    input_path = Path(value).expanduser()
    if not input_path.name.endswith(".ssmd-bundle"):
        raise DocumentWorkspaceError(
            "document bundle input must be the canonical .ssmd-bundle directory root"
        )
    if input_path.is_symlink():
        raise DocumentWorkspaceError("document bundle root must not be a symlink")
    try:
        root = input_path.resolve(strict=True)
        if not root.is_dir():
            raise DocumentWorkspaceError(
                "document bundle input is not a directory root"
            )
        manifest = root / "manifest.json"
        if manifest.is_symlink():
            raise DocumentWorkspaceError("manifest.json must not be a symlink")
        manifest = manifest.resolve(strict=True)
        if not _within(manifest, root) or not manifest.is_file():
            raise DocumentWorkspaceError(
                "manifest.json is not a contained regular file"
            )
        if manifest.stat().st_size > MAX_MANIFEST_BYTES:
            raise DocumentWorkspaceError("manifest.json exceeds the 2 MiB size limit")
        raw = manifest.read_bytes().decode("utf-8", errors="strict")
        data: Any = json.loads(raw)
    except DocumentWorkspaceError:
        raise
    except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
        raise DocumentWorkspaceError(f"cannot read document bundle: {exc}") from exc
    if not isinstance(data, dict):
        raise DocumentWorkspaceError("manifest root must be a JSON object")
    if (
        data.get("format") != DOCUMENT_FORMAT
        or type(data.get("schema_version")) is not int
        or data.get("schema_version") != 1
        or data.get("ssmd_version") != "0.9"
    ):
        raise DocumentWorkspaceError(
            "unsupported document bundle manifest format or version"
        )
    document = data.get("document")
    if not isinstance(document, dict) or document.get("id") != "document-0001":
        raise DocumentWorkspaceError(
            "document bundle manifest must have canonical id document-0001"
        )
    relative = document.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or "\x00" in relative
        or "\\" in relative
        or re.match(r"^[A-Za-z]:", relative)
    ):
        raise DocumentWorkspaceError("document bundle manifest path is unsafe")
    posix = PurePosixPath(relative)
    if (
        posix.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.split("/"))
        or posix.as_posix() != relative
    ):
        raise DocumentWorkspaceError(
            "document bundle manifest path must be normalized and relative"
        )
    if relative != "document.ssmd.md":
        raise DocumentWorkspaceError(
            "schema v1 document bundle path must be document.ssmd.md"
        )
    document_path = root / relative
    if document_path.is_symlink():
        raise DocumentWorkspaceError(
            "canonical document must not escape through a symlink"
        )
    try:
        resolved_document = document_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DocumentWorkspaceError(
            f"canonical document is missing or unsafe: {exc}"
        ) from exc
    if not _within(resolved_document, root) or not resolved_document.is_file():
        raise DocumentWorkspaceError(
            "canonical document is not a contained regular file"
        )
    return DocumentWorkspaceIndex(root, manifest, "document-0001", resolved_document)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = [
    "DocumentWorkspaceError",
    "DocumentWorkspaceIndex",
    "load_document_workspace",
]
