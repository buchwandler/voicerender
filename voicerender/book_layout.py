"""Filesystem resolution and safety checks for book-level rendering."""

from __future__ import annotations

import stat
from pathlib import Path, PureWindowsPath

from .book_contract import BookRenderError


def resolve_workspace(workspace: str | Path) -> Path:
    path = Path(workspace).expanduser()
    if not path.name.endswith(".ssmd-bundle"):
        raise BookRenderError(
            "book bundle input must be the canonical .ssmd-bundle directory root"
        )
    if path.is_symlink():
        raise BookRenderError("book bundle root must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BookRenderError(f"cannot resolve book bundle {path}: {exc}") from exc
    if not resolved.is_dir():
        raise BookRenderError(f"book bundle is not a directory: {resolved}")
    return resolved


def resolve_root(path: str | Path, label: str) -> Path:
    root = Path(path).expanduser().resolve(strict=False)
    if root.exists() and not root.is_dir():
        raise BookRenderError(f"{label} root is not a directory: {root}")
    return root


def resolve_variant_directory(renderplan_root: Path, variant: str) -> Path:
    if (
        not isinstance(variant, str)
        or not variant
        or variant in {".", ".."}
        or Path(variant).name != variant
        or PureWindowsPath(variant).name != variant
        or "/" in variant
        or "\\" in variant
        or "\x00" in variant
    ):
        raise BookRenderError(
            "--variant must be an exact directory basename",
            code="variant.identity_mismatch",
        )

    candidate = renderplan_root / variant
    if not candidate.exists():
        raise BookRenderError(
            f"RenderPlan variant directory does not exist: {candidate}",
            code="variant.not_found",
        )
    try:
        directory = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BookRenderError(
            f"cannot resolve RenderPlan variant directory {candidate}: {exc}",
            code="variant.not_found",
        ) from exc
    if not directory.is_dir() or not is_within(directory, renderplan_root):
        raise BookRenderError(
            f"RenderPlan variant is not a directory inside its root: {candidate}",
            code="variant.not_found",
        )
    return directory


def require_contained_file(
    path: Path,
    root: Path,
    label: str,
    *,
    missing_code: str,
    reject_symlink: bool,
) -> Path:
    if reject_symlink and path.is_symlink():
        raise BookRenderError(
            f"{label} must not be a symlink: {path}", code=missing_code
        )
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BookRenderError(
            f"{label} is missing or unreadable: {path}", code=missing_code
        ) from exc
    try:
        root_resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BookRenderError(f"cannot resolve containment root {root}: {exc}") from exc
    if not is_within(resolved, root_resolved):
        raise BookRenderError(
            f"{label} resolves outside its variant directory: {path}", code=missing_code
        )
    if not resolved.is_file():
        raise BookRenderError(
            f"{label} is not a regular file: {path}", code=missing_code
        )
    return resolved


def regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def assert_safe_output_metadata(path: Path, output_dir: Path) -> None:
    if path.is_symlink():
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise BookRenderError(
                f"output metadata symlink cannot be resolved: {path}",
                code="workspace.invalid",
            ) from exc
        if not is_within(target, output_dir.resolve(strict=False)):
            raise BookRenderError(
                f"output metadata symlink escapes chapter directory: {path}",
                code="workspace.invalid",
            )
    if path.exists() and path.is_dir():
        raise BookRenderError(
            f"output metadata path is a directory: {path}", code="workspace.invalid"
        )


def is_within(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except ValueError:
        return False


__all__ = [
    "assert_safe_output_metadata",
    "is_within",
    "regular_file",
    "require_contained_file",
    "resolve_root",
    "resolve_variant_directory",
    "resolve_workspace",
]
