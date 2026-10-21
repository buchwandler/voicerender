"""Safe identity reader for canonical SSMD bundle roots."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .errors import VoiceRenderError

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
BOOK_FORMAT = "ssmdconvert.book"
DOCUMENT_FORMAT = "ssmdconvert.document"

BundleKind = Literal["book", "document"]


class BundleContractError(VoiceRenderError, ValueError):
    """An input is not a safe, supported SSMD bundle root."""

    def __init__(self, message: str, *, code: str = "bundle.invalid") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True, slots=True)
class BundleIdentity:
    root: Path
    manifest_path: Path
    kind: BundleKind


def load_bundle_identity(path: str | Path) -> BundleIdentity:
    """Read only the bounded, authoritative identity fields of a bundle manifest."""
    input_path = Path(path).expanduser()
    if not input_path.name.endswith(".ssmd-bundle"):
        raise BundleContractError(
            "bundle input must be the canonical .ssmd-bundle directory root"
        )
    if input_path.is_symlink():
        raise BundleContractError("SSMD bundle root must not be a symlink")

    try:
        root = input_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BundleContractError(
            f"cannot resolve SSMD bundle root {input_path}: {exc}"
        ) from exc
    if not root.is_dir():
        raise BundleContractError(f"SSMD bundle root is not a directory: {root}")

    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise BundleContractError("SSMD bundle manifest.json must not be a symlink")
    try:
        manifest_stat = manifest_path.lstat()
    except OSError as exc:
        raise BundleContractError(
            f"SSMD bundle manifest.json is missing or unreadable: {manifest_path}"
        ) from exc
    if not stat.S_ISREG(manifest_stat.st_mode):
        raise BundleContractError(
            f"SSMD bundle manifest.json must be a regular file: {manifest_path}"
        )
    if manifest_stat.st_size > MAX_MANIFEST_BYTES:
        raise BundleContractError(
            f"SSMD bundle manifest.json exceeds the {MAX_MANIFEST_BYTES}-byte size limit"
        )

    try:
        raw = manifest_path.read_bytes()
        if len(raw) > MAX_MANIFEST_BYTES:
            raise BundleContractError(
                f"SSMD bundle manifest.json exceeds the {MAX_MANIFEST_BYTES}-byte size limit"
            )
        data: Any = json.loads(raw.decode("utf-8", errors="strict"))
    except BundleContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleContractError(
            f"cannot read SSMD bundle manifest.json: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise BundleContractError("SSMD bundle manifest root must be a JSON object")

    bundle_format = data.get("format")
    if bundle_format not in (BOOK_FORMAT, DOCUMENT_FORMAT):
        raise BundleContractError(
            f"unsupported SSMD bundle format {bundle_format!r}; expected "
            f"{BOOK_FORMAT!r} or {DOCUMENT_FORMAT!r}",
            code="bundle.unsupported",
        )
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise BundleContractError(
            f"unsupported SSMD bundle schema_version {schema_version!r}; expected 1",
            code="bundle.unsupported",
        )

    kind: BundleKind = "book" if bundle_format == BOOK_FORMAT else "document"
    return BundleIdentity(root=root, manifest_path=manifest_path, kind=kind)


__all__ = [
    "BOOK_FORMAT",
    "DOCUMENT_FORMAT",
    "MAX_MANIFEST_BYTES",
    "BundleContractError",
    "BundleIdentity",
    "BundleKind",
    "load_bundle_identity",
]
