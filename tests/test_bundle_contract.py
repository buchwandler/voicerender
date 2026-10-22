"""Tests for safe, manifest-driven SSMD bundle identity resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicerender.bundle_contract import (
    MAX_MANIFEST_BYTES,
    BundleContractError,
    BundleIdentity,
    load_bundle_identity,
)


def _bundle(
    root: Path,
    *,
    bundle_format: str = "ssmdconvert.book",
    schema_version: object = 1,
) -> Path:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"format": bundle_format, "schema_version": schema_version}),
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize(
    ("bundle_format", "kind"),
    [
        ("ssmdconvert.book", "book"),
        ("ssmdconvert.document", "document"),
    ],
)
def test_loads_supported_bundle_identity(
    tmp_path: Path, bundle_format: str, kind: str
) -> None:
    root = _bundle(tmp_path / "sample.ssmd-bundle", bundle_format=bundle_format)

    identity = load_bundle_identity(root)

    assert identity == BundleIdentity(
        root=root.resolve(),
        manifest_path=(root / "manifest.json").resolve(),
        kind=kind,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("working_directory", "path"),
    [("root", "."), ("root", "./"), ("child", "..")],
)
def test_bundle_identity_accepts_aliases_resolving_to_canonical_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    working_directory: str,
    path: str,
) -> None:
    root = _bundle(tmp_path / "story.ssmd-bundle")
    child = root / "child"
    child.mkdir()
    monkeypatch.chdir(root if working_directory == "root" else child)

    identity = load_bundle_identity(path)

    assert identity.root == root.resolve()


@pytest.mark.parametrize("name", ["workspace", "book.ssmd-bundle/chapters"])
def test_rejects_noncanonical_root_shape(tmp_path: Path, name: str) -> None:
    with pytest.raises(BundleContractError, match="canonical .ssmd-bundle"):
        load_bundle_identity(tmp_path / name)


def test_rejects_existing_noncanonical_directories_even_with_manifest(
    tmp_path: Path,
) -> None:
    workspace = _bundle(tmp_path / "workspace")
    bundle = _bundle(tmp_path / "story.ssmd-bundle")
    renderplan_root = bundle / "renderplan"
    renderplan_root.mkdir()

    with pytest.raises(BundleContractError, match="canonical .ssmd-bundle"):
        load_bundle_identity(workspace)
    with pytest.raises(BundleContractError, match="canonical .ssmd-bundle"):
        load_bundle_identity(renderplan_root)


def test_rejects_document_file_and_zip_shortcuts(tmp_path: Path) -> None:
    root = tmp_path / "sample.ssmd-bundle"
    root.mkdir()
    document = root / "document.ssmd.md"
    document.write_text("Hello.", encoding="utf-8")

    with pytest.raises(BundleContractError, match="canonical .ssmd-bundle"):
        load_bundle_identity(document)
    with pytest.raises(BundleContractError, match="canonical .ssmd-bundle"):
        load_bundle_identity(tmp_path / "sample.ssmd-bundle.zip")


def test_rejects_symlink_root(tmp_path: Path) -> None:
    target = _bundle(tmp_path / "target.ssmd-bundle")
    link = tmp_path / "link.ssmd-bundle"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(BundleContractError, match="root must not be a symlink"):
        load_bundle_identity(link)


def test_rejects_non_directory_canonical_path(tmp_path: Path) -> None:
    path = tmp_path / "sample.ssmd-bundle"
    path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(BundleContractError, match="not a directory"):
        load_bundle_identity(path)


@pytest.mark.parametrize(
    ("bundle_format", "schema_version", "message"),
    [
        ("unknown.format", 1, "unsupported SSMD bundle format"),
        ("ssmdconvert.book", 2, "schema_version 2"),
        ("ssmdconvert.document", True, "schema_version True"),
    ],
)
def test_rejects_unsupported_manifest_identity(
    tmp_path: Path, bundle_format: str, schema_version: object, message: str
) -> None:
    root = _bundle(
        tmp_path / "sample.ssmd-bundle",
        bundle_format=bundle_format,
        schema_version=schema_version,
    )

    with pytest.raises(BundleContractError, match=message) as caught:
        load_bundle_identity(root)
    assert caught.value.code == "bundle.unsupported"


def test_rejects_non_object_and_malformed_manifest(tmp_path: Path) -> None:
    root = tmp_path / "sample.ssmd-bundle"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(BundleContractError, match="must be a JSON object"):
        load_bundle_identity(root)

    manifest.write_text("{", encoding="utf-8")
    with pytest.raises(BundleContractError, match="cannot read SSMD bundle manifest"):
        load_bundle_identity(root)


def test_rejects_oversized_and_symlink_manifest(tmp_path: Path) -> None:
    root = tmp_path / "sample.ssmd-bundle"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
    with pytest.raises(BundleContractError, match="size limit"):
        load_bundle_identity(root)

    outside = tmp_path / "manifest.json"
    outside.write_text('{"format":"ssmdconvert.book","schema_version":1}')
    manifest.unlink()
    try:
        manifest.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")
    with pytest.raises(BundleContractError, match="must not be a symlink"):
        load_bundle_identity(root)


def test_rejects_missing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "sample.ssmd-bundle"
    root.mkdir()

    with pytest.raises(BundleContractError, match="missing or unreadable"):
        load_bundle_identity(root)
