from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import tomlkit
from renderplan import RenderProfile, RoleProfile, save_profile

from voicerender import cli
from voicerender.bundle_roles import (
    BundleRolesError,
    inspect_bundle_roles,
    read_bundle_role_inventory,
)
from voicerender.role_bindings import (
    duplicate_voice_groups,
    role_voice_identity,
    role_voice_selector,
)


def _make_bundle(
    tmp_path: Path,
    *,
    kind: str = "document",
    items: tuple[dict[str, Any], ...] | None = None,
    name: str = "sample",
) -> tuple[Path, Path]:
    root = tmp_path / f"{name}.ssmd-bundle"
    root.mkdir()
    manifest = root / "manifest.json"
    bundle_format = "ssmdconvert.book" if kind == "book" else "ssmdconvert.document"
    manifest.write_text(
        json.dumps({"format": bundle_format, "schema_version": 1}), encoding="utf-8"
    )
    source_specs: tuple[dict[str, Any], ...] = items or (
        {"id": "document-0001", "source": "document.ssmd.md", "roles": {"narrator": 2}},
    )
    aggregate_order: list[str] = []
    aggregate_counts: dict[str, int] = {}
    implicit_total = 0
    item_records: list[dict[str, Any]] = []
    for index, spec in enumerate(source_specs):
        source_relative = spec["source"]
        source_path = root / source_relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(f"source-{index}", encoding="utf-8")
        roles = dict(spec.get("roles", {}))
        implicit = spec.get("implicit", 0)
        for role, count in roles.items():
            if role not in aggregate_counts:
                aggregate_order.append(role)
                aggregate_counts[role] = 0
            aggregate_counts[role] += count
        implicit_total += implicit
        item_records.append(
            {
                "id": spec["id"],
                "source": source_relative,
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "plan": spec.get("plan", f"utterplan/{spec['id']}.utterplan.toml"),
                "plan_id": spec.get("plan_id", "sha256:" + str(index + 1) * 64),
                "explicit_roles": list(roles),
                "uses_default_role": implicit > 0,
                "implicit_occurrences": implicit,
                "explicit_occurrences": roles,
            }
        )
    inventory = {
        "format": "utterplan.bundle-role-inventory",
        "version": 1,
        "bundle_kind": kind,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "uses_default_role": implicit_total > 0,
        "implicit_occurrences": implicit_total,
        "roles": [
            {"name": name, "occurrences": aggregate_counts[name]}
            for name in aggregate_order
        ],
        "items": item_records,
    }
    inventory_dir = root / "utterplan"
    inventory_dir.mkdir()
    inventory_path = inventory_dir / "role-inventory.toml"
    inventory_path.write_text(tomlkit.dumps(inventory), encoding="utf-8")
    return root, inventory_path


def _rewrite_inventory(path: Path, change) -> None:
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(tomlkit.dumps(document), encoding="utf-8")


def _profile(
    path: Path,
    roles: dict[str, tuple[str, dict[str, Any]]],
    *,
    default_role: str,
) -> RenderProfile:
    profile = RenderProfile(
        roles={
            name: RoleProfile(language=language, binding=binding)
            for name, (language, binding) in roles.items()
        },
        default_role=default_role,
    ).with_identity()
    save_profile(profile, path)
    return profile


def _binding(
    *,
    renderer: str = "kokoro",
    target: str = "v1.0",
    voice: str | None = "voice-a",
    speaker: str | int | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"renderer": renderer, "target_id": target}
    if voice is not None:
        result["voice"] = voice
    if speaker is not None:
        result["speaker"] = speaker
    if options is not None:
        result["options"] = options
    return result


def test_document_and_book_inventory_models_preserve_role_order_and_default_counts(
    tmp_path: Path,
) -> None:
    for kind in ("document", "book"):
        root, _ = _make_bundle(
            tmp_path,
            kind=kind,
            name=kind,
            items=(
                {
                    "id": "first",
                    "source": "parts/first.md",
                    "roles": {"narrator": 2, "host": 3},
                    "implicit": 4,
                },
                {
                    "id": "second",
                    "source": "parts/second.md",
                    "roles": {"guest": 5, "narrator": 1},
                    "implicit": 2,
                },
            ),
        )
        loaded = read_bundle_role_inventory(root)
        assert loaded.inventory.bundle_kind == kind
        assert [role.name for role in loaded.inventory.roles] == [
            "narrator",
            "host",
            "guest",
        ]
        assert [role.occurrences for role in loaded.inventory.roles] == [3, 3, 5]
        assert loaded.inventory.uses_default_role is True
        assert loaded.inventory.implicit_occurrences == 6
        assert [item.id for item in loaded.inventory.items] == ["first", "second"]


def test_missing_inventory_has_actionable_stable_error(tmp_path: Path) -> None:
    root, inventory_path = _make_bundle(tmp_path)
    inventory_path.unlink()
    with pytest.raises(BundleRolesError) as error:
        read_bundle_role_inventory(root)
    assert error.value.code == "bundle.roles_missing"
    assert "utterplan bundle compile" in str(error.value)


def test_inventory_symlink_and_oversize_are_rejected(tmp_path: Path) -> None:
    root, inventory_path = _make_bundle(tmp_path)
    external = tmp_path / "external.toml"
    external.write_bytes(inventory_path.read_bytes())
    inventory_path.unlink()
    inventory_path.symlink_to(external)
    with pytest.raises(BundleRolesError) as error:
        read_bundle_role_inventory(root)
    assert error.value.code == "path.unsafe"

    inventory_path.unlink()
    inventory_path.write_bytes(b" " * (2 * 1024 * 1024 + 1))
    with pytest.raises(BundleRolesError) as error:
        read_bundle_role_inventory(root)
    assert error.value.code == "bundle.roles_invalid"
    assert "size limit" in str(error.value)


def test_inventory_requires_strict_utf8_toml_format_and_supported_version(
    tmp_path: Path,
) -> None:
    root, inventory_path = _make_bundle(tmp_path)
    inventory_path.write_bytes(b"\xff")
    with pytest.raises(BundleRolesError, match="UTF-8"):
        read_bundle_role_inventory(root)

    for field, value in (("format", "unknown"), ("version", 2)):
        root, inventory_path = _make_bundle(tmp_path, name=f"bad-{field}")
        _rewrite_inventory(
            inventory_path,
            lambda data, key=field, new=value: data.__setitem__(key, new),
        )
        with pytest.raises(BundleRolesError) as error:
            read_bundle_role_inventory(root)
        assert error.value.code == "bundle.roles_invalid"


def test_inventory_rejects_duplicate_roles_items_negative_counts_and_bad_aggregates(
    tmp_path: Path,
) -> None:
    mutations = (
        ("duplicate-role", lambda data: data["roles"].append(dict(data["roles"][0]))),
        ("duplicate-item", lambda data: data["items"].append(dict(data["items"][0]))),
        (
            "negative-count",
            lambda data: data["items"][0]["explicit_occurrences"].__setitem__(
                "narrator", -1
            ),
        ),
        (
            "aggregate-order",
            lambda data: data["roles"].append({"name": "extra", "occurrences": 1}),
        ),
    )
    for name, mutation in mutations:
        root, inventory_path = _make_bundle(tmp_path, name=name)
        _rewrite_inventory(inventory_path, mutation)
        with pytest.raises(BundleRolesError) as error:
            read_bundle_role_inventory(root)
        assert error.value.code == "bundle.roles_invalid"


def test_manifest_and_source_content_hashes_detect_stale_inventory(
    tmp_path: Path,
) -> None:
    root, _ = _make_bundle(tmp_path, name="manifest-stale")
    (root / "manifest.json").write_text(
        json.dumps(
            {"format": "ssmdconvert.document", "schema_version": 1, "changed": True}
        ),
        encoding="utf-8",
    )
    with pytest.raises(BundleRolesError) as error:
        read_bundle_role_inventory(root)
    assert error.value.code == "bundle.roles_stale"
    assert "utterplan bundle compile" in str(error.value)

    root, _ = _make_bundle(tmp_path, name="source-stale")
    (root / "document.ssmd.md").write_text("edited", encoding="utf-8")
    with pytest.raises(BundleRolesError) as error:
        read_bundle_role_inventory(root)
    assert error.value.code == "bundle.roles_stale"

    root, _ = _make_bundle(tmp_path, name="source-missing")
    (root / "document.ssmd.md").unlink()
    with pytest.raises(BundleRolesError) as error:
        read_bundle_role_inventory(root)
    assert error.value.code == "bundle.roles_stale"


def test_inventory_paths_are_posix_relative_and_source_symlinks_cannot_escape(
    tmp_path: Path,
) -> None:
    unsafe_paths = (
        "/absolute.md",
        "../outside.md",
        "C:/outside.md",
        "a\\b.md",
        "a//b.md",
    )
    for index, unsafe_path in enumerate(unsafe_paths):
        root, inventory_path = _make_bundle(tmp_path, name=f"unsafe-{index}")
        _rewrite_inventory(
            inventory_path,
            lambda data, value=unsafe_path: data["items"][0].__setitem__(
                "source", value
            ),
        )
        with pytest.raises(BundleRolesError) as error:
            read_bundle_role_inventory(root)
        assert error.value.code == "path.unsafe"

    root, _ = _make_bundle(tmp_path, name="symlink-escape")
    external = tmp_path / "outside.md"
    external.write_text("source-0", encoding="utf-8")
    (root / "document.ssmd.md").unlink()
    (root / "document.ssmd.md").symlink_to(external)
    with pytest.raises(BundleRolesError) as error:
        read_bundle_role_inventory(root)
    assert error.value.code == "path.unsafe"


def test_nested_source_and_custom_utterplan_directory_are_supported(
    tmp_path: Path,
) -> None:
    root, default_inventory = _make_bundle(
        tmp_path,
        items=(
            {"id": "part-1", "source": "chapters/one.md", "roles": {"narrator": 1}},
        ),
    )
    custom = tmp_path / "custom-utterplan"
    custom.mkdir()
    custom_inventory = custom / "role-inventory.toml"
    custom_inventory.write_bytes(default_inventory.read_bytes())
    loaded = read_bundle_role_inventory(root, utterplan_root=custom)
    assert loaded.path == custom_inventory.resolve()
    assert loaded.inventory.items[0].source == "chapters/one.md"


def test_source_only_mode_does_not_create_default_role_and_profile_coverage_is_ordered(
    tmp_path: Path,
) -> None:
    root, _ = _make_bundle(
        tmp_path,
        items=(
            {
                "id": "doc",
                "source": "document.ssmd.md",
                "roles": {"host": 2},
                "implicit": 3,
            },
        ),
    )
    source_only = inspect_bundle_roles(root)
    assert [row.role for row in source_only.roles] == ["host"]
    assert source_only.ready is None

    profile_path = tmp_path / "cast.renderprofile.toml"
    _profile(
        profile_path,
        {
            "narrator": (
                "en-US",
                _binding(renderer="piper", target="v2", voice="narrator"),
            ),
            "host": ("en-GB", _binding(renderer="kokoro", target="v1", voice="host")),
            "unused": ("en-US", _binding(renderer="piper", target="v3", voice="extra")),
        },
        default_role="narrator",
    )
    result = inspect_bundle_roles(root, profile=profile_path)
    assert result.coverage is not None
    assert result.coverage.required == ("host", "narrator")
    assert result.coverage.missing == ()
    assert result.coverage.unused == ("unused",)
    default_row = result.roles[1]
    assert default_row.required_by_default is True
    assert default_row.implicit_occurrences == 3
    assert result.ready is True
    assert json.loads(json.dumps(result.to_dict()))["ready"] is True


def test_missing_roles_are_reported_together_and_explicit_default_is_not_duplicated(
    tmp_path: Path,
) -> None:
    root, _ = _make_bundle(
        tmp_path,
        items=(
            {
                "id": "doc",
                "source": "document.ssmd.md",
                "roles": {"narrator": 4, "guest": 2, "analyst": 3},
                "implicit": 7,
            },
        ),
    )
    profile_path = tmp_path / "partial.renderprofile.toml"
    _profile(
        profile_path,
        {"narrator": ("en-US", _binding())},
        default_role="narrator",
    )
    result = inspect_bundle_roles(root, profile=profile_path)
    assert result.coverage is not None
    assert result.coverage.required == ("narrator", "guest", "analyst")
    assert result.coverage.missing == ("guest", "analyst")
    assert [row.role for row in result.roles] == ["narrator", "guest", "analyst"]
    assert result.roles[0].required_by_default is True
    assert result.roles[0].occurrences == 4
    assert result.roles[0].implicit_occurrences == 7
    assert result.ready is False


def test_duplicate_named_speaker_and_target_only_voice_identities() -> None:
    named_a = RoleProfile(
        language="en-US", binding=_binding(options={"speed": 1, "gain": 0.5})
    )
    named_b = RoleProfile(
        language="fr-FR", binding=_binding(options={"speed": 2, "gain": 0.8})
    )
    assert role_voice_selector(named_a) == "voice-a"
    groups = duplicate_voice_groups(
        (
            ("narrator", role_voice_identity(named_a)),
            ("analyst", role_voice_identity(named_b)),
        )
    )
    assert [group.roles for group in groups] == [("narrator", "analyst")]

    diff_renderer = RoleProfile(language="en-US", binding=_binding(renderer="piper"))
    diff_target = RoleProfile(language="en-US", binding=_binding(target="v2"))
    assert (
        duplicate_voice_groups(
            (
                ("one", role_voice_identity(named_a)),
                ("two", role_voice_identity(diff_renderer)),
            )
        )
        == ()
    )
    assert (
        duplicate_voice_groups(
            (
                ("one", role_voice_identity(named_a)),
                ("two", role_voice_identity(diff_target)),
            )
        )
        == ()
    )

    speaker_a = RoleProfile(language="en-US", binding=_binding(voice=None, speaker=3))
    speaker_b = RoleProfile(language="en-US", binding=_binding(voice=None, speaker=3))
    speaker_c = RoleProfile(language="en-US", binding=_binding(voice=None, speaker=4))
    assert role_voice_selector(speaker_a) == "speaker:3"
    assert (
        len(
            duplicate_voice_groups(
                (
                    ("a", role_voice_identity(speaker_a)),
                    ("b", role_voice_identity(speaker_b)),
                )
            )
        )
        == 1
    )
    assert (
        duplicate_voice_groups(
            (
                ("a", role_voice_identity(speaker_a)),
                ("c", role_voice_identity(speaker_c)),
            )
        )
        == ()
    )

    target_only_a = RoleProfile(
        language="en-US", binding={"renderer": "piper", "target_id": "fixed"}
    )
    target_only_b = RoleProfile(
        language="en-US", binding={"renderer": "piper", "target_id": "fixed"}
    )
    target_only_c = RoleProfile(
        language="en-US", binding={"renderer": "piper", "target_id": "other"}
    )
    assert (
        len(
            duplicate_voice_groups(
                (
                    ("a", role_voice_identity(target_only_a)),
                    ("b", role_voice_identity(target_only_b)),
                )
            )
        )
        == 1
    )
    assert (
        duplicate_voice_groups(
            (
                ("a", role_voice_identity(target_only_a)),
                ("c", role_voice_identity(target_only_c)),
            )
        )
        == ()
    )


def test_reference_and_prepared_voice_identity_ignores_local_paths() -> None:
    digest_a = "a" * 64
    ref_a = RoleProfile(
        language="en-US",
        binding={
            "renderer": "chatterbox",
            "target_id": "model-a",
            "options": {
                "voice_source": {
                    "kind": "reference",
                    "value": "/tmp/one.wav",
                    "sha256": digest_a,
                }
            },
        },
    )
    ref_copy = RoleProfile(
        language="en-GB",
        binding={
            "renderer": "chatterbox",
            "target_id": "model-a",
            "options": {
                "voice_source": {
                    "kind": "reference",
                    "value": "/elsewhere/copy.wav",
                    "sha256": digest_a,
                }
            },
        },
    )
    ref_other = RoleProfile(
        language="en-US",
        binding={
            "renderer": "chatterbox",
            "target_id": "model-a",
            "options": {
                "voice_source": {
                    "kind": "reference",
                    "value": "/tmp/other.wav",
                    "sha256": "b" * 64,
                }
            },
        },
    )
    assert (
        len(
            duplicate_voice_groups(
                (
                    ("one", role_voice_identity(ref_a)),
                    ("copy", role_voice_identity(ref_copy)),
                )
            )
        )
        == 1
    )
    assert (
        duplicate_voice_groups(
            (
                ("one", role_voice_identity(ref_a)),
                ("other", role_voice_identity(ref_other)),
            )
        )
        == ()
    )

    prepared_a = RoleProfile(
        language="en-US",
        binding={
            "renderer": "qwen3",
            "target_id": "model-a",
            "voice_source": {
                "kind": "prepared",
                "value": "/tmp/a.qvoice",
                "sha256": digest_a,
                "format": "qwen3-qvoice",
            },
        },
    )
    prepared_copy = RoleProfile(
        language="en-US",
        binding={
            "renderer": "qwen3",
            "target_id": "model-a",
            "voice_source": {
                "kind": "prepared",
                "value": "/tmp/b.qvoice",
                "sha256": digest_a,
                "format": "qwen3-qvoice",
            },
        },
    )
    assert (
        len(
            duplicate_voice_groups(
                (
                    ("one", role_voice_identity(prepared_a)),
                    ("copy", role_voice_identity(prepared_copy)),
                )
            )
        )
        == 1
    )


def test_extra_unused_role_sharing_required_voice_does_not_affect_distinctness(
    tmp_path: Path,
) -> None:
    root, _ = _make_bundle(
        tmp_path,
        items=({"id": "doc", "source": "document.ssmd.md", "roles": {"narrator": 1}},),
    )
    same = _binding()
    profile_path = tmp_path / "cast.renderprofile.toml"
    _profile(
        profile_path,
        {
            "narrator": ("en-US", same),
            "unused": ("en-US", same),
        },
        default_role="narrator",
    )
    result = inspect_bundle_roles(root, profile=profile_path)
    assert result.coverage is not None and result.coverage.unused == ("unused",)
    assert result.distinctness is not None and result.distinctness.distinct
    assert result.ready is True


def test_invalid_required_binding_is_reported_without_engine_validation(
    tmp_path: Path,
) -> None:
    root, _ = _make_bundle(tmp_path)
    profile_path = tmp_path / "invalid.renderprofile.toml"
    _profile(
        profile_path,
        {"narrator": ("en-US", {"voice": "voice-without-target"})},
        default_role="narrator",
    )
    result = inspect_bundle_roles(root, profile=profile_path)
    assert result.roles[0].bound is True
    assert result.roles[0].binding_valid is False
    assert result.ready is False


def test_bundle_roles_cli_outputs_and_readiness_exit_codes(
    tmp_path: Path, capsys
) -> None:
    root, _ = _make_bundle(
        tmp_path,
        items=(
            {
                "id": "doc",
                "source": "document.ssmd.md",
                "roles": {"narrator": 1, "host": 1},
            },
        ),
    )
    assert cli.main(["bundle", "roles", str(root)]) == 0
    human = capsys.readouterr().out
    assert "VoiceRender bundle roles" in human
    assert "narrator" in human and "host" in human
    assert "VOICE/SPEAKER/SOURCE" not in human

    assert cli.main(["bundle", "roles", str(root), "--json"]) == 0
    source_json = capsys.readouterr().out
    source_payload = json.loads(source_json)
    assert source_payload["format"] == "voicerender-bundle-roles"
    assert source_payload["profile"] is None
    assert "VoiceRender bundle roles" not in source_json

    complete_path = tmp_path / "complete.renderprofile.toml"
    _profile(
        complete_path,
        {
            "narrator": (
                "en-US",
                _binding(renderer="kokoro", target="v1", voice="one"),
            ),
            "host": ("en-GB", _binding(renderer="piper", target="v2", voice="two")),
        },
        default_role="narrator",
    )
    assert (
        cli.main(
            [
                "bundle",
                "roles",
                str(root),
                "--profile",
                str(complete_path),
                "--check",
                "--json",
            ]
        )
        == 0
    )
    complete_payload = json.loads(capsys.readouterr().out)
    assert complete_payload["coverage"]["complete"] is True
    assert complete_payload["ready"] is True

    partial_path = tmp_path / "partial.renderprofile.toml"
    _profile(
        partial_path,
        {"narrator": ("en-US", _binding())},
        default_role="narrator",
    )

    assert (
        cli.main(
            ["bundle", "roles", str(root), "--profile", str(partial_path), "--json"]
        )
        == 0
    )
    partial_inspection = json.loads(capsys.readouterr().out)
    assert partial_inspection["ready"] is False
    assert (
        cli.main(
            [
                "bundle",
                "roles",
                str(root),
                "--profile",
                str(partial_path),
                "--check",
                "--json",
            ]
        )
        == 1
    )
    partial_payload = json.loads(capsys.readouterr().out)
    assert partial_payload["coverage"]["missing"] == ["host"]
    assert partial_payload["ready"] is False

    duplicate_path = tmp_path / "duplicate.renderprofile.toml"
    shared = _binding(renderer="kokoro", target="v1", voice="shared")
    _profile(
        duplicate_path,
        {"narrator": ("en-US", shared), "host": ("en-US", shared)},
        default_role="narrator",
    )
    assert (
        cli.main(
            [
                "bundle",
                "roles",
                str(root),
                "--profile",
                str(duplicate_path),
                "--check",
                "--json",
            ]
        )
        == 1
    )
    duplicate_payload = json.loads(capsys.readouterr().out)
    assert duplicate_payload["distinctness"]["distinct"] is False

    assert (
        cli.main(
            ["bundle", "roles", str(root), "--profile", str(duplicate_path), "--json"]
        )
        == 0
    )
    duplicate_inspection = json.loads(capsys.readouterr().out)
    assert duplicate_inspection["ready"] is False
    assert duplicate_payload["distinctness"]["duplicate_groups"][0]["roles"] == [
        "narrator",
        "host",
    ]

    assert cli.main(["bundle", "roles", str(root), "--check", "--json"]) == 2
    error_json = capsys.readouterr().out
    assert json.loads(error_json)["error_code"] == "bundle.roles_invalid"
    assert "error_code" in error_json and "VoiceRender bundle roles" not in error_json

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "bundle",
                "roles",
                str(root),
                "--profile",
                str(complete_path),
                "--variant",
                "variant",
            ]
        )
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err

    stale_root, _ = _make_bundle(tmp_path, name="cli-stale")
    (stale_root / "document.ssmd.md").write_text("changed", encoding="utf-8")
    assert cli.main(["bundle", "roles", str(stale_root), "--json"]) == 2
    stale_payload = json.loads(capsys.readouterr().out)
    assert stale_payload["error_code"] == "bundle.roles_stale"
