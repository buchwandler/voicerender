from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import tomlkit
from renderplan import (
    RenderPlan,
    RoleProfile,
    SourceSpan,
    SpeechItem,
    save_plan,
    save_profile,
)

from voicerender import cli
from voicerender.bundle_contract import BundleContractError
from voicerender.bundle_inspect import inspect_bundle
from voicerender.variants import list_bundle_variants


def _write_bundle(tmp_path: Path, name: str = "bundle") -> Path:
    root = tmp_path / f"{name}.ssmd-bundle"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "ssmdconvert.document",
                "schema_version": 1,
                "ssmd_version": "0.9",
                "document": {"id": "document-0001", "path": "document.ssmd.md"},
            }
        ),
        encoding="utf-8",
    )
    source = root / "document.ssmd.md"
    source.write_text("A small document.", encoding="utf-8")
    utterplan = root / "utterplan"
    utterplan.mkdir()
    inventory: dict[str, Any] = {
        "format": "utterplan.bundle-role-inventory",
        "version": 1,
        "bundle_kind": "document",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "uses_default_role": False,
        "implicit_occurrences": 0,
        "roles": [{"name": "narrator", "occurrences": 1}],
        "items": [
            {
                "id": "document-0001",
                "source": "document.ssmd.md",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "plan": "utterplan/document.utterplan.toml",
                "plan_id": "sha256:" + "1" * 64,
                "explicit_roles": ["narrator"],
                "uses_default_role": False,
                "implicit_occurrences": 0,
                "explicit_occurrences": {"narrator": 1},
            }
        ],
    }
    (utterplan / "role-inventory.toml").write_text(
        tomlkit.dumps(inventory), encoding="utf-8"
    )
    return root


def _make_profile(voice: str, *, target: str = "model-a"):
    from renderplan import RenderProfile

    return RenderProfile(
        roles={
            "narrator": RoleProfile(
                language="en-US",
                binding={"renderer": "kokoro", "target_id": target, "voice": voice},
            )
        },
        default_role="narrator",
    ).with_identity()


def _write_variant(root: Path, voice: str) -> tuple[str, Any]:
    profile = _make_profile(voice)
    profile_id = profile.profile_id or profile.computed_id()
    variant_name = f"narration--{profile_id.removeprefix('sha256:')[:12]}"
    variant_dir = root / "renderplan" / variant_name
    variant_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")
    speech = SpeechItem(
        number=1,
        text="This is a fixture.",
        language="en-US",
        role="narrator",
        binding={"renderer": "kokoro", "target_id": "model-a"},
        source=(SourceSpan(locator="document.ssmd.md", start=1, end=2),),
    )
    plan = RenderPlan(
        source_plan_id="sha256:" + "2" * 64,
        render_profile_id=profile_id,
        flow=(speech,),
    ).with_identity()
    save_plan(plan, variant_dir / "document.renderplan.toml")
    return variant_name, profile


def test_inspect_without_profile_or_variants_is_read_only(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    result = inspect_bundle(root)
    assert result.roles.ready is None
    assert result.roles.coverage is None
    assert result.variants.variants == ()
    assert not (root / "renderplan").exists()
    assert not (root / "voicerender").exists()
    assert result.to_dict()["format"] == "voicerender-bundle-inspect"


def test_inspect_lists_multiple_variants_and_reuses_existing_render_state(
    tmp_path: Path,
) -> None:
    root = _write_bundle(tmp_path)
    first, _ = _write_variant(root, "voice-a")
    second, _ = _write_variant(root, "voice-b")
    invalid = root / "renderplan" / "broken--variant"
    invalid.mkdir()
    output_root = tmp_path / "custom-output-root"

    result = inspect_bundle(root, output_root=output_root)
    assert {variant.name for variant in result.variants.variants} >= {first, second}
    assert result.variants.output_root == output_root.resolve()
    assert all(
        variant.render_status == "not_rendered"
        for variant in result.variants.variants
        if variant.name in {first, second}
    )
    invalid_result = next(
        variant for variant in result.variants.variants if variant.name == invalid.name
    )
    assert invalid_result.plan_status == "invalid"
    assert invalid_result.validation_error
    assert not output_root.exists()
    assert not (root / "voicerender").exists()
    assert (
        result.variants.to_dict()
        == list_bundle_variants(root, output_root=output_root).to_dict()
    )


def test_selected_variant_uses_immutable_profile_snapshot_not_authored_file(
    tmp_path: Path,
) -> None:
    root = _write_bundle(tmp_path)
    variant_name, snapshot = _write_variant(root, "snapshot-voice")
    authored_path = tmp_path / "mutable.renderprofile.toml"
    authored = _make_profile("current-voice", target="current-model")
    save_profile(authored, authored_path)

    selected = inspect_bundle(root, variant=variant_name)
    assert selected.roles.profile_source == "variant"
    assert selected.roles.profile_path == (
        root / "renderplan" / variant_name / "renderprofile.toml"
    )
    assert selected.roles.profile_id == snapshot.profile_id
    assert selected.roles.roles[0].voice_identity is not None
    assert selected.roles.roles[0].voice_identity.voice == "snapshot-voice"
    assert selected.roles.ready is True

    current = inspect_bundle(root, profile=authored_path)
    assert current.roles.profile_source == "profile"
    assert current.roles.roles[0].voice_identity is not None
    assert current.roles.roles[0].voice_identity.voice == "current-voice"


def test_inspect_profile_with_no_variants_reports_readiness(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    authored_path = tmp_path / "cast.renderprofile.toml"
    save_profile(_make_profile("ready-without-variant"), authored_path)

    result = inspect_bundle(root, profile=authored_path)
    assert result.roles.ready is True
    assert result.roles.profile_source == "profile"
    assert result.variants.variants == ()
    assert not (root / "renderplan").exists()
    assert not (root / "voicerender").exists()


def test_check_readiness_does_not_require_render_completion(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    variant_name, _ = _write_variant(root, "ready-voice")
    result = inspect_bundle(root, variant=variant_name)
    assert result.roles.ready is True
    assert result.variants.variants[0].render_status == "not_rendered"


def test_invalid_selected_variant_is_actionable(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    with pytest.raises(BundleContractError) as error:
        inspect_bundle(root, variant="missing--variant")
    assert error.value.code == "variant.invalid"
    assert "missing--variant" in str(error.value)


def test_render_state_matches_existing_variants_api(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    _write_variant(root, "same-state")
    inspected = inspect_bundle(root)
    listed = list_bundle_variants(root)
    assert [item.to_dict() for item in inspected.variants.variants] == [
        item.to_dict() for item in listed.variants
    ]


def test_bundle_roles_and_inspect_cli_variant_human_json_and_check(
    tmp_path: Path, capsys
) -> None:
    root = _write_bundle(tmp_path)
    variant_name, _ = _write_variant(root, "cli-snapshot")

    assert (
        cli.main(
            [
                "bundle",
                "roles",
                str(root),
                "--variant",
                variant_name,
                "--check",
                "--json",
            ]
        )
        == 0
    )
    roles_json = capsys.readouterr().out
    roles_payload = json.loads(roles_json)
    assert roles_payload["profile"]["source"] == "variant"
    assert roles_payload["ready"] is True
    assert "VoiceRender bundle roles" not in roles_json

    assert cli.main(["bundle", "inspect", str(root), "--json"]) == 0
    inspect_json = capsys.readouterr().out
    inspect_payload = json.loads(inspect_json)
    assert inspect_payload["format"] == "voicerender-bundle-inspect"
    assert len(inspect_payload["variants"]["variants"]) == 1

    assert cli.main(["bundle", "inspect", str(root)]) == 0
    human = capsys.readouterr().out
    assert "VoiceRender bundle inspect" in human
    assert "Variants" in human
    assert "\nCast\n" not in human

    assert cli.main(["bundle", "inspect", str(root), "--check", "--json"]) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["error_code"] == "bundle.roles_invalid"
