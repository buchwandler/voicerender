from __future__ import annotations

import json
from pathlib import Path

import tomlkit
from renderplan import load_profile

from voicerender.cli import main
from voicerender.renderprofile import reidentify_render_profile


def test_reidentify_helper_uses_typed_identity_without_engine_validation() -> None:
    source = Path(__file__).parents[1] / "examples/voices/narration.renderprofile.toml"
    profile = load_profile(source)
    result = reidentify_render_profile(profile)
    assert result.profile_id == profile.computed_id()
    assert result.roles == profile.roles
    assert result.planning == profile.planning
    assert result.pauses == profile.pauses


def _stale_profile(tmp_path: Path) -> tuple[Path, bytes]:
    source = Path(__file__).parents[1] / "examples/voices/narration.renderprofile.toml"
    text = source.read_text(encoding="utf-8")
    document = tomlkit.parse(text)
    document["profile_id"] = "sha256:" + "0" * 64
    roles = document["roles"]
    role = next(iter(roles.values()))
    role["binding"].setdefault("options", {})["quality"] = "high"
    path = tmp_path / "stale.renderprofile.toml"
    path.write_text(tomlkit.dumps(document), encoding="utf-8")
    return path, path.read_bytes()


def test_reidentify_output_preserves_source_and_profile_content(
    tmp_path: Path, capsys
) -> None:
    source, original = _stale_profile(tmp_path)
    output = tmp_path / "reidentified.renderprofile.toml"

    assert (
        main(["profile", "reidentify", str(source), "--output", str(output), "--json"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert set(result) == {"path", "old_profile_id", "profile_id", "changed"}
    assert result["old_profile_id"] == "sha256:" + "0" * 64
    assert result["changed"] is True
    assert source.read_bytes() == original
    profile = load_profile(output)
    assert profile.profile_id == result["profile_id"]
    assert profile.roles[profile.default_role].binding["options"]["quality"] == "high"


def test_reidentify_in_place_is_idempotent_and_rejects_same_output(
    tmp_path: Path, capsys
) -> None:
    source, _ = _stale_profile(tmp_path)
    assert main(["profile", "reidentify", str(source), "--in-place"]) == 0
    capsys.readouterr()
    repaired = source.read_bytes()
    assert main(["profile", "reidentify", str(source), "--in-place", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["changed"] is False
    assert source.read_bytes() == repaired
    assert main(["profile", "reidentify", str(source), "--output", str(source)]) != 0


def test_stale_identity_remains_strict_for_inspect_and_invalid_input_is_untouched(
    tmp_path: Path, capsys
) -> None:
    source, original = _stale_profile(tmp_path)
    assert main(["profile", "inspect", str(source)]) != 0
    capsys.readouterr()
    assert source.read_bytes() == original

    invalid = tmp_path / "invalid.renderprofile.toml"
    invalid.write_text("not = [valid", encoding="utf-8")
    before = invalid.read_bytes()
    assert main(["profile", "reidentify", str(invalid), "--in-place"]) != 0
    capsys.readouterr()
    assert invalid.read_bytes() == before


def test_reidentify_supports_missing_profile_id(tmp_path: Path, capsys) -> None:
    source = Path(__file__).parents[1] / "examples/voices/narration.renderprofile.toml"
    document = tomlkit.parse(source.read_text(encoding="utf-8"))
    del document["profile_id"]
    path = tmp_path / "missing-id.renderprofile.toml"
    path.write_text(tomlkit.dumps(document), encoding="utf-8")
    output = tmp_path / "repaired.renderprofile.toml"

    assert (
        main(["profile", "reidentify", str(path), "--output", str(output), "--json"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["old_profile_id"] is None
    assert result["changed"] is True
    assert load_profile(output).profile_id == result["profile_id"]
