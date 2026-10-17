from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from renderplan import RenderProfile, RoleProfile

from voicerender import cli
from voicerender.renderprofile import load_render_profile, save_render_profile


def _write_profile(path: Path) -> RenderProfile:
    profile = RenderProfile(
        roles={
            "narrator": RoleProfile(
                language="en-US",
                binding={"renderer": "fixture", "target_id": "target-a"},
            ),
            "guest": RoleProfile(
                language="en-GB",
                binding={"renderer": "fixture", "target_id": "target-b"},
            ),
        },
        default_role="narrator",
    ).with_identity()
    save_render_profile(profile, path)
    return profile


def test_profile_role_set_writes_only_explicit_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "source.renderprofile.toml"
    output = tmp_path / "updated.renderprofile.toml"
    original = _write_profile(source)

    def fake_set_profile_role(profile, role, **kwargs):
        loaded = load_render_profile(profile)
        roles = dict(loaded.roles)
        roles[role] = RoleProfile(
            language=kwargs["language"],
            binding={
                "renderer": kwargs["engine"],
                "target_id": kwargs["target_id"],
                "options": dict(kwargs["options"] or {}),
            },
        )
        return replace(loaded, roles=roles, profile_id=None).with_identity()

    monkeypatch.setattr(cli, "set_profile_role", fake_set_profile_role)
    exit_code = cli.main(
        [
            "profile",
            "role",
            "set",
            str(source),
            "announcer",
            "--engine",
            "fixture",
            "--target",
            "target-c",
            "--language",
            "en-US",
            "--option",
            'style="warm"',
            "-o",
            str(output),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert load_render_profile(source) == original
    updated = load_render_profile(output)
    assert "announcer" in updated.roles
    assert payload["path"] == str(output)
    assert {item["role"] for item in payload["roles"]} == {
        "announcer",
        "guest",
        "narrator",
    }


def test_default_role_in_place_and_inspect_json(tmp_path: Path, capsys) -> None:
    source = tmp_path / "profile.renderprofile.toml"
    _write_profile(source)

    exit_code = cli.main(
        ["profile", "default-role", str(source), "guest", "--in-place", "--json"]
    )
    mutation = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert mutation["default_role"] == "guest"
    assert load_render_profile(source).default_role == "guest"

    exit_code = cli.main(["profile", "inspect", str(source), "--json"])
    inspection = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert inspection["default_role"] == "guest"
    assert [item["role"] for item in inspection["roles"]] == ["guest", "narrator"]
    assert next(item for item in inspection["roles"] if item["role"] == "guest")[
        "default"
    ]


def test_role_remove_default_requires_replacement_without_writing(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "profile.renderprofile.toml"
    output = tmp_path / "out.renderprofile.toml"
    _write_profile(source)
    original_bytes = source.read_bytes()

    exit_code = cli.main(
        [
            "profile",
            "role",
            "remove",
            str(source),
            "narrator",
            "-o",
            str(output),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert "--new-default-role" in payload["error"]
    assert source.read_bytes() == original_bytes
    assert not output.exists()


def test_role_list_human_output_is_descriptive_not_python_repr(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "profile.renderprofile.toml"
    _write_profile(source)

    exit_code = cli.main(["profile", "role", "list", str(source)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "* narrator  en-US  fixture/target-a" in output
    assert "guest  en-GB  fixture/target-b" in output
    assert "RoleProfile(" not in output
