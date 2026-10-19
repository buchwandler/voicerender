from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
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


def test_profile_bind_delegates_to_role_set_and_requires_destination(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "source.renderprofile.toml"
    original = _write_profile(source)
    calls: dict[str, object] = {}

    def fake_set_profile_role(profile, role, **kwargs):
        calls.update(kwargs)
        loaded = load_render_profile(profile)
        roles = dict(loaded.roles)
        roles[role] = RoleProfile(
            language=kwargs["language"],
            binding={"renderer": kwargs["engine"], "target_id": kwargs["target_id"]},
        )
        default_role = role if kwargs["make_default"] else loaded.default_role
        return replace(
            loaded, roles=roles, default_role=default_role, profile_id=None
        ).with_identity()

    monkeypatch.setattr(cli, "set_profile_role", fake_set_profile_role)
    exit_code = cli.main(
        [
            "profile",
            "bind",
            str(source),
            "host",
            "--engine",
            "kitten",
            "--target",
            "mini-0.8",
            "--language",
            "en-US",
            "--voice",
            "Jasper",
            "--offline",
            "--make-default",
            "--in-place",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls["voice"] == "Jasper"
    assert calls["offline"] is True
    assert calls["make_default"] is True
    assert load_render_profile(source).default_role == "host"
    assert payload["path"] == str(source)
    assert original.default_role == "narrator"

    parser = cli.build_parser()
    command = [
        "profile",
        "bind",
        str(source),
        "guest",
        "--engine",
        "piper",
        "--target",
        "en_US-amy-medium",
        "--language",
        "en-US",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(command)
    with pytest.raises(SystemExit):
        parser.parse_args([*command, "--in-place", "-o", str(tmp_path / "out.toml")])


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


def test_role_list_and_profile_show_present_the_complete_cast(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "profile.renderprofile.toml"
    profile = _write_profile(source)
    roles = dict(profile.roles)
    roles["narrator"] = replace(
        roles["narrator"],
        binding={
            **roles["narrator"].binding,
            "voice": "af_heart",
            "renderer_metadata": {"private_detail": "not shown"},
        },
    )
    roles["guest"] = replace(
        roles["guest"],
        binding={**roles["guest"].binding, "speaker": "Jasper"},
    )
    roles["source"] = RoleProfile(
        language="en-US",
        binding={
            "renderer": "piper",
            "target_id": "en_US-amy-medium",
            "options": {
                "voice_source": {
                    "kind": "managed_reference",
                    "reference": "large-reference-mapping-is-not-displayed",
                }
            },
        },
    )
    save_render_profile(
        replace(profile, roles=roles, profile_id=None).with_identity(), source
    )

    exit_code = cli.main(["profile", "role", "list", str(source)])
    role_output = capsys.readouterr().out
    assert exit_code == 0
    assert "VOICE/SPEAKER/SOURCE" in role_output
    assert "af_heart" in role_output
    assert "speaker:Jasper" in role_output
    assert "source:managed_reference" in role_output
    assert "private_detail" not in role_output
    assert "large-reference-mapping-is-not-displayed" not in role_output
    assert "RoleProfile(" not in role_output

    exit_code = cli.main(["profile", "show", str(source)])
    show_output = capsys.readouterr().out
    assert exit_code == 0
    assert "RenderProfile " in show_output
    assert "Planning:" in show_output
    assert "Pauses:" in show_output
    assert "af_heart" in show_output
    assert "speaker:Jasper" in show_output
    assert "source:managed_reference" in show_output

    exit_code = cli.main(["profile", "show", str(source), "--json"])
    show_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    source_role = next(
        item for item in show_payload["roles"] if item["role"] == "source"
    )
    assert source_role["binding"]["options"]["voice_source"]["reference"] == (
        "large-reference-mapping-is-not-displayed"
    )
    narrator_role = next(
        item for item in show_payload["roles"] if item["role"] == "narrator"
    )
    assert "private_detail" in narrator_role["binding"]["renderer_metadata"]
