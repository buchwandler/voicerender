from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from renderplan import PausePolicy, RenderProfile, RoleProfile

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
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference audio")
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
            "--reference-wav",
            str(reference),
            "--transcript",
            "Exact reference words.",
            "--offline",
            "--make-default",
            "--in-place",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls["voice_source"]["kind"] == "reference"
    assert calls["transcript"] == "Exact reference words."
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


def test_profile_inspection_exposes_effective_pronunciation_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    source = tmp_path / "kokoro.renderprofile.toml"
    profile = _write_profile(source)
    roles = dict(profile.roles)
    binding = {
        "renderer": "kokoro",
        "target_id": "kokoro-test",
        "options": {
            "lexicons": ["lexhint"],
            "g2lex_files": [
                {
                    "path": str(tmp_path / "editorial.g2lex"),
                    "sha256": "sha256:" + "a" * 64,
                }
            ],
        },
    }
    roles["narrator"] = replace(roles["narrator"], binding=binding)

    save_render_profile(
        replace(profile, roles=roles, profile_id=None).with_identity(), source
    )

    class FakeAdapter:
        def pronunciation_stack(self, *, language, target_id, options):
            files = options.get("g2lex_files", [])
            named = options.get("lexicons", [])
            local = (
                [
                    {
                        "path": files[0]["path"],
                        "sha256": files[0]["sha256"],
                        "status": "valid",
                    }
                ]
                if files
                else []
            )
            precedence = [
                f"local G2Lex file {index + 1}" for index in range(len(local))
            ]
            precedence.extend(["contextual heteronym rules", *named, "espeak"])
            return {
                "engine": "kokoro",
                "language": language,
                "target_id": target_id,
                "local_g2lex": local,
                "named_lexicons": [{"selector": item} for item in named],
                "precedence": precedence,
            }

    monkeypatch.setattr(cli, "get_adapter", lambda _renderer: FakeAdapter())
    assert cli.main(["profile", "inspect", str(source), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    narrator = next(item for item in payload["roles"] if item["role"] == "narrator")
    assert narrator["pronunciation_stack"]["precedence"][0] == "local G2Lex file 1"
    assert (
        narrator["pronunciation_stack"]["local_g2lex"][0]["sha256"]
        == "sha256:" + "a" * 64
    )

    assert cli.main(["profile", "inspect", str(source)]) == 0
    assert (
        "Pronunciation stack: local G2Lex file 1 > contextual heteronym rules"
        in capsys.readouterr().out
    )


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
            "renderer": "kokoro",
            "options": {},
            "voice": "af_heart",
            "renderer_metadata": {"private_detail": "not shown"},
        },
    )
    roles["guest"] = replace(
        roles["guest"],
        binding={
            **roles["guest"].binding,
            "renderer": "kokoro",
            "speaker": "Jasper",
            "options": {
                "short_sentence": {
                    "enabled": True,
                    "resolve_mode": "wrap",
                    "min_phoneme_length": 24,
                }
            },
        },
    )
    roles["explicit-off"] = RoleProfile(
        language="en-US",
        binding={
            "renderer": "kokoro",
            "target_id": "kokoro-test",
            "options": {"short_sentence": "off"},
        },
    )
    roles["source"] = RoleProfile(
        language="en-US",
        binding={
            "renderer": "piper",
            "target_id": "en_US-amy-medium",
            "options": {
                "voice_source": {
                    "kind": "managed_reference",
                    "resolver": "voicefolio",
                    "value": "fine-read-it-voices:S5339",
                    "sha256": "a" * 64,
                    "source_format": "wav",
                },
                "voice_enrollment": {
                    "engine": "akinvox",
                    "reference_text": "Exact transcript must not be printed.",
                },
            },
        },
    )
    save_render_profile(
        replace(
            profile,
            roles=roles,
            pauses=PausePolicy(mode="none"),
            profile_id=None,
        ).with_identity(),
        source,
    )

    exit_code = cli.main(["profile", "role", "list", str(source)])
    role_output = capsys.readouterr().out
    assert exit_code == 0
    assert "VOICE/SPEAKER/SOURCE" in role_output
    assert "af_heart" in role_output
    assert "speaker:Jasper" in role_output
    assert "source:managed_reference" in role_output
    assert "private_detail" not in role_output
    assert "fine-read-it-voices:S5339" in role_output
    assert "enrollment:akinvox" in role_output
    assert "Exact transcript must not be printed." not in role_output
    assert "RoleProfile(" not in role_output

    exit_code = cli.main(["profile", "show", str(source)])
    show_output = capsys.readouterr().out
    assert exit_code == 0
    assert "RenderProfile " in show_output
    assert "Planning:" in show_output
    assert "Pauses\n  mode: none" in show_output
    assert "speech artifact tails: trimmed" in show_output
    assert "no policy pause is inserted" in show_output
    assert "short sentence: off (implicit)" in show_output
    assert "short sentence: off (explicit)" in show_output
    assert "short sentence: wrap" in show_output
    assert "threshold: <24 phonemes" in show_output
    assert "enabled: yes" in show_output
    assert "af_heart" in show_output
    assert "speaker:Jasper" in show_output
    assert "source:managed_reference" in show_output

    exit_code = cli.main(["profile", "show", str(source), "--json"])
    show_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    source_role = next(
        item for item in show_payload["roles"] if item["role"] == "source"
    )
    source_binding = source_role["binding"]["options"]
    assert source_binding["voice_source"] == {
        "kind": "managed_reference",
        "ref": "fine-read-it-voices:S5339",
        "resolver": "voicefolio",
        "sha256": "a" * 64,
        "source_format": "wav",
    }
    enrollment = source_binding["voice_enrollment"]
    assert enrollment == {
        "engine": "akinvox",
        "reference_text_sha256": hashlib.sha256(
            b"Exact transcript must not be printed."
        ).hexdigest(),
    }
    assert "Exact transcript must not be printed." not in json.dumps(show_payload)
    narrator_role = next(
        item for item in show_payload["roles"] if item["role"] == "narrator"
    )
    assert "private_detail" in narrator_role["binding"]["renderer_metadata"]


@pytest.mark.parametrize("mode", ("off", "wrap", "phrase", "randomized-phrase"))
def test_profile_bind_short_sentence_flag_persists_engine_option(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    source = tmp_path / "source.renderprofile.toml"
    output = tmp_path / "bound.renderprofile.toml"
    _write_profile(source)
    captured: dict[str, object] = {}

    def fake_set_profile_role(profile_path, role, **kwargs):
        captured.update(kwargs)
        loaded = load_render_profile(profile_path)
        roles = dict(loaded.roles)
        current = roles[role]
        binding = {
            **current.binding,
            "renderer": kwargs["engine"],
            "target_id": kwargs["target_id"],
            "options": dict(kwargs["options"] or {}),
        }
        roles[role] = replace(current, binding=binding)
        return replace(loaded, roles=roles, profile_id=None).with_identity()

    monkeypatch.setattr(cli, "set_profile_role", fake_set_profile_role)
    result = cli.main(
        [
            "profile",
            "bind",
            str(source),
            "narrator",
            "--engine",
            "kokoro",
            "--target",
            "kokoro-0.8",
            "--language",
            "en-US",
            "--short-sentence",
            mode,
            "--output",
            str(output),
            "--json",
        ]
    )

    assert result == 0
    assert captured["options"]["short_sentence"] == mode
    bound = load_render_profile(output)
    assert bound.roles["narrator"].binding["options"]["short_sentence"] == mode


def test_profile_show_explains_manual_pauses_follow_normalized_speech(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "manual.renderprofile.toml"
    profile = _write_profile(source)
    profile = replace(
        profile,
        pauses=PausePolicy(
            mode="manual",
            durations_ms={
                "clause": 180,
                "parenthetical": 180,
                "sentence": 500,
                "paragraph": 800,
            },
        ),
        profile_id=None,
    ).with_identity()
    save_render_profile(profile, source)

    assert cli.main(["profile", "show", str(source)]) == 0
    output = capsys.readouterr().out
    assert "Pauses\n  mode: manual" in output
    assert "  clause: 180" in output
    assert "  sentence: 500" in output
    assert "  paragraph: 800" in output
    assert "  parenthetical: 180" in output
    assert "speech artifact tails: trimmed" in output
    assert "configured pause is inserted after normalized speech" in output
