from __future__ import annotations

import json
from pathlib import Path

from renderplan import RenderProfile, RoleProfile

from voicerender import cli


def test_voice_clone_cli_emits_stable_json_and_persists_profile(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference audio")
    output = tmp_path / "clone.renderprofile.toml"
    profile = RenderProfile(
        roles={
            "narrator": RoleProfile(
                language="en-US",
                binding={"renderer": "fixture", "target_id": "target-a"},
            )
        },
        default_role="narrator",
    ).with_identity()
    requests = []

    def clone(request, *, role):
        requests.append((request, role))
        return profile

    monkeypatch.setattr(cli, "clone_voice_profile", clone)
    exit_code = cli.main(
        [
            "voice",
            "clone",
            "--engine",
            "fixture",
            "--target",
            "target-a",
            "--language",
            "en-US",
            "--reference-wav",
            str(reference),
            "--transcript",
            "Exact words.",
            "--option",
            "temperature=0.7",
            "-o",
            str(output),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {
        "path": str(output),
        "profile_id": profile.profile_id,
        "role": "narrator",
    }
    assert len(requests) == 1
    request, role = requests[0]
    assert request.engine == "fixture"
    assert request.transcript == "Exact words."
    assert request.options == {"temperature": 0.7}
    assert role == "narrator"
    assert output.exists()
