from __future__ import annotations

import json
from pathlib import Path

from voicerender import cli, voice_sources
from voicerender.voice_source_codec import load_voice_source


def test_source_create_inspect_and_verify_pins_metadata_and_never_mutates_audio(
    tmp_path: Path, capsys
) -> None:
    wav = tmp_path / "speaker.wav"
    original = b"exact reference WAV bytes"
    wav.write_bytes(original)
    sidecar = tmp_path / "speaker.voice-source.toml"

    create_exit = cli.main(
        [
            "voice",
            "source",
            "create",
            "--reference-wav",
            str(wav),
            "--language",
            "en-US",
            "--transcript",
            "Exact words.",
            "--name",
            "Speaker",
            "--metadata",
            "rating=5",
            "-o",
            str(sidecar),
            "--json",
        ]
    )
    created = json.loads(capsys.readouterr().out)

    assert create_exit == 0
    source = load_voice_source(sidecar)
    assert source["kind"] == "reference"
    assert source["metadata"] == {
        "language": "en-US",
        "name": "Speaker",
        "rating": 5,
        "transcript": "Exact words.",
    }
    assert wav.read_bytes() == original
    assert created["voice_source"]["sha256"] == source["sha256"]

    inspect_exit = cli.main(["voice", "source", "inspect", str(sidecar), "--json"])
    inspected = json.loads(capsys.readouterr().out)
    assert inspect_exit == 0
    assert inspected["metadata"]["transcript"] == "Exact words."

    verify_exit = cli.main(["voice", "source", "verify", str(sidecar), "--json"])
    verified = json.loads(capsys.readouterr().out)
    assert verify_exit == 0
    assert verified["valid"] is True

    wav.write_bytes(b"changed bytes")
    changed_exit = cli.main(["voice", "source", "verify", str(sidecar), "--json"])
    changed = json.loads(capsys.readouterr().out)
    assert changed_exit == 1
    assert changed["valid"] is False
    assert changed["diagnostics"][0]["code"] == "voice_source.reference_changed"


def test_managed_source_creation_keeps_pins_without_fetching_audio(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    source = {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "value": "catalog:voice-1",
        "sha256": "a" * 64,
        "source_revision": "revision-1",
        "source_repository": "voice-repo",
        "source_path": "voices/reference.wav",
        "license": "CC0-1.0",
    }
    monkeypatch.setattr(
        voice_sources, "managed_reference_source", lambda ref: dict(source)
    )
    monkeypatch.setattr(
        voice_sources, "validate_managed_voice", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        voice_sources,
        "resolve_managed_voice",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("audio fetch attempted")
        ),
    )
    sidecar = tmp_path / "managed.voice-source.toml"

    exit_code = cli.main(
        [
            "voice",
            "source",
            "create",
            "--voicefolio-ref",
            "catalog:voice-1",
            "--language",
            "en-US",
            "-o",
            str(sidecar),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)

    assert exit_code == 0
    loaded = load_voice_source(sidecar)
    assert loaded["kind"] == "managed_reference"
    assert loaded["sha256"] == "a" * 64
    assert loaded["source_revision"] == "revision-1"
    assert loaded["metadata"]["language"] == "en-US"
    assert payload["voice_source"]["resolver"] == "voicefolio"


def test_source_creation_error_has_stable_json_payload(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.wav"
    sidecar = tmp_path / "missing.toml"

    exit_code = cli.main(
        [
            "voice",
            "source",
            "create",
            "--reference-wav",
            str(missing),
            "-o",
            str(sidecar),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["error_code"] == "voice_source.reference_missing"
