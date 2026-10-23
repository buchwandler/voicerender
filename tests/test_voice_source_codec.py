from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from voicerender.errors import InvalidEngineVoiceError
from voicerender.voice_source_codec import load_voice_source, save_voice_source
from voicerender.voice_sources import (
    local_prepared_voice_source,
    local_reference_source,
    normalize_voice_source,
    resolve_prepared_voice,
    resolve_reference_audio,
    validate_voice_source,
    voice_source_identity,
)


def _qwen_origin() -> dict[str, object]:
    return {
        "schema": "voicerender.voice-origin.qwen3-design.v1",
        "generator": "qwen3",
        "target_id": "voice-design-1.7b",
        "model_revision": "qwen3-voice-design-v1",
        "qwen3synth_version": "0.1.0",
        "onnxvoice_version": "0.1.0",
        "prompt": "A warm, measured speaking voice.",
        "reference_text": "A short reference sentence.",
        "language": "English",
        "quality": "standard",
        "provider": "cpu",
        "temperature": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_new_tokens": 128,
        "seed": 7,
    }


def test_qwen_origin_is_normalized_but_not_part_of_voice_identity(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "design.wav"
    reference.write_bytes(b"materialized voice bytes")
    source = local_reference_source(reference)
    source["origin"] = _qwen_origin()

    normalized = normalize_voice_source(source)

    assert normalized.as_mapping()["origin"] == _qwen_origin()
    assert voice_source_identity(normalized) == {
        "kind": "reference",
        "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "unsupported"},
        {"unexpected": "field"},
        {"temperature": float("nan")},
        {"seed": True},
    ],
)
def test_invalid_qwen_origin_is_rejected(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    reference = tmp_path / "design.wav"
    reference.write_bytes(b"audio")
    source = local_reference_source(reference)
    source["origin"] = {**_qwen_origin(), **changes}

    with pytest.raises(InvalidEngineVoiceError) as error:
        normalize_voice_source(source)

    assert error.value.code == "voice_source.origin_invalid"


def test_prepared_voice_source_is_hash_pinned_and_not_reference_audio(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "voice.qvoice"
    artifact.write_bytes(b"prepared artifact bytes")
    source = local_prepared_voice_source(
        artifact,
        format="qwen3.qvoice.v2",
        metadata={"model_id": "qwen3-base", "model_revision": "revision-a"},
    )

    resolved = resolve_prepared_voice(source, expected_format="qwen3.qvoice.v2")

    assert resolved.path == artifact.resolve()
    assert resolved.sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert resolved.model_id == "qwen3-base"
    assert voice_source_identity(source) == {
        "kind": "prepared",
        "format": "qwen3.qvoice.v2",
        "sha256": resolved.sha256,
    }
    assert str(artifact) not in repr(voice_source_identity(source))
    prepared_sidecar = tmp_path / "prepared.toml"
    save_voice_source(source, prepared_sidecar)
    assert load_voice_source(prepared_sidecar) == source
    with pytest.raises(InvalidEngineVoiceError) as error:
        resolve_reference_audio(source)
    assert error.value.code == "voice_source.prepared_invalid"

    artifact.write_bytes(b"modified artifact")
    diagnostics = validate_voice_source(source)
    assert diagnostics[0].code == "voice_source.prepared_invalid"
    assert (
        diagnostics[0].details["expected_sha256"]
        != diagnostics[0].details["actual_sha256"]
    )


def test_prepared_source_requires_absolute_normalized_path() -> None:
    for value in ("relative.qvoice", "/tmp/voices/../voice.qvoice"):
        with pytest.raises(InvalidEngineVoiceError) as error:
            normalize_voice_source(
                {
                    "kind": "prepared",
                    "format": "qwen3.qvoice.v2",
                    "value": value,
                    "sha256": "a" * 64,
                }
            )
        assert error.value.code == "voice_source.prepared_invalid"


def test_voice_source_sidecar_roundtrip_is_deterministic(tmp_path: Path) -> None:
    reference = tmp_path / "design.wav"
    reference.write_bytes(b"materialized design bytes")
    source = local_reference_source(reference)
    source["origin"] = _qwen_origin()
    sidecar = tmp_path / "voice-source.toml"

    save_voice_source(source, sidecar)
    first_contents = sidecar.read_text(encoding="utf-8")
    save_voice_source(source, sidecar)

    assert sidecar.read_text(encoding="utf-8") == first_contents
    assert load_voice_source(sidecar) == normalize_voice_source(source).as_mapping()
    assert "[voice_source.origin]" in first_contents
    assert 'format = "voicerender-voice-source"' in first_contents


def test_sidecar_rejects_unknown_fields_and_stale_artifact_hash(tmp_path: Path) -> None:
    reference = tmp_path / "design.wav"
    reference.write_bytes(b"materialized design bytes")
    source = local_reference_source(reference)
    sidecar = tmp_path / "voice-source.toml"
    save_voice_source(source, sidecar)

    contents = sidecar.read_text(encoding="utf-8")
    sidecar.write_text(
        contents.replace("[voice_source]", "unexpected = true\n\n[voice_source]"),
        encoding="utf-8",
    )
    with pytest.raises(InvalidEngineVoiceError) as unknown:
        load_voice_source(sidecar)
    assert unknown.value.code == "voice_source.sidecar_invalid"

    source["sha256"] = "0" * 64
    with pytest.raises(InvalidEngineVoiceError) as stale:
        save_voice_source(source, sidecar)
    assert stale.value.code == "voice_source.reference_changed"


def test_profile_voice_source_binding_accepts_verified_prepared_source(
    tmp_path: Path,
) -> None:
    from voicerender.engines.profiles import validate_voice_source_binding

    artifact = tmp_path / "voice.qvoice"
    artifact.write_bytes(b"prepared bytes")
    source = local_prepared_voice_source(artifact, format="qwen3.qvoice.v2")

    validate_voice_source_binding({"options": {"voice_source": source}})


@pytest.mark.parametrize("source_format", ["wav", "flac", None])
def test_managed_source_format_sidecar_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_format: str | None,
) -> None:
    from voicerender import voice_source_codec

    source: dict[str, object] = {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "value": "catalog:voice-id",
        "sha256": "a" * 64,
        "source_revision": "revision-1",
        "source_repository": "voice-repo",
        "source_path": "voices/reference.wav",
    }
    if source_format is not None:
        source["source_format"] = source_format
    monkeypatch.setattr(voice_source_codec, "validate_voice_source", lambda _source: ())
    sidecar = tmp_path / "managed-voice-source.toml"

    save_voice_source(source, sidecar)
    loaded = load_voice_source(sidecar)

    if source_format is None:
        assert "source_format" not in loaded
    else:
        assert loaded["source_format"] == source_format
        assert f'source_format = "{source_format}"' in sidecar.read_text(
            encoding="utf-8"
        )
    assert loaded["kind"] == "managed_reference"
