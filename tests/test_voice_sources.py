from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import pytest

from voicerender import voice_sources
from voicerender.errors import InvalidEngineOptionError, InvalidEngineVoiceError


def _managed_source() -> dict[str, Any]:
    return {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "value": "catalog:voice-id",
        "sha256": "a" * 64,
        "source_revision": "revision-1",
        "source_repository": "voice-repo",
        "source_path": "voices/reference.wav",
        "license": "CC0-1.0",
        "dataset": "voice-donations",
        "variant": "original",
    }


def test_local_reference_source_resolves_absolute_path_and_computes_digest(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "speaker.wav"
    audio = b"reference wave bytes"
    source_path.write_bytes(audio)

    source = voice_sources.local_reference_source(source_path)

    assert source == {
        "kind": "reference",
        "value": str(source_path.resolve()),
        "sha256": hashlib.sha256(audio).hexdigest(),
    }


def test_local_reference_identity_excludes_path(tmp_path: Path) -> None:
    audio = b"same reference content"
    first = tmp_path / "first.wav"
    second = tmp_path / "elsewhere" / "second.wav"
    second.parent.mkdir()
    first.write_bytes(audio)
    second.write_bytes(audio)

    first_source = voice_sources.local_reference_source(first)
    second_source = voice_sources.local_reference_source(second)

    assert voice_sources.voice_source_identity(first_source) == {
        "kind": "reference",
        "sha256": hashlib.sha256(audio).hexdigest(),
    }
    assert voice_sources.voice_source_identity(first_source) == (
        voice_sources.voice_source_identity(second_source)
    )
    assert str(first) not in repr(voice_sources.voice_source_identity(first_source))


def test_changed_local_reference_is_rejected_with_hash_details(tmp_path: Path) -> None:
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"before")
    source = voice_sources.local_reference_source(reference)
    reference.write_bytes(b"after")

    diagnostics = voice_sources.validate_voice_source(source)

    assert len(diagnostics) == 1
    assert diagnostics[0].code == "voice_source.reference_changed"
    assert diagnostics[0].details["expected_sha256"] == source["sha256"]
    assert (
        diagnostics[0].details["actual_sha256"] == hashlib.sha256(b"after").hexdigest()
    )
    assert diagnostics[0].as_dict()["details"]["path"] == str(reference)
    with pytest.raises(InvalidEngineVoiceError) as error:
        voice_sources.resolve_reference_audio(source)
    assert error.value.code == "voice_source.reference_changed"
    assert error.value.details["expected_sha256"] == source["sha256"]


def test_malformed_digest_and_relative_local_path_are_rejected() -> None:
    with pytest.raises(InvalidEngineVoiceError) as digest_error:
        voice_sources.normalize_voice_source(
            {"kind": "reference", "value": "/tmp/speaker.wav", "sha256": "ABC"}
        )
    assert digest_error.value.code == "voice_source.invalid"

    with pytest.raises(InvalidEngineVoiceError) as path_error:
        voice_sources.normalize_voice_source(
            {"kind": "reference", "value": "speaker.wav", "sha256": "a" * 64}
        )
    assert path_error.value.code == "voice_source.invalid"


def test_local_reference_constructor_requires_existing_regular_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(InvalidEngineVoiceError) as missing:
        voice_sources.local_reference_source(tmp_path / "missing.wav")
    assert missing.value.code == "voice_source.reference_missing"

    with pytest.raises(InvalidEngineVoiceError) as not_file:
        voice_sources.local_reference_source(tmp_path)
    assert not_file.value.code == "voice_source.reference_not_file"

    with pytest.raises(InvalidEngineVoiceError) as remote:
        voice_sources.local_reference_source("https://example.invalid/speaker.wav")
    assert remote.value.code == "voice_source.invalid"


def test_managed_validation_delegates_to_voicefolio_without_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _managed_source()
    calls: list[dict[str, Any]] = []

    def validate(value: Any, **kwargs: Any) -> object:
        calls.append({"source": value, **kwargs})
        return object()

    monkeypatch.setattr(voice_sources, "validate_managed_voice", validate)
    diagnostics = voice_sources.validate_voice_source(
        source, catalog_paths="voice-catalog.json"
    )

    assert diagnostics == ()
    assert calls == [{"source": source, "catalog_paths": "voice-catalog.json"}]


def test_managed_resolution_preserves_pins_and_identity_omits_cache_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _managed_source()
    resolved_path = tmp_path / "cache" / "verified.wav"
    resolved_path.parent.mkdir()
    resolved_path.write_bytes(b"verified audio")
    calls: list[dict[str, Any]] = []

    class Resolved:
        resolver = "voicefolio"
        ref = source["value"]
        path = resolved_path
        sha256 = source["sha256"]
        source_revision = source["source_revision"]
        source_repository = source["source_repository"]
        source_path = source["source_path"]

    def resolve(value: Any, **kwargs: Any) -> Resolved:
        calls.append({"source": value, **kwargs})
        return Resolved()

    monkeypatch.setattr(voice_sources, "resolve_managed_voice", resolve)
    resolved = voice_sources.resolve_reference_audio(
        source, cache_dir=tmp_path / "cache", offline=True
    )

    assert resolved.kind == "managed_reference"
    assert resolved.path == resolved_path
    assert resolved.sha256 == source["sha256"]
    assert resolved.ref == source["value"]
    assert resolved.source_revision == source["source_revision"]
    assert calls[0]["cache_dir"] == tmp_path / "cache"
    identity = voice_sources.voice_source_identity(source)
    assert identity == {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "ref": source["value"],
        "sha256": source["sha256"],
        "source_revision": source["source_revision"],
    }
    assert str(resolved_path) not in repr(identity)


def test_unsupported_resolver_is_rejected_and_legacy_shim_is_limited() -> None:
    unsupported = _managed_source()
    unsupported["resolver"] = "other"
    with pytest.raises(InvalidEngineOptionError) as error:
        voice_sources.normalize_voice_source(unsupported)
    assert error.value.code == "voice_source.resolver_unsupported"

    historical = _managed_source()
    historical.pop("resolver")
    historical["value"] = "kyutai-tts-voices:dataset/voice"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        normalized = voice_sources.normalize_voice_source(historical)
    assert normalized.resolver == "voicefolio"
    assert len(captured) == 1
    assert issubclass(captured[0].category, DeprecationWarning)

    historical["value"] = "new-catalog:voice"
    with pytest.raises(InvalidEngineVoiceError) as error:
        voice_sources.normalize_voice_source(historical)
    assert error.value.code == "managed_voice.resolver_missing"
