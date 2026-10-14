from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import voicefolio

from voicerender import managed_voices
from voicerender.errors import (
    EngineBackendError,
    InvalidEngineOptionError,
    InvalidEngineVoiceError,
)


def _entry() -> voicefolio.VoiceEntry:
    return voicefolio.VoiceEntry(
        ref=voicefolio.VoiceRef("test-voices", "sample"),
        dataset="test-dataset",
        variant="original",
        format="wav",
        url="https://example.invalid/sample.wav",
        size=12,
        sha256="a" * 64,
        source_repository="https://example.invalid/repo",
        source_revision="revision-1",
        source_path="voices/sample.wav",
        license="CC-BY-4.0",
    )


def _source(entry: voicefolio.VoiceEntry | None = None) -> dict[str, Any]:
    entry = entry or _entry()
    return {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "value": str(entry.ref),
        "sha256": entry.sha256,
        "source_revision": entry.source_revision,
        "source_repository": entry.source_repository,
        "source_path": entry.source_path,
        "license": entry.license,
        "dataset": entry.dataset,
        "variant": entry.variant,
    }


def _install_catalog(
    monkeypatch: pytest.MonkeyPatch, entry: voicefolio.VoiceEntry
) -> voicefolio.Catalog:
    catalog = voicefolio.Catalog([entry])
    monkeypatch.setattr(managed_voices.voicefolio, "load_catalog", lambda **_: catalog)
    return catalog


def test_validate_managed_voice_checks_pins_without_fetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry()
    _install_catalog(monkeypatch, entry)

    def unexpected_fetch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("metadata validation must not fetch a WAV")

    monkeypatch.setattr(managed_voices.voicefolio, "resolve_voice_source", unexpected_fetch)

    result = managed_voices.validate_managed_voice(_source(entry))

    assert result is entry


def test_validate_managed_voice_rejects_stale_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry()
    _install_catalog(monkeypatch, entry)
    source = _source(entry)
    source["source_revision"] = "old-revision"

    with pytest.raises(InvalidEngineVoiceError) as error:
        managed_voices.validate_managed_voice(source)

    assert error.value.code == "managed_voice.invalid_source"
    assert error.value.details["ref"] == str(entry.ref)


def test_validate_managed_voice_requires_all_standard_pins() -> None:
    source = _source()
    del source["sha256"]

    with pytest.raises(InvalidEngineVoiceError) as error:
        managed_voices.validate_managed_voice(source)

    assert error.value.code == "managed_voice.invalid_source"
    assert error.value.details["field"] == "sha256"


def test_managed_voice_rejects_unknown_resolver() -> None:
    source = _source()
    source["resolver"] = "arbitrary-plugin"

    with pytest.raises(InvalidEngineOptionError) as error:
        managed_voices.validate_managed_voice(source)

    assert error.value.code == "managed_voice.resolver_unsupported"


def test_resolve_managed_voice_returns_pinned_metadata_and_local_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entry = _entry()
    catalog = _install_catalog(monkeypatch, entry)
    path = tmp_path / "verified.wav"
    calls: list[dict[str, Any]] = []

    def resolve(source: Any, **kwargs: Any) -> voicefolio.ResolvedVoice:
        calls.append(kwargs)
        return voicefolio.ResolvedVoice(entry=entry, path=path)

    monkeypatch.setattr(managed_voices.voicefolio, "resolve_voice_source", resolve)

    resolved = managed_voices.resolve_managed_voice(
        _source(entry), cache_dir=tmp_path, offline=True
    )

    assert resolved == managed_voices.ResolvedManagedVoice(
        resolver="voicefolio",
        ref=str(entry.ref),
        path=path,
        sha256=entry.sha256,
        source_revision=entry.source_revision,
        source_repository=entry.source_repository,
        source_path=entry.source_path,
        license=entry.license,
        dataset=entry.dataset,
        variant=entry.variant,
    )
    assert calls == [{"catalog": catalog, "cache_dir": tmp_path, "offline": True}]


def test_resolve_managed_voice_maps_offline_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry()
    _install_catalog(monkeypatch, entry)

    def offline(*args: Any, **kwargs: Any) -> Any:
        raise voicefolio.OfflineError("not cached")

    monkeypatch.setattr(managed_voices.voicefolio, "resolve_voice_source", offline)

    with pytest.raises(EngineBackendError) as error:
        managed_voices.resolve_managed_voice(_source(entry), offline=True)

    assert error.value.code == "managed_voice.offline_cache_miss"
    assert error.value.details["resolver"] == "voicefolio"



def test_resolve_managed_voice_maps_unknown_reference_to_invalid_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry()
    _install_catalog(monkeypatch, entry)

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise voicefolio.VoiceNotFoundError("missing from catalog")

    monkeypatch.setattr(managed_voices.voicefolio, "resolve_voice_source", missing)

    with pytest.raises(InvalidEngineVoiceError) as error:
        managed_voices.resolve_managed_voice(_source(entry))

    assert error.value.code == "managed_voice.invalid_source"
