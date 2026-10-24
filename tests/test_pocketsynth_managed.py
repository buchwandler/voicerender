from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import voicefolio

import voicerender.engines.pocketsynth as pocketsynth_adapter
from voicerender import managed_voices
from voicerender.engines.base import PCM_RANGE_POLICY, EngineSelection, SpeechRequest
from voicerender.engines.pocketsynth import (
    PocketSynthEngineAdapter,
    PocketSynthEngineSession,
    _manager,
    _voice_identity,
    _voice_source,
)
from voicerender.engines.profiles import reject_reference_binding
from voicerender.engines.selection import EngineRequest
from voicerender.errors import EngineBackendError, ProfileError
from voicerender.voice_sources import (
    ResolvedReferenceAudio,
    local_reference_source,
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
) -> None:
    catalog = voicefolio.Catalog([entry])
    monkeypatch.setattr(managed_voices.voicefolio, "load_catalog", lambda **_: catalog)


def _request(source: dict[str, Any], **options: Any) -> EngineRequest:
    request_options = {**options, "voice_source": source}
    return EngineRequest(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        options=request_options,
    )


def test_pocket_resolve_preserves_voicefolio_pins_without_fetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry()
    source = _source(entry)
    _install_catalog(monkeypatch, entry)

    def unexpected_fetch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("adapter metadata resolution must not fetch a WAV")

    monkeypatch.setattr(
        managed_voices.voicefolio, "resolve_voice_source", unexpected_fetch
    )

    selection, diagnostics = PocketSynthEngineAdapter().resolve(_request(source))

    assert diagnostics == ()
    assert selection.metadata["voice_source"] == source


def test_pocket_resolve_reports_stale_pins_without_rewriting_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry()
    source = _source(entry)
    source["sha256"] = "b" * 64
    _install_catalog(monkeypatch, entry)

    selection, diagnostics = PocketSynthEngineAdapter().resolve(_request(source))

    assert [item.code for item in diagnostics] == ["managed_voice.invalid_source"]
    assert selection.metadata["voice_source"] == source


def test_voice_source_preserves_resolver_and_has_limited_legacy_shim() -> None:
    entry = _entry()
    source = _source(entry)
    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        metadata={"voice_source": source},
    )

    normalized = _voice_source(selection)

    assert normalized is not None
    assert normalized["resolver"] == "voicefolio"
    assert normalized["source_revision"] == entry.source_revision

    legacy = dict(source, value="kyutai-tts-voices:sample")
    legacy.pop("resolver")
    with pytest.warns(DeprecationWarning, match="resolver='voicefolio'"):
        normalized_legacy = _voice_source(
            EngineSelection(
                engine="pocket",
                target_id="bundle-test",
                language="en-US",
                metadata={"voice_source": legacy},
            )
        )
    assert normalized_legacy is not None
    assert normalized_legacy["resolver"] == "voicefolio"


def test_pocket_profile_fragment_retains_authored_managed_source() -> None:
    source = _source()
    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        metadata={"voice_source": source},
    )

    fragment = PocketSynthEngineAdapter().renderprofile_fragment(selection)

    assert fragment.binding["options"]["voice_source"] == source
    reject_reference_binding(fragment.binding)


def test_shared_profile_accepts_voicefolio_but_rejects_incomplete_managed_source() -> (
    None
):
    source = _source()

    reject_reference_binding({"options": {"voice_source": source}})

    incomplete = dict(source)
    del incomplete["sha256"]
    with pytest.raises(ProfileError, match="lowercase 64-character SHA-256 digest"):
        reject_reference_binding({"options": {"voice_source": incomplete}})


def test_pocket_profile_fragment_accepts_local_reference(tmp_path: Path) -> None:
    reference = tmp_path / "local.wav"
    reference.write_bytes(b"local reference")
    source = local_reference_source(reference)
    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        metadata={"voice_source": source},
    )

    fragment = PocketSynthEngineAdapter().renderprofile_fragment(selection)

    assert fragment.binding["options"]["voice_source"] == source
    reject_reference_binding(fragment.binding)


def test_pocket_uses_copied_qwen_reference_without_qwen3synth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference = tmp_path / "copied-designed.wav"
    reference.write_bytes(b"copied materialized voice")
    source = local_reference_source(reference)
    source["origin"] = {
        "schema": "voicerender.voice-origin.qwen3-design.v1",
        "generator": "qwen3",
        "target_id": "voice-design-1.7b",
        "model_revision": "revision-1",
        "qwen3synth_version": "0.1.0",
        "onnxvoice_version": "0.1.0",
        "prompt": "A warm speaking voice.",
        "reference_text": "This exact generated sample.",
        "language": "en",
        "quality": "int4",
        "provider": "cpu",
        "temperature": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_new_tokens": 128,
        "seed": 11,
        "model_id": "wavekat/voice-design-1.7b",
    }
    original_import = builtins.__import__

    def reject_qwen_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", 1)[0] == "qwen3synth":
            raise AssertionError("PocketSynth must not import Qwen3Synth")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_qwen_import)
    monkeypatch.setitem(sys.modules, "qwen3synth", None)
    resolved = ResolvedReferenceAudio(
        kind="reference",
        path=reference.resolve(),
        sha256=source["sha256"],
    )
    monkeypatch.setattr(
        pocketsynth_adapter, "resolve_reference_audio", lambda _: resolved
    )
    prepared_inputs: list[Path] = []
    prepared_voice = object()

    class Runtime:
        def prepare_voice(self, path: Path) -> object:
            prepared_inputs.append(path)
            return prepared_voice

    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        metadata={"voice_source": source},
    )
    fragment = PocketSynthEngineAdapter().renderprofile_fragment(selection)
    assert fragment.binding["options"]["voice_source"] == source

    request = SpeechRequest(
        id="request-1",
        text="hello",
        language="en-US",
        options={"voice_source": source},
    )
    session = PocketSynthEngineSession(Runtime(), selection, None, None)
    assert session._voice(request) is prepared_voice
    assert prepared_inputs == [reference.resolve()]


def test_managed_execution_resolves_once_passes_path_and_reuses_prepared_voice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _source()
    voice_path = tmp_path / "verified.wav"
    resolved = managed_voices.ResolvedManagedVoice(
        resolver="voicefolio",
        ref=source["value"],
        path=voice_path,
        source_sha256=source["sha256"],
        resolved_sha256=source["sha256"],
        source_format="wav",
        source_revision=source["source_revision"],
        source_repository=source["source_repository"],
        source_path=source["source_path"],
        license=source["license"],
        dataset=source["dataset"],
        variant=source["variant"],
    )
    resolve_calls: list[dict[str, Any]] = []
    prepared_paths: list[Path] = []
    prepared = object()

    def resolve(managed_source: Any, **kwargs: Any) -> Any:
        resolve_calls.append({"source": managed_source, **kwargs})
        return resolved

    class Runtime:
        def prepare_voice(self, path: Path) -> object:
            prepared_paths.append(path)
            return prepared

    monkeypatch.setattr(pocketsynth_adapter, "resolve_reference_audio", resolve)
    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        options={"cache_dir": tmp_path, "voice_catalog_path": tmp_path / "voices.json"},
        metadata={"voice_source": source},
        offline=True,
    )
    session = PocketSynthEngineSession(Runtime(), selection, None, None)
    request = SpeechRequest(
        id="request-1",
        text="hello",
        language="en-US",
        options={"voice_source": source},
    )

    first = session._voice(request)
    second = session._voice(request)

    assert first is prepared and second is prepared
    assert prepared_paths == [voice_path]
    assert resolve_calls == [
        {
            "source": source,
            "cache_dir": tmp_path,
            "offline": True,
            "catalog_paths": tmp_path / "voices.json",
        }
    ]
    assert next(iter(session._voices)) == (
        "managed_reference",
        "voicefolio",
        source["value"],
        source["sha256"],
        source["source_revision"],
    )


def test_managed_execution_checks_resolver_provenance_before_prepare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _source()
    resolved = managed_voices.ResolvedManagedVoice(
        resolver="voicefolio",
        ref=source["value"],
        path=tmp_path / "verified.wav",
        source_sha256=source["sha256"],
        resolved_sha256=source["sha256"],
        source_format="wav",
        source_revision=source["source_revision"],
        source_repository=source["source_repository"],
        source_path="different/path.wav",
    )

    monkeypatch.setattr(
        pocketsynth_adapter,
        "resolve_reference_audio",
        lambda *args, **kwargs: resolved,
    )

    class Runtime:
        def prepare_voice(self, path: Path) -> object:
            raise AssertionError("mismatched provenance must be rejected first")

    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        metadata={"voice_source": source},
    )
    session = PocketSynthEngineSession(Runtime(), selection, None, None)
    request = SpeechRequest(id="request-1", text="hello", language="en-US")

    with pytest.raises(EngineBackendError) as error:
        session._voice(request)

    assert error.value.code == "pocket.managed_reference_changed"
    assert error.value.details["resolver"] == "voicefolio"


def test_managed_identity_keeps_resolver_and_excludes_local_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        options={"voice_catalog_path": "/machine/voices.json"},
        metadata={"voice_source": source},
    )
    monkeypatch.setitem(
        sys.modules,
        "pocketsynth",
        SimpleNamespace(runtime_identity=lambda: {"runtime": "test"}),
    )

    identity = _voice_identity(selection)
    canonical = PocketSynthEngineAdapter().canonical_synthesis_identity(selection)

    assert identity == {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "ref": source["value"],
        "sha256": source["sha256"],
        "source_revision": source["source_revision"],
    }
    assert canonical["voice"] == identity
    assert "/machine/voices.json" not in repr(canonical)


def test_catalog_options_remain_separate_and_voice_browsing_uses_voicefolio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entry = _entry()
    catalog = voicefolio.Catalog([entry])
    voice_catalog_path = tmp_path / "voice-catalog.json"
    model_catalog_path = tmp_path / "pocket-model-catalog.json"
    voicefolio_paths: list[tuple[Any, ...]] = []

    def load_catalog(*, extra_paths: Any = ()) -> voicefolio.Catalog:
        voicefolio_paths.append(tuple(extra_paths))
        return catalog

    monkeypatch.setattr(managed_voices.voicefolio, "load_catalog", load_catalog)
    source = _source(entry)
    selection, diagnostics = PocketSynthEngineAdapter().resolve(
        _request(
            source,
            voice_catalog_path=voice_catalog_path,
            catalog_path=model_catalog_path,
        )
    )
    assert diagnostics == ()
    assert selection.options["voice_catalog_path"] == voice_catalog_path
    assert selection.options["catalog_path"] == model_catalog_path

    class BundleManager:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    bundle_calls: list[dict[str, Any]] = []

    def discover_bundles(**kwargs: Any) -> tuple[Any, ...]:
        bundle_calls.append(kwargs)
        return ()

    monkeypatch.setitem(
        sys.modules,
        "pocketsynth",
        SimpleNamespace(
            BundleAssetManager=BundleManager,
            discover_bundles=discover_bundles,
        ),
    )
    manager = _manager(
        options={
            "cache_dir": tmp_path,
            "catalog_path": model_catalog_path,
            "voice_catalog_path": voice_catalog_path,
        },
        offline=False,
    )
    PocketSynthEngineAdapter().list_voice_prompts(
        dataset=entry.dataset, voice_catalog_path=voice_catalog_path
    )
    assert manager.kwargs["catalog_path"] == model_catalog_path
    assert "voice_catalog_path" not in manager.kwargs
    assert voicefolio_paths == [(voice_catalog_path,), (voice_catalog_path,)]

    from voicerender.engines.pocketsynth import _discover_bundles

    _discover_bundles(
        options={
            "cache_dir": tmp_path,
            "catalog_path": model_catalog_path,
            "voice_catalog_path": voice_catalog_path,
        },
        language="en",
        offline=False,
        refresh=False,
    )
    assert bundle_calls[0]["catalog_path"] == model_catalog_path
    assert "voice_catalog_path" not in bundle_calls[0]


def test_offline_cached_voicefolio_asset_renders_with_managed_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _source()
    voice_path = tmp_path / "cached-verified.wav"
    resolved = managed_voices.ResolvedManagedVoice(
        resolver="voicefolio",
        ref=source["value"],
        path=voice_path,
        source_sha256=source["sha256"],
        resolved_sha256=source["sha256"],
        source_format="wav",
        source_revision=source["source_revision"],
        source_repository=source["source_repository"],
        source_path=source["source_path"],
    )
    calls: list[dict[str, Any]] = []
    prepared = object()

    def resolve(managed_source: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return resolved

    class Runtime:
        def prepare_voice(self, path: Path) -> object:
            assert path == voice_path
            return prepared

        def synthesize(self, native: Any, *, voice: Any, config: Any, voice_level: Any):
            assert voice is prepared
            return SimpleNamespace(
                id=native.id,
                audio=np.ones(8, dtype=np.float32) / 4,
                sample_rate=24000,
                warnings=(),
                metadata={"token_count": 1},
            )

    monkeypatch.setattr(pocketsynth_adapter, "resolve_reference_audio", resolve)
    monkeypatch.setitem(
        sys.modules,
        "pocketsynth",
        SimpleNamespace(SynthesisRequest=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        options={"voice_catalog_path": tmp_path / "voices.json"},
        metadata={"voice_source": source},
        offline=True,
    )
    session = PocketSynthEngineSession(
        Runtime(),
        selection,
        generation=object(),
        voice_level=SimpleNamespace(mode="off"),
    )
    request = SpeechRequest(
        id="render-1",
        text="hello",
        language="en-US",
        options={"voice_source": source},
    )

    rendered = session.synthesize(request)

    assert rendered.sample_rate == 24000
    assert rendered.metadata["voice"] == {
        "kind": "managed_reference",
        "resolver": "voicefolio",
        "ref": source["value"],
        "sha256": source["sha256"],
        "source_revision": source["source_revision"],
    }
    assert calls == [
        {
            "cache_dir": None,
            "offline": True,
            "catalog_paths": tmp_path / "voices.json",
        }
    ]


def test_offline_voicefolio_cache_miss_keeps_precise_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    error = EngineBackendError(
        "Voicefolio has no cached WAV while offline.",
        code="managed_voice.offline_cache_miss",
        details={"resolver": "voicefolio", "ref": source["value"]},
    )

    def fail_resolution(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(pocketsynth_adapter, "resolve_reference_audio", fail_resolution)

    class Runtime:
        def prepare_voice(self, path: Path) -> object:
            raise AssertionError("a Voicefolio cache miss must fail before PocketSynth")

    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        metadata={"voice_source": source},
        offline=True,
    )
    session = PocketSynthEngineSession(Runtime(), selection, None, None)
    request = SpeechRequest(id="request-1", text="hello", language="en-US")

    with pytest.raises(EngineBackendError) as caught:
        session._voice(request)

    assert caught.value.code == "managed_voice.offline_cache_miss"
    assert caught.value.details["resolver"] == "voicefolio"


def _synthesize_pocket_pcm(
    monkeypatch: pytest.MonkeyPatch,
    audio: Any,
    *,
    warnings: tuple[str, ...] = (),
    result_id: str = "request-1",
    sample_rate: int = 24000,
) -> Any:
    class Runtime:
        def prepare_voice(self, voice: str) -> str:
            return voice

        def synthesize(
            self,
            native: Any,
            *,
            voice: Any,
            config: Any,
            voice_level: Any,
        ) -> Any:
            return SimpleNamespace(
                id=result_id,
                audio=audio,
                sample_rate=sample_rate,
                warnings=warnings,
                metadata={"token_count": 1},
            )

    monkeypatch.setitem(
        sys.modules,
        "pocketsynth",
        SimpleNamespace(
            SynthesisRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        voice="default",
    )
    session = PocketSynthEngineSession(
        Runtime(),
        selection,
        generation=None,
        voice_level=SimpleNamespace(mode="off"),
    )
    request = SpeechRequest(
        id="request-1",
        text="hello",
        language="en-US",
        voice="default",
    )
    return session.synthesize(request)


def test_pocket_pcm_in_range_audio_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = np.array([-0.75, 0.0, 1.0], dtype=np.float32)

    rendered = _synthesize_pocket_pcm(monkeypatch, audio)

    assert np.array_equal(rendered.audio, audio)
    assert rendered.audio.dtype == np.float32
    assert rendered.warnings == ()
    assert rendered.metadata["pcm_range"] == {
        "policy": PCM_RANGE_POLICY,
        "peak_before_clip": 1.0,
        "clipped_samples": 0,
    }


def test_pocket_pcm_real_failure_peak_is_clamped_with_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = _synthesize_pocket_pcm(
        monkeypatch,
        np.array([0.0, -0.25, 1.048518], dtype=np.float32),
    )

    assert np.max(np.abs(rendered.audio)) == 1.0
    assert rendered.audio[-1] == 1.0
    assert rendered.metadata["pcm_range"]["peak_before_clip"] == pytest.approx(1.048518)
    assert rendered.metadata["pcm_range"]["clipped_samples"] == 1
    assert len(rendered.warnings) == 1
    assert "PocketSynth returned audio above full scale" in rendered.warnings[0]
    assert "clamped 1 sample(s)" in rendered.warnings[0]
    assert "1.048518" in rendered.warnings[0]


def test_pocket_pcm_clamps_negative_and_multiple_overshoots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = _synthesize_pocket_pcm(
        monkeypatch,
        np.array([-1.2, 0.0, 1.048518, -1.1], dtype=np.float32),
    )

    np.testing.assert_array_equal(rendered.audio, [-1.0, 0.0, 1.0, -1.0])
    assert rendered.metadata["pcm_range"]["clipped_samples"] == 3


def test_pocket_pcm_clamps_values_inside_validator_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = _synthesize_pocket_pcm(
        monkeypatch,
        np.array([0.0, 1.000005], dtype=np.float32),
    )

    assert rendered.audio[-1] == 1.0
    assert rendered.metadata["pcm_range"]["peak_before_clip"] == pytest.approx(1.000005)
    assert rendered.metadata["pcm_range"]["clipped_samples"] == 1
    assert len(rendered.warnings) == 1


@pytest.mark.parametrize("sample", [np.nan, np.inf], ids=["nan", "infinity"])
def test_pocket_pcm_non_finite_audio_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
    sample: float,
) -> None:
    with pytest.raises(EngineBackendError, match="finite samples"):
        _synthesize_pocket_pcm(
            monkeypatch,
            np.array([0.0, sample], dtype=np.float32),
        )


def test_pocket_pcm_malformed_multichannel_audio_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(EngineBackendError, match="audio must be mono"):
        _synthesize_pocket_pcm(
            monkeypatch,
            np.array([[0.0, 1.1], [0.2, 0.3]], dtype=np.float32),
        )


def test_pocket_pcm_preserves_native_warning_before_clamp_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = _synthesize_pocket_pcm(
        monkeypatch,
        np.array([0.0, 1.1], dtype=np.float32),
        warnings=("native Pocket diagnostic",),
    )

    assert rendered.warnings[0] == "native Pocket diagnostic"
    assert "PocketSynth returned audio above full scale" in rendered.warnings[1]


@pytest.mark.parametrize(
    ("audio", "result_id", "sample_rate", "message"),
    [
        (np.array([], dtype=np.float32), "request-1", 24000, "non-empty"),
        (
            np.array([0.5], dtype=np.float32),
            "other-id",
            24000,
            "does not match request id",
        ),
        (
            np.array([0.5], dtype=np.float32),
            "request-1",
            0,
            "sample rate must be a positive integer",
        ),
    ],
    ids=["empty-audio", "bad-id", "bad-sample-rate"],
)
def test_pocket_pcm_strict_validation_still_rejects_invalid_results(
    monkeypatch: pytest.MonkeyPatch,
    audio: np.ndarray,
    result_id: str,
    sample_rate: int,
    message: str,
) -> None:
    with pytest.raises(EngineBackendError, match=message):
        _synthesize_pocket_pcm(
            monkeypatch,
            audio,
            result_id=result_id,
            sample_rate=sample_rate,
        )


def test_pocket_canonical_identity_includes_pcm_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pocketsynth",
        SimpleNamespace(runtime_identity=lambda: {"runtime": "test"}),
    )
    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        voice="default",
    )

    identity = PocketSynthEngineAdapter().canonical_synthesis_identity(selection)

    assert identity["pcm_range_policy"] == PCM_RANGE_POLICY


def test_pocket_request_measurement_remains_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pocketsynth",
        SimpleNamespace(SynthesisRequest=lambda **kwargs: SimpleNamespace(**kwargs)),
    )

    class Runtime:
        def measure_request(self, native: Any) -> Any:
            assert native.text == "hello"
            return SimpleNamespace(fits=True, amount=11, maximum=50)

    selection = EngineSelection(
        engine="pocket",
        target_id="bundle-test",
        language="en-US",
        voice="default",
    )
    session = PocketSynthEngineSession(Runtime(), selection, None, None)
    request = SpeechRequest(
        id="request-1",
        text="hello",
        language="en-US",
        voice="default",
    )

    measured = session.measure(request)

    assert measured.fits is True
    assert measured.amount == 11
    assert measured.maximum == 50
    assert measured.unit == "model_tokens"
    assert measured.source == "pocketsynth.measure_request"
