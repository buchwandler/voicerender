from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voicerender.engines.base import EngineSelection, RequestMeasure, SpeechRequest
from voicerender.engines.kittensynth import (
    KITTEN_PCM_RANGE_POLICY,
    KittenSynthEngineAdapter,
    KittenSynthEngineSession,
)
from voicerender.errors import EmptySpeechTextError, EngineBackendError


class _FakeVoice:
    def __init__(self, measurement: Any, synthesis_result: Any = None) -> None:
        self.measurement = measurement
        self.synthesis_result = synthesis_result
        self.measure_calls: list[str] = []
        self.synthesize_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def measure_prepared(self, text: str) -> Any:
        self.measure_calls.append(text)
        if isinstance(self.measurement, Exception):
            raise self.measurement
        return self.measurement

    def synthesize_prepared(self, *args: Any, **kwargs: Any) -> Any:
        self.synthesize_calls.append((args, kwargs))
        return self.synthesis_result

def _session(
    voice: _FakeVoice, selection: EngineSelection | None = None
) -> KittenSynthEngineSession:
    if selection is None:
        selection = EngineSelection(
            engine="kitten",
            target_id="nano-0.8-int8",
            language="en-US",
            voice="Jasper",
        )
    return KittenSynthEngineSession(voice, selection)

def _request() -> SpeechRequest:
    return SpeechRequest(id="request-1", text="prepared speech", language="en-US")


def test_measure_maps_exact_amount_without_maximum_and_never_synthesizes() -> None:
    native_metadata = {"frontend": "kitteng2p"}
    voice = _FakeVoice(
        SimpleNamespace(
            token_count=27,
            max_tokens=None,
            fits=None,
            model_id="nano-0.8-int8",
            metadata=native_metadata,
        )
    )

    measured = _session(voice).measure(_request())

    assert measured == RequestMeasure(
        fits=None,
        amount=27,
        maximum=None,
        unit="model_tokens",
        source="kittensynth.measure_prepared",
        details={
            "model_id": "nano-0.8-int8",
            "native_metadata": {"frontend": "kitteng2p"},
        },
    )
    assert voice.measure_calls == ["prepared speech"]
    assert voice.synthesize_calls == []


def test_measure_maps_authoritative_known_maximum() -> None:
    voice = _FakeVoice(
        SimpleNamespace(
            token_count=101,
            max_tokens=100,
            fits=False,
            model_id="nano-0.8-int8",
            metadata=None,
        )
    )

    measured = _session(voice).measure(_request())

    assert measured.amount == 101
    assert measured.maximum == 100
    assert measured.fits is False
    assert measured.unit == "model_tokens"
    assert measured.source == "kittensynth.measure_prepared"
    assert measured.details == {"model_id": "nano-0.8-int8"}
    assert voice.synthesize_calls == []


def test_measure_translates_native_empty_text_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    native_error_type = type("EmptyTextError", (Exception,), {})
    module.EmptyTextError = native_error_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    voice = _FakeVoice(native_error_type("empty input"))

    with pytest.raises(EmptySpeechTextError) as caught:
        _session(voice).measure(_request())

    assert caught.value.code == "synthesis.empty_text"
    assert caught.value.native_error_type == "EmptyTextError"
    assert voice.measure_calls == ["prepared speech"]
    assert voice.synthesize_calls == []



def _synthesis_result(audio: Any) -> SimpleNamespace:
    return SimpleNamespace(
        audio=audio,
        sample_rate=24000,
        metadata={},
        voice="Jasper",
        model_ref="nano-0.8-int8",
        speed=1.0,
    )


def test_synthesize_preserves_in_range_audio_and_reports_policy() -> None:
    audio = np.array([-0.5, 0.0, 0.75], dtype=np.float32)
    voice = _FakeVoice(None, _synthesis_result(audio))

    rendered = _session(voice).synthesize(_request())

    assert rendered.audio.dtype == np.float32
    np.testing.assert_array_equal(rendered.audio, audio)
    assert rendered.warnings == ()
    assert rendered.metadata["pcm_range"] == {
        "policy": KITTEN_PCM_RANGE_POLICY,
        "peak_before_clip": pytest.approx(0.75),
        "clipped_samples": 0,
    }
    assert voice.synthesize_calls[0][0] == ("prepared speech",)


def test_synthesize_clamps_over_range_audio_and_reports_diagnostics() -> None:
    voice = _FakeVoice(
        None,
        _synthesis_result(np.array([-1.2, -0.5, 0.0, 1.1], dtype=np.float32)),
    )

    rendered = _session(voice).synthesize(_request())

    np.testing.assert_array_equal(
        rendered.audio, np.array([-1.0, -0.5, 0.0, 1.0], dtype=np.float32)
    )
    assert np.max(np.abs(rendered.audio)) <= 1.0
    assert rendered.metadata["pcm_range"]["clipped_samples"] == 2
    assert rendered.metadata["pcm_range"]["peak_before_clip"] == pytest.approx(1.2)
    assert "2 sample(s)" in rendered.warnings[0]
    assert "1.200000" in rendered.warnings[0]


@pytest.mark.parametrize("bad_sample", [np.nan, np.inf])
def test_synthesize_does_not_hide_non_finite_audio(bad_sample: float) -> None:
    voice = _FakeVoice(None, _synthesis_result(np.array([0.0, bad_sample])))

    with pytest.raises(EngineBackendError, match="finite"):
        _session(voice).synthesize(_request())


def test_synthesize_does_not_flatten_malformed_channels() -> None:
    audio = np.array([[0.0, 1.1], [0.2, 0.3]], dtype=np.float32)
    voice = _FakeVoice(None, _synthesis_result(audio))

    with pytest.raises(EngineBackendError, match="mono"):
        _session(voice).synthesize(_request())


def test_canonical_identity_contains_versioned_pcm_range_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    module.runtime_identity = lambda: {"runtime": "fake"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    selection = EngineSelection(
        engine="kitten",
        target_id="nano-0.8-int8",
        language="en-US",
        voice="Jasper",
        metadata={"source_revision": "rev-1"},
    )

    identity = KittenSynthEngineAdapter().canonical_synthesis_identity(selection)

    assert identity["pcm_range_policy"] == "kitten-native-wav-clamp.v1"
