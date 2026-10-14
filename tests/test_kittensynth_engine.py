from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from voicerender.engines.base import EngineSelection, RequestMeasure, SpeechRequest
from voicerender.engines.kittensynth import KittenSynthEngineSession
from voicerender.errors import EmptySpeechTextError


class _FakeVoice:
    def __init__(self, measurement: Any) -> None:
        self.measurement = measurement
        self.measure_calls: list[str] = []
        self.synthesize_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def measure_prepared(self, text: str) -> Any:
        self.measure_calls.append(text)
        if isinstance(self.measurement, Exception):
            raise self.measurement
        return self.measurement

    def synthesize_prepared(self, *args: Any, **kwargs: Any) -> None:
        self.synthesize_calls.append((args, kwargs))


def _session(voice: _FakeVoice) -> KittenSynthEngineSession:
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
