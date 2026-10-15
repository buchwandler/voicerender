"""Prepared-text request measurement tests for InflectSynth."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from voicerender.engines.base import EngineSelection, RequestMeasure, SpeechRequest
from voicerender.engines.inflectsynth import (
    InflectSynthEngineAdapter,
    InflectSynthEngineSession,
    _target_from_model,
)
from voicerender.errors import SpeechRequestTooLongError


class _Voice:
    def __init__(self, measured: Any = None, error: Exception | None = None) -> None:
        self.measured = measured
        self.error = error
        self.measure_calls: list[str] = []
        self.synthesis_calls = 0

    def measure_prepared(self, text: str) -> Any:
        self.measure_calls.append(text)
        if self.error is not None:
            raise self.error
        return self.measured

    def synthesize_prepared(self, *args: Any, **kwargs: Any) -> Any:
        self.synthesis_calls += 1
        raise AssertionError("measurement must not synthesize")


def _session(voice: Any) -> InflectSynthEngineSession:
    return InflectSynthEngineSession(
        voice,
        EngineSelection(engine="inflect", target_id="managed-model", language="en-US"),
        config=None,
    )


def _request() -> SpeechRequest:
    return SpeechRequest(id="request-1", text="hello world", language="en-US")


def test_legacy_measurement_fields_are_normalized_without_synthesis() -> None:
    voice = _Voice(
        SimpleNamespace(
            amount=12,
            maximum=20,
            fits=True,
            model_id="managed-model",
        )
    )

    measured = _session(voice).measure(_request())

    assert measured == RequestMeasure(
        fits=True,
        amount=12,
        maximum=20,
        unit="model_tokens",
        source="inflectsynth.measure_prepared",
        details={"model_id": "managed-model"},
    )
    assert voice.measure_calls == ["hello world"]
    assert voice.synthesis_calls == 0


def test_common_measurement_fields_preserve_overflow_and_support_mapping() -> None:
    voice = _Voice(
        {
            "token_count": 21,
            "max_tokens": 20,
            "fits": False,
            "model_id": "managed-model",
        }
    )

    measured = _session(voice).measure(_request())

    assert measured.fits is False
    assert measured.amount == 21
    assert measured.maximum == 20
    assert measured.unit == "model_tokens"
    assert measured.source == "inflectsynth.measure_prepared"


def test_exact_amount_survives_unknown_maximum_and_fit() -> None:
    measured = _session(
        _Voice(SimpleNamespace(amount=80, maximum=None, fits=None))
    ).measure(_request())

    assert measured.amount == 80
    assert measured.maximum is None
    assert measured.fits is None
    assert measured.unit == "model_tokens"
    assert measured.source == "inflectsynth.measure_prepared"


@pytest.mark.parametrize(
    ("result", "invalid_field"),
    [
        (SimpleNamespace(amount="80", maximum=None, fits=None), "amount"),
        (SimpleNamespace(amount=True, maximum=None, fits=None), "amount"),
        (SimpleNamespace(amount=80, maximum=0, fits=None), "maximum"),
        (SimpleNamespace(amount=80, maximum=100, fits="yes"), "fits"),
    ],
)
def test_malformed_measurement_fields_are_reported(
    result: Any, invalid_field: str
) -> None:
    measured = _session(_Voice(result)).measure(_request())

    assert measured.source == "inflectsynth.invalid_measurement"
    assert invalid_field in measured.details["invalid_fields"]


def test_missing_measurement_method_uses_no_measurement_source() -> None:
    measured = _session(object()).measure(_request())

    assert measured == RequestMeasure(
        fits=None,
        amount=None,
        maximum=None,
        unit="unknown",
        source="inflectsynth.no_capacity_measurement",
    )


def test_typed_too_long_exception_is_translated_without_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeTooLong(Exception):
        pass

    fake_module = ModuleType("inflectsynth")
    fake_module.SynthesisInputTooLongError = NativeTooLong
    monkeypatch.setitem(sys.modules, "inflectsynth", fake_module)
    error = NativeTooLong("request exceeds model capacity")
    error.token_count = 101
    error.max_tokens = 100
    voice = _Voice(error=error)

    with pytest.raises(SpeechRequestTooLongError) as raised:
        _session(voice).measure(_request())

    assert raised.value.amount == 101
    assert raised.value.maximum == 100
    assert raised.value.unit == "model_tokens"
    assert voice.synthesis_calls == 0


def test_runtime_capacity_must_match_discovered_managed_maximum() -> None:
    voice = _Voice(SimpleNamespace(amount=12, maximum=90, fits=True))
    selection = EngineSelection(
        engine="inflect",
        target_id="managed-model",
        language="en-US",
        metadata={"max_input_tokens": 100, "capacity_policy_revision": "budget.v1"},
    )

    measured = InflectSynthEngineSession(voice, selection, config=None).measure(
        _request()
    )

    assert measured.source == "inflectsynth.capacity_metadata_mismatch"
    assert measured.amount == 12
    assert measured.maximum == 90
    assert measured.details == {
        "discovered_max_input_tokens": 100,
        "runtime_maximum": 90,
    }
    assert voice.synthesis_calls == 0


def test_measurement_provenance_includes_selection_policy_revision() -> None:
    voice = _Voice(SimpleNamespace(amount=12, maximum=100, fits=True))
    selection = EngineSelection(
        engine="inflect",
        target_id="managed-model",
        language="en-US",
        metadata={"max_input_tokens": 100, "capacity_policy_revision": "budget.v1"},
    )

    measured = InflectSynthEngineSession(voice, selection, config=None).measure(
        _request()
    )

    assert measured.details["capacity_policy_revision"] == "budget.v1"
    assert measured.amount == 12
    assert measured.maximum == 100
    assert measured.fits is True


def test_discovered_target_normalizes_capacity_policy_identity() -> None:
    target = _target_from_model(
        SimpleNamespace(
            id="managed-model",
            language="en-US",
            voices=(),
            metadata={
                "max_input_tokens": 128,
                "capacity_revision": "budget.v2",
            },
            sample_rate=24000,
            runtime_available=True,
        )
    )

    assert target.metadata["max_input_tokens"] == 128
    assert target.metadata["capacity_policy_revision"] == "budget.v2"


def test_renderprofile_carries_discovered_capacity_and_policy_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = InflectSynthEngineAdapter()
    monkeypatch.setattr(
        adapter,
        "target_metadata",
        lambda selection: {
            "max_input_tokens": 128,
            "capacity_policy_revision": "budget.v2",
        },
    )
    selection = EngineSelection(
        engine="inflect",
        target_id="managed-model",
        language="en-US",
    )

    fragment = adapter.renderprofile_fragment(selection)

    assert fragment.renderer_capacity is not None
    assert fragment.renderer_capacity.maximum == 128
    assert fragment.binding["renderer_metadata"]["max_input_tokens"] == 128
    assert (
        fragment.binding["renderer_metadata"]["capacity_policy_revision"] == "budget.v2"
    )


def test_canonical_identity_includes_capacity_policy_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("inflectsynth")
    fake_module.DEFAULT_VOICE = "default"
    fake_module.runtime_identity = lambda: {"runtime": "runtime.v1"}
    fake_module.capacity_api_contract = lambda: {"policy_revision": "budget.v2"}
    monkeypatch.setitem(sys.modules, "inflectsynth", fake_module)
    selection = EngineSelection(
        engine="inflect",
        target_id="managed-model",
        language="en-US",
        metadata={"capacity_policy_revision": "budget.v2"},
    )

    identity = InflectSynthEngineAdapter().canonical_synthesis_identity(selection)

    assert identity["capacity_policy_revision"] == "budget.v2"
