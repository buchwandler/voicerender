from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from renderplan import CapacityConstraint

from voicerender.engines.base import EngineSelection, RequestMeasure, SpeechRequest
from voicerender.engines.kittensynth import (
    KITTEN_PCM_RANGE_POLICY,
    KittenSynthEngineAdapter,
    KittenSynthEngineSession,
    _target_from_model,
)
from voicerender.error_diagnostics import exception_diagnostic
from voicerender.errors import (
    EmptySpeechTextError,
    EngineBackendError,
    InvalidEngineOptionError,
    SpeechRequestTooLongError,
)
from voicerender.renderplan_input import PreflightDisposition, classify_measurement


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


_MISSING = object()


def _discovered_model(
    *, metadata: dict[str, Any] | None = None, max_tokens: Any = _MISSING
) -> SimpleNamespace:
    model = SimpleNamespace(
        id="nano-0.8-int8",
        display_name="Nano",
        version="0.8",
        language="en-US",
        quality="int8",
        sample_rate=24000,
        aliases=(),
        voices=(),
        voice_ids=(),
        default_voice=None,
        metadata=dict(metadata or {}),
        runtime_available=True,
    )
    if max_tokens is not _MISSING:
        model.max_tokens = max_tokens
    return model


def test_discovery_normalizes_public_maximum_for_hard_profile_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target_from_model(
        _discovered_model(
            metadata={"max_tokens": 200, "capacity_revision": "kitten.capacity.v1"},
            max_tokens=256,
        )
    )
    adapter = KittenSynthEngineAdapter()
    selection = EngineSelection(engine="kitten", target_id=target.id, language="en-US")
    monkeypatch.setattr(adapter, "discover", lambda request: (target,))

    metadata = adapter.target_metadata(selection)
    fragment = adapter.renderprofile_fragment(selection)

    assert target.metadata["max_tokens"] == 256
    assert target.metadata["capacity_policy_revision"] == "kitten.capacity.v1"
    assert metadata["max_tokens"] == 256
    assert metadata["capacity_policy_revision"] == "kitten.capacity.v1"
    assert fragment.renderer_capacity == CapacityConstraint(
        metric="model_tokens", maximum=256, guarantee="hard"
    )
    assert fragment.binding["renderer_metadata"]["max_tokens"] == 256
    assert (
        fragment.binding["renderer_metadata"]["capacity_policy_revision"]
        == "kitten.capacity.v1"
    )


def test_discovery_uses_documented_metadata_key_and_keeps_unknown_capacity_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target_from_model(_discovered_model(metadata={"max_tokens": 128}))
    unknown = _target_from_model(_discovered_model())
    adapter = KittenSynthEngineAdapter()
    selection = EngineSelection(engine="kitten", target_id=target.id, language="en-US")
    monkeypatch.setattr(adapter, "discover", lambda request: (target,))

    fragment = adapter.renderprofile_fragment(selection)

    assert target.metadata["max_tokens"] == 128
    assert fragment.renderer_capacity == CapacityConstraint(
        metric="model_tokens", maximum=128, guarantee="hard"
    )
    assert "max_tokens" not in unknown.metadata
    unknown_adapter = KittenSynthEngineAdapter()
    monkeypatch.setattr(unknown_adapter, "discover", lambda request: (unknown,))
    unknown_fragment = unknown_adapter.renderprofile_fragment(
        EngineSelection(engine="kitten", target_id=unknown.id, language="en-US")
    )
    assert unknown_fragment.renderer_capacity is None


@pytest.mark.parametrize("maximum", [None, True, 0, -1, 1.5, "128"])
def test_discovery_rejects_invalid_public_maximum_without_metadata_fallback(
    maximum: Any,
) -> None:
    target = _target_from_model(
        _discovered_model(metadata={"max_tokens": 128}, max_tokens=maximum)
    )

    assert "max_tokens" not in target.metadata


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


def test_over_limit_measurement_is_replanable_without_synthesis() -> None:
    voice = _FakeVoice(
        SimpleNamespace(
            token_count=513,
            max_tokens=512,
            fits=False,
            model_id="nano-0.8-int8",
            metadata={"capacity_policy_revision": "kitten.capacity.v1"},
        )
    )

    measured = _session(voice).measure(_request())
    outcome = classify_measurement(measured)

    assert measured.fits is False
    assert outcome.disposition is PreflightDisposition.TOO_LONG
    assert outcome.measurement is not None
    assert outcome.measurement.amount == 513
    assert outcome.measurement.maximum == 512
    assert outcome.measurement.unit == "model_tokens"
    assert voice.synthesize_calls == []


@pytest.mark.parametrize(
    ("measurement", "invalid_field"),
    [
        (SimpleNamespace(token_count=True, max_tokens=None, fits=None), "amount"),
        (SimpleNamespace(token_count=27, max_tokens=0, fits=None), "maximum"),
        (SimpleNamespace(token_count=27, max_tokens=100, fits="yes"), "fits"),
    ],
)
def test_measure_rejects_malformed_values(measurement: Any, invalid_field: str) -> None:
    measured = _session(_FakeVoice(measurement)).measure(_request())

    assert measured.source == "kittensynth.invalid_measurement"
    assert invalid_field in measured.details["invalid_fields"]


def test_measure_missing_method_reports_no_measurement() -> None:
    voice = _FakeVoice(None)
    voice.measure_prepared = None  # type: ignore[method-assign]

    measured = _session(voice).measure(_request())

    assert measured == RequestMeasure(
        fits=None,
        amount=None,
        maximum=None,
        unit="unknown",
        source="kittensynth.no_capacity_measurement",
    )


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


def test_synthesis_too_long_error_translates_with_replan_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    native_error_type = type("SynthesisInputTooLongError", (Exception,), {})
    module.SynthesisInputTooLongError = native_error_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    native_error = native_error_type("request exceeds model context")
    native_error.amount = 513
    native_error.maximum = 512
    native_error.unit = "model_tokens"

    class ErrorVoice(_FakeVoice):
        def synthesize_prepared(self, *args: Any, **kwargs: Any) -> Any:
            raise native_error

    with pytest.raises(SpeechRequestTooLongError) as caught:
        _session(ErrorVoice(None)).synthesize(_request())

    assert caught.value.code == "synthesis.request_too_long"
    assert caught.value.amount == 513
    assert caught.value.maximum == 512
    assert caught.value.unit == "model_tokens"
    assert caught.value.native_error_type == "SynthesisInputTooLongError"
    outcome = classify_measurement(caught.value)
    assert outcome.disposition is PreflightDisposition.TOO_LONG
    assert outcome.measurement is not None
    assert outcome.measurement.amount == 513
    assert outcome.measurement.maximum == 512


def test_managed_dependency_error_is_a_non_replanable_runtime_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    module.KittenG2PConfig = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    native_error_type = type("ManagedModelDependencyError", (Exception,), {})
    module.ManagedModelDependencyError = native_error_type  # type: ignore[attr-defined]
    native_error = native_error_type("managed download transport unavailable")

    class FailingKittenVoice:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            raise native_error

    module.KittenVoice = FailingKittenVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    selection = EngineSelection(
        engine="kitten",
        target_id="nano-0.8-int8",
        language="en-US",
        voice="Jasper",
    )

    with pytest.raises(EngineBackendError) as caught:
        KittenSynthEngineAdapter().open(selection)

    assert caught.value.code == "kitten.runtime_dependency_error"
    assert caught.value.native_error_type == "ManagedModelDependencyError"
    assert classify_measurement(caught.value).disposition is PreflightDisposition.REJECT


def test_model_download_error_preserves_safe_transport_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    module.KittenG2PConfig = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    download_error_type = type("ModelDownloadError", (RuntimeError,), {})
    transport_error_type = type("AssetTransportError", (RuntimeError,), {})
    module.ModelDownloadError = download_error_type  # type: ignore[attr-defined]
    transport_error = transport_error_type(
        "The Hub Xet/hf_xet transport failed; pinned HTTPS fallback failed: "
        "timeout https://example.test/model?X-Amz-Signature=signed-secret"
    )
    transport_error.stage = "xet_fallback"
    transport_error.native_error_type = "TimeoutError"
    transport_error.fallback_attempted = True
    transport_error.fallback_failed = True
    transport_error.fallback_outcome = "failed"
    transport_error.retry_attempted = True
    transport_error.retry_failed = True
    transport_error.retry_outcome = "failed"
    native_error = download_error_type(
        f"Managed Kitten model download failed: {transport_error}"
    )
    native_error.__cause__ = transport_error

    class FailingKittenVoice:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            raise native_error

    module.KittenVoice = FailingKittenVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    selection = EngineSelection(
        engine="kitten",
        target_id="nano-0.8-int8",
        language="en-US",
        voice="Jasper",
    )

    with pytest.raises(EngineBackendError) as caught:
        KittenSynthEngineAdapter().open(selection)

    error = caught.value
    assert error.code == "kitten.model_download_error"
    assert error.details["native_error_type"] == "ModelDownloadError"
    assert error.details["engine"] == "kitten"
    assert error.details["target_id"] == "nano-0.8-int8"
    assert error.details["language"] == "en-US"
    assert error.details["voice"] == "Jasper"
    assert error.details["stage"] == "model_install"
    assert error.details["native_stage"] == "xet_fallback"
    assert error.details["transport_error_type"] == "AssetTransportError"
    native_error_message = error.details["native_error_message"]
    assert isinstance(native_error_message, str)
    assert "Xet/hf_xet transport failed" in native_error_message
    assert "signed-secret" not in native_error_message
    assert error.details["transport_native_error_type"] == "TimeoutError"
    assert error.details["fallback_attempted"] is True
    assert error.details["fallback_failed"] is True
    assert error.details["retry_outcome"] == "failed"
    assert "Xet/hf_xet transport failed" in str(error)
    assert "pinned HTTPS fallback failed" in str(error)
    diagnostic = json.dumps(exception_diagnostic(error))
    assert "signed-secret" not in diagnostic
    assert "X-Amz-Signature" not in diagnostic
    assert "?[REDACTED]" in diagnostic


def test_open_error_fallbacks_do_not_claim_model_download_or_xet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    module.KittenG2PConfig = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    module.ModelDownloadError = type("ModelDownloadError", (RuntimeError,), {})  # type: ignore[attr-defined]
    native_error = ValueError("generic options failure")

    class FailingKittenVoice:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            raise native_error

    module.KittenVoice = FailingKittenVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    selection = EngineSelection(
        engine="kitten", target_id="nano-0.8-int8", language="en-US", voice="Jasper"
    )

    with pytest.raises(InvalidEngineOptionError) as caught:
        KittenSynthEngineAdapter().open(selection)

    assert caught.value.code == "kitten.invalid_options"
    assert "Xet" not in str(caught.value)


def test_arbitrary_asset_download_error_keeps_generic_backend_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("kittensynth")
    module.KittenG2PConfig = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    module.ModelDownloadError = type("ModelDownloadError", (RuntimeError,), {})  # type: ignore[attr-defined]
    asset_error_type = type("AssetDownloadError", (RuntimeError,), {})
    native_error = asset_error_type("unrelated asset failure")

    class FailingKittenVoice:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            raise native_error

    module.KittenVoice = FailingKittenVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kittensynth", module)
    selection = EngineSelection(
        engine="kitten", target_id="nano-0.8-int8", language="en-US", voice="Jasper"
    )

    with pytest.raises(EngineBackendError) as caught:
        KittenSynthEngineAdapter().open(selection)

    assert caught.value.code == "kitten.engine_error"
    assert "Xet" not in str(caught.value)


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

    assert identity["pcm_range_policy"] == KITTEN_PCM_RANGE_POLICY
