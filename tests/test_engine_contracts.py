from __future__ import annotations

import numpy as np
import pytest

from voicerender.engines.base import (
    PronunciationSpan,
    RenderedSpeech,
    SayAsSpan,
    SpeechRequest,
    validate_rendered_speech,
)
from voicerender.errors import EngineBackendError


def _validate(audio: np.ndarray) -> RenderedSpeech:
    request = SpeechRequest(id="request-1", text="hello", language="en-US")
    result = RenderedSpeech(id=request.id, audio=audio, sample_rate=24000)
    return validate_rendered_speech(request, result, engine="fake")


def test_say_as_span_validates_basic_fields() -> None:
    span = SayAsSpan(0, 3, "characters", format="spell-out", detail="letters")
    assert span.start == 0
    assert span.end == 3
    assert span.interpret_as == "characters"

    with pytest.raises(TypeError, match="start"):
        SayAsSpan(True, 3, "characters")
    with pytest.raises(ValueError, match="range"):
        SayAsSpan(3, 3, "characters")
    with pytest.raises(ValueError, match="interpret_as"):
        SayAsSpan(0, 1, "  ")
    with pytest.raises(TypeError, match="detail"):
        SayAsSpan(0, 1, "characters", detail=1)  # type: ignore[arg-type]


def test_speech_request_rejects_overlapping_say_as_and_pronunciation() -> None:
    with pytest.raises(ValueError, match="overlap say_as and pronunciation"):
        SpeechRequest(
            id="request-1",
            text="A R T",
            language="en-US",
            pronunciation_overrides=(PronunciationSpan(2, 5, phonemes="ɑːɹ tiː"),),
            say_as=(SayAsSpan(0, 5, "characters"),),
        )


def test_rendered_speech_accepts_pcm_full_scale_peak() -> None:
    rendered = _validate(np.array([-1.0, 0.0, 1.0], dtype=np.float32))

    np.testing.assert_array_equal(rendered.audio, [-1.0, 0.0, 1.0])


@pytest.mark.parametrize("peak", [1.0, 1.000005, 1.00001])
def test_rendered_speech_preserves_existing_pcm_range_tolerance(peak: float) -> None:
    rendered = _validate(np.array([0.0, peak], dtype=np.float32))

    assert rendered.audio[-1] == pytest.approx(peak)


def test_rendered_speech_rejects_audio_above_pcm_range_with_peak() -> None:
    with pytest.raises(
        EngineBackendError, match=r"PCM full-scale range.*peak=1\.010000"
    ):
        _validate(np.array([0.0, 1.01], dtype=np.float32))


@pytest.mark.parametrize("bad_sample", [np.nan, np.inf])
def test_rendered_speech_rejects_non_finite_audio(bad_sample: float) -> None:
    with pytest.raises(EngineBackendError, match="finite"):
        _validate(np.array([0.0, bad_sample], dtype=np.float32))


def test_rendered_speech_rejects_multichannel_audio() -> None:
    with pytest.raises(EngineBackendError, match="mono"):
        _validate(np.zeros((2, 3), dtype=np.float32))
