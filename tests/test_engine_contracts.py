from __future__ import annotations

import numpy as np
import pytest

from voicerender.engines.base import (
    RenderedSpeech,
    SpeechRequest,
    validate_rendered_speech,
)
from voicerender.errors import EngineBackendError


def _validate(audio: np.ndarray) -> RenderedSpeech:
    request = SpeechRequest(id="request-1", text="hello", language="en-US")
    result = RenderedSpeech(id=request.id, audio=audio, sample_rate=24000)
    return validate_rendered_speech(request, result, engine="fake")


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
