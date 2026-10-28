from __future__ import annotations

import numpy as np
import pytest

from voicerender.audio_boundaries import (
    BOUNDARY_POLICY_ID,
    BoundaryNormalization,
    clamp_word_timings_to_audio,
    trim_renderer_trailing_silence,
)
from voicerender.engines.base import SpeechWordTiming

_SAMPLE_RATE = 1000


def _speech(frames: int) -> np.ndarray:
    samples = np.arange(frames, dtype=np.float32)
    return (0.2 * np.sin(2.0 * np.pi * 120.0 * samples / _SAMPLE_RATE)).astype(
        np.float32
    )


def test_trailing_zeros_are_removed_with_policy_statistics() -> None:
    active = _speech(1000)
    source = np.concatenate((active, np.zeros(500, dtype=np.float32)))

    normalized, stats = trim_renderer_trailing_silence(source, sample_rate=_SAMPLE_RATE)

    assert np.array_equal(normalized, active)
    assert stats == BoundaryNormalization(
        policy=BOUNDARY_POLICY_ID,
        input_frames=1500,
        output_frames=1000,
        removed_frames=500,
        removed_seconds=0.5,
    )


def test_audio_without_trailing_silence_is_unchanged() -> None:
    source = _speech(1000)

    normalized, stats = trim_renderer_trailing_silence(source, sample_rate=_SAMPLE_RATE)

    assert np.array_equal(normalized, source)
    assert stats.removed_frames == 0
    assert stats.output_frames == source.size


def test_leading_silence_is_preserved_while_tail_is_removed() -> None:
    leading = np.zeros(200, dtype=np.float32)
    active = _speech(1000)
    source = np.concatenate((leading, active, np.zeros(500, dtype=np.float32)))

    normalized, stats = trim_renderer_trailing_silence(source, sample_rate=_SAMPLE_RATE)

    assert normalized.size == leading.size + active.size
    assert np.array_equal(normalized[: leading.size], leading)
    assert np.array_equal(normalized[leading.size :], active)
    assert stats.removed_frames == 500


def test_all_silent_input_is_preserved_without_fabricating_audio() -> None:
    source = np.zeros(400, dtype=np.float32)

    normalized, stats = trim_renderer_trailing_silence(source, sample_rate=_SAMPLE_RATE)

    assert normalized.size == source.size
    assert np.array_equal(normalized, source)
    assert stats.removed_frames == 0


def test_result_is_mono_float32_and_repeated_calls_are_deterministic() -> None:
    source = np.concatenate((_speech(1000), np.zeros(500, dtype=np.float64)))

    first, first_stats = trim_renderer_trailing_silence(
        source, sample_rate=_SAMPLE_RATE
    )
    second, second_stats = trim_renderer_trailing_silence(
        source, sample_rate=_SAMPLE_RATE
    )

    assert first.dtype == np.float32
    assert first.ndim == 1
    assert np.array_equal(first, second)
    assert first_stats == second_stats


def test_clamp_word_timing_ends_without_shifting_starts_and_rejects_invalid_start() -> (
    None
):
    timing = SpeechWordTiming(
        text="word",
        char_start=0,
        char_end=4,
        start_sample=80,
        end_sample=110,
    )

    clamped = clamp_word_timings_to_audio((timing,), frame_count=100)

    assert clamped[0].start_sample == 80
    assert clamped[0].end_sample == 100
    with pytest.raises(ValueError, match="starts beyond"):
        clamp_word_timings_to_audio(
            (SpeechWordTiming("word", 0, 4, 100, 110),), frame_count=100
        )
