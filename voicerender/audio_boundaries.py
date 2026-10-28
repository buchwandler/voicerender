"""VoiceRender-owned normalization of renderer-added audio boundaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from audiosig import trim_trail

from .engines.base import SpeechWordTiming

BOUNDARY_POLICY_ID = "voicerender.trailing-silence.v1"


@dataclass(frozen=True, slots=True)
class BoundaryNormalization:
    """Facts about trailing-silence normalization for one speech artifact."""

    policy: str
    input_frames: int
    output_frames: int
    removed_frames: int
    removed_seconds: float


def trim_renderer_trailing_silence(
    audio: Any,
    *,
    sample_rate: int,
) -> tuple[np.ndarray, BoundaryNormalization]:
    """Remove detected renderer tail silence without changing the leading boundary.

    The caller must pass audio that has already passed its engine adapter's
    rendered-speech validation. The returned array is mono float32.
    """
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
    ):
        raise ValueError("sample_rate must be a positive integer")

    source = np.asarray(audio, dtype=np.float32).reshape(-1)
    trimmed = np.asarray(
        trim_trail(source, sample_rate=sample_rate, trail_seconds=0.0),
        dtype=np.float32,
    ).reshape(-1)
    if trimmed.size > source.size:
        raise ValueError("trailing-silence normalization must not add audio frames")

    removed = int(source.size - trimmed.size)
    return trimmed, BoundaryNormalization(
        policy=BOUNDARY_POLICY_ID,
        input_frames=int(source.size),
        output_frames=int(trimmed.size),
        removed_frames=removed,
        removed_seconds=removed / float(sample_rate),
    )


def clamp_word_timings_to_audio(
    timings: tuple[SpeechWordTiming, ...],
    frame_count: int,
) -> tuple[SpeechWordTiming, ...]:
    """Keep legacy word coordinates valid after trailing-only audio trimming.

    Timing starts are never moved. A start outside the new audio boundary is
    an invariant failure; an end extending only into the removed tail is
    clamped to the final frame boundary.
    """
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise ValueError("frame_count must be a positive integer")

    normalized: list[SpeechWordTiming] = []
    for timing in timings:
        if timing.start_sample >= frame_count:
            raise ValueError("word timing starts beyond the trimmed audio boundary")
        if timing.end_sample > frame_count:
            normalized.append(replace(timing, end_sample=frame_count))
        else:
            normalized.append(timing)
    return tuple(normalized)
