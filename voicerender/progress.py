"""Formatting-neutral progress accounting and live synthesis ETA estimation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from statistics import median
from time import monotonic
from typing import Literal

EtaConfidence = Literal["unavailable", "low", "medium", "high"]
TerminalOutcome = Literal["rendered", "reused", "failed"]


@dataclass(frozen=True, slots=True)
class RenderProgress:
    """Immutable progress snapshot for one render invocation."""

    completed_count: int
    total_count: int
    progress: float
    rendered_count: int
    reused_count: int
    failed_count: int
    elapsed_seconds: float
    eta_seconds: float | None
    eta_confidence: EtaConfidence


class RollingEta:
    """Estimate remaining time from a bounded window of successful syntheses."""

    def __init__(self, *, window_size: int = 8) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self._durations: deque[float] = deque(maxlen=window_size)

    @property
    def sample_count(self) -> int:
        return len(self._durations)

    @property
    def confidence(self) -> EtaConfidence:
        count = self.sample_count
        if count == 0:
            return "unavailable"
        if count == 1:
            return "low"
        if count < 5:
            return "medium"
        return "high"

    def add_sample(self, duration_seconds: float) -> None:
        duration = float(duration_seconds)
        if not isfinite(duration) or duration < 0:
            raise ValueError("synthesis duration must be a finite non-negative number")
        self._durations.append(duration)

    def estimate(self, remaining_items: int) -> float | None:
        if remaining_items < 0:
            raise ValueError("remaining_items must be non-negative")
        if remaining_items == 0:
            return 0.0
        if not self._durations:
            return None
        return float(median(self._durations)) * remaining_items


class ProgressTracker:
    """Track terminal speech-item outcomes and estimate remaining synthesis time.

    Reuse and failure advance occurrence progress, but only successfully rendered
    items contribute duration samples. The ETA applies the rolling median
    synthesis duration to unresolved occurrences; it is intentionally unknown
    until a successful synthesis sample exists.
    """

    def __init__(
        self,
        total_count: int,
        *,
        clock: Callable[[], float] = monotonic,
        eta_window: int = 8,
    ) -> None:
        if total_count < 0:
            raise ValueError("total_count must be non-negative")
        self.total_count = total_count
        self._clock = clock
        self._started_at = float(clock())
        if not isfinite(self._started_at):
            raise ValueError("clock must return a finite value")
        self._last_elapsed = 0.0
        self._completed_count = 0
        self._rendered_count = 0
        self._reused_count = 0
        self._failed_count = 0
        self._eta = RollingEta(window_size=eta_window)

    @property
    def completed_count(self) -> int:
        return self._completed_count

    @property
    def rendered_count(self) -> int:
        return self._rendered_count

    @property
    def reused_count(self) -> int:
        return self._reused_count

    @property
    def failed_count(self) -> int:
        return self._failed_count

    @property
    def eta_confidence(self) -> EtaConfidence:
        return self._eta.confidence

    def advance(self) -> RenderProgress:
        """Advance terminal work without assigning it to a synthesis result category."""

        if self._completed_count >= self.total_count:
            raise ValueError("cannot complete more items than total_count")
        self._completed_count += 1
        return self.snapshot()

    def complete(
        self,
        outcome: TerminalOutcome,
        *,
        synthesis_seconds: float | None = None,
    ) -> RenderProgress:
        """Record one terminal item; pass duration only after artifact commit."""

        if outcome not in ("rendered", "reused", "failed"):
            raise ValueError(f"unknown terminal outcome: {outcome!r}")
        if self._completed_count >= self.total_count:
            raise ValueError("cannot complete more items than total_count")
        if synthesis_seconds is not None:
            if outcome != "rendered":
                raise ValueError("only rendered items may contribute synthesis timing")
            self._eta.add_sample(synthesis_seconds)

        self._completed_count += 1
        if outcome == "rendered":
            self._rendered_count += 1
        elif outcome == "reused":
            self._reused_count += 1
        else:
            self._failed_count += 1
        return self.snapshot()

    def snapshot(self) -> RenderProgress:
        """Return a current immutable snapshot without changing item counts."""

        now = float(self._clock())
        if not isfinite(now):
            raise ValueError("clock must return a finite value")
        self._last_elapsed = max(self._last_elapsed, now - self._started_at, 0.0)
        remaining = self.total_count - self._completed_count
        fraction = self._completed_count / self.total_count if self.total_count else 1.0
        return RenderProgress(
            completed_count=self._completed_count,
            total_count=self.total_count,
            progress=fraction,
            rendered_count=self._rendered_count,
            reused_count=self._reused_count,
            failed_count=self._failed_count,
            elapsed_seconds=self._last_elapsed,
            eta_seconds=self._eta.estimate(remaining),
            eta_confidence=self._eta.confidence,
        )


__all__ = ["EtaConfidence", "ProgressTracker", "RenderProgress", "RollingEta"]
