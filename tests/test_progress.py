from __future__ import annotations

import pytest

from voicerender.progress import ProgressTracker, RollingEta


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_tracker_starts_at_zero_and_uses_injected_monotonic_clock() -> None:
    clock = _Clock(10.0)
    tracker = ProgressTracker(2, clock=clock)

    initial = tracker.snapshot()
    assert initial.completed_count == 0
    assert initial.progress == 0.0
    assert initial.eta_seconds is None
    assert initial.eta_confidence == "unavailable"

    clock.value = 13.25
    assert tracker.snapshot().elapsed_seconds == 3.25


def test_terminal_outcomes_advance_progress_and_finish_at_one() -> None:
    tracker = ProgressTracker(3, clock=_Clock())

    first = tracker.complete("reused")
    second = tracker.complete("failed")
    final = tracker.complete("rendered", synthesis_seconds=2.5)

    assert [first.progress, second.progress, final.progress] == [1 / 3, 2 / 3, 1.0]
    assert final.completed_count == final.total_count == 3
    assert final.rendered_count == 1
    assert final.reused_count == 1
    assert final.failed_count == 1
    assert final.eta_seconds == 0.0
    assert [first.progress, second.progress, final.progress] == sorted(
        [first.progress, second.progress, final.progress]
    )


def test_zero_total_is_a_complete_empty_workload() -> None:
    snapshot = ProgressTracker(0, clock=_Clock()).snapshot()

    assert snapshot.completed_count == 0
    assert snapshot.total_count == 0
    assert snapshot.progress == 1.0
    assert snapshot.eta_seconds == 0.0


def test_eta_is_unknown_until_synthesis_and_reuse_reduces_remaining_work() -> None:
    tracker = ProgressTracker(4, clock=_Clock())

    reused = tracker.complete("reused")
    assert reused.eta_seconds is None
    assert reused.eta_confidence == "unavailable"

    synthesized = tracker.complete("rendered", synthesis_seconds=3.0)
    assert synthesized.eta_seconds == 6.0
    assert synthesized.eta_confidence == "low"

    after_reuse = tracker.complete("reused")
    assert after_reuse.eta_seconds == 3.0
    assert after_reuse.eta_confidence == "low"


def test_reuse_cannot_contribute_a_synthesis_timing_sample() -> None:
    tracker = ProgressTracker(3, clock=_Clock())
    tracker.complete("rendered", synthesis_seconds=10.0)

    with pytest.raises(ValueError, match="only rendered"):
        tracker.complete("reused", synthesis_seconds=0.001)

    snapshot = tracker.snapshot()
    assert snapshot.completed_count == 1
    assert snapshot.eta_seconds == 20.0


def test_eta_uses_bounded_rolling_median_and_confidence_thresholds() -> None:
    estimator = RollingEta(window_size=8)
    assert estimator.confidence == "unavailable"
    assert estimator.estimate(3) is None

    estimator.add_sample(10.0)
    assert estimator.confidence == "low"
    assert estimator.estimate(2) == 20.0

    estimator.add_sample(2.0)
    assert estimator.confidence == "medium"
    assert estimator.estimate(2) == 12.0
    estimator.add_sample(4.0)
    estimator.add_sample(6.0)
    assert estimator.confidence == "medium"
    estimator.add_sample(8.0)
    assert estimator.confidence == "high"

    # The first 10-second warmup sample has fallen outside the bounded window.
    for duration in (10.0, 10.0, 10.0, 10.0):
        estimator.add_sample(duration)
    assert estimator.sample_count == 8
    assert estimator.estimate(1) == 9.0


def test_elapsed_time_never_regresses_if_test_clock_moves_backwards() -> None:
    clock = _Clock(5.0)
    tracker = ProgressTracker(2, clock=clock)
    clock.value = 8.0
    assert tracker.snapshot().elapsed_seconds == 3.0
    clock.value = 7.0
    assert tracker.snapshot().elapsed_seconds == 3.0


def test_invalid_totals_durations_and_over_completion_are_rejected() -> None:
    with pytest.raises(ValueError, match="total_count"):
        ProgressTracker(-1, clock=_Clock())

    with pytest.raises(ValueError, match="finite non-negative"):
        RollingEta().add_sample(float("nan"))

    with pytest.raises(ValueError, match="remaining_items"):
        RollingEta().estimate(-1)

    tracker = ProgressTracker(1, clock=_Clock())
    tracker.complete("failed")
    with pytest.raises(ValueError, match="more items"):
        tracker.complete("failed")


def test_advance_tracks_non_synthesis_work_without_result_counts() -> None:
    tracker = ProgressTracker(2, clock=_Clock())

    snapshot = tracker.advance()

    assert snapshot.completed_count == 1
    assert snapshot.progress == 0.5
    assert (
        snapshot.rendered_count == snapshot.reused_count == snapshot.failed_count == 0
    )
    assert snapshot.eta_seconds is None
