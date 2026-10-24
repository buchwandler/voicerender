from __future__ import annotations

from contextlib import contextmanager
from typing import cast

import pytest

from voicerender import session_pool
from voicerender.engines.base import EngineSelection
from voicerender.errors import PlanError
from voicerender.session_pool import EngineSessionPool, SessionOpenFailure


class _Session:
    def measure(self, _request):
        return None

    def synthesize(self, _request):
        return None


def _selection(key: str) -> EngineSelection:
    return EngineSelection(engine="test", target_id=key, language="en-US")


def _install_open(monkeypatch, events, active):
    @contextmanager
    def open_session(selection):
        events.append(("open", selection.target_id))
        active.add(selection.target_id)
        assert len(active) <= 2
        try:
            yield _Session()
        finally:
            active.remove(selection.target_id)
            events.append(("close", selection.target_id))

    monkeypatch.setattr(session_pool, "open_session", open_session)


def test_lru_eviction_closes_before_open_and_closes_in_reverse_use_order(
    monkeypatch,
):
    events = []
    active = set()
    _install_open(monkeypatch, events, active)

    with EngineSessionPool(max_open_sessions=2) as pool:
        assert pool.acquire("A", _selection("A"))
        assert pool.acquire("B", _selection("B"))
        assert pool.acquire("A", _selection("A"))
        assert pool.acquire("C", _selection("C"))
        assert events[:4] == [
            ("open", "A"),
            ("open", "B"),
            ("close", "B"),
            ("open", "C"),
        ]
        assert pool.stats.peak_open == 2
        assert pool.stats.evictions == 1
        assert pool.stats.hits == 1

    assert active == set()
    assert events[-2:] == [("close", "C"), ("close", "A")]
    assert pool.stats.opens == pool.stats.closes == 3


def test_default_limit_adjacent_hits_and_cleanup_after_exception(monkeypatch):
    events = []
    active = set()
    _install_open(monkeypatch, events, active)

    with (
        pytest.raises(RuntimeError, match="render failed"),
        EngineSessionPool(max_open_sessions=1) as pool,
    ):
        assert pool.acquire("A", _selection("A"))
        assert pool.acquire("A", _selection("A"))
        assert pool.stats.hits == 1
        assert pool.stats.peak_open == 1
        raise RuntimeError("render failed")

    assert active == set()
    assert events == [("open", "A"), ("close", "A")]
    assert pool.stats.opens == pool.stats.closes == 1


def test_open_failure_is_memoized_and_does_not_occupy_capacity(monkeypatch):
    attempts = []

    @contextmanager
    def open_session(selection):
        attempts.append(selection.target_id)
        if selection.target_id == "failed":
            raise RuntimeError("startup failed")
        yield _Session()

    monkeypatch.setattr(session_pool, "open_session", open_session)
    with EngineSessionPool(max_open_sessions=1) as pool:
        first = pool.acquire("failed", _selection("failed"))
        second = pool.acquire("failed", _selection("failed"))
        assert isinstance(first, SessionOpenFailure)
        assert second is first
        assert pool.acquire("good", _selection("good"))
        assert attempts == ["failed", "good"]
        assert pool.stats.peak_open == 1


def test_invalid_session_limits_are_rejected():
    for value in (True, False, 0, -1, 1.5, "2"):
        with pytest.raises(PlanError, match="max_open_sessions"):
            EngineSessionPool(max_open_sessions=cast(int, value))
