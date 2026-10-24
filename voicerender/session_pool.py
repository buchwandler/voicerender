"""Bounded LRU residency for engine sessions."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal

from .engines.base import EngineSelection, EngineSession
from .engines.registry import open_session
from .errors import PlanError


@dataclass(slots=True)
class SessionPoolStats:
    opens: int = 0
    closes: int = 0
    evictions: int = 0
    hits: int = 0
    peak_open: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "opens": self.opens,
            "closes": self.closes,
            "evictions": self.evictions,
            "hits": self.hits,
            "peak_open_sessions": self.peak_open,
        }


@dataclass(frozen=True, slots=True)
class SessionOpenFailure:
    error: Exception


@dataclass(slots=True)
class _SessionEntry:
    key: str
    selection: EngineSelection
    session: EngineSession
    stack: ExitStack
    context: Mapping[str, object] = field(default_factory=dict)


SessionCallback = Callable[..., None]


def validate_max_open_sessions(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PlanError(
            "max_open_sessions must be an integer greater than or equal to 1"
        )
    return value


class EngineSessionPool:
    """Keep at most N sessions resident, evicting the least-recently-used first."""

    def __init__(
        self,
        *,
        max_open_sessions: int,
        on_open_started: SessionCallback | None = None,
        on_open_finished: SessionCallback | None = None,
        on_close_started: SessionCallback | None = None,
        on_close_finished: SessionCallback | None = None,
    ) -> None:
        self.max_open_sessions = validate_max_open_sessions(max_open_sessions)
        self.on_open_started = on_open_started
        self.on_open_finished = on_open_finished
        self.on_close_started = on_close_started
        self.on_close_finished = on_close_finished
        self.stats = SessionPoolStats()
        self._entries: OrderedDict[str, _SessionEntry] = OrderedDict()
        self._failures: dict[str, SessionOpenFailure] = {}
        self._closed = False

    def acquire(
        self,
        key: str,
        selection: EngineSelection,
        *,
        context: Mapping[str, object] | None = None,
    ) -> EngineSession | SessionOpenFailure:
        if self._closed:
            raise RuntimeError(
                "Cannot acquire a session from a closed EngineSessionPool"
            )
        failed = self._failures.get(key)
        if failed is not None:
            return failed
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            self.stats.hits += 1
            return entry.session

        if len(self._entries) >= self.max_open_sessions:
            oldest_key = next(iter(self._entries))
            self.stats.evictions += 1
            self._close_entry(oldest_key, status="evicted")

        event_context = dict(context or {})
        started = monotonic()
        self._notify(self.on_open_started, key, selection, event_context)
        stack = ExitStack()
        try:
            session = stack.enter_context(open_session(selection))
        except Exception as exc:  # noqa: BLE001 - adapter startup errors are memoized per key.
            failure = SessionOpenFailure(exc)
            self._failures[key] = failure
            self._notify(
                self.on_open_finished,
                key,
                selection,
                event_context,
                "failed",
                exc,
                max(0.0, monotonic() - started),
            )
            return failure

        entry = _SessionEntry(key, selection, session, stack, event_context)
        self._entries[key] = entry
        self.stats.opens += 1
        self.stats.peak_open = max(self.stats.peak_open, len(self._entries))
        self._notify(
            self.on_open_finished,
            key,
            selection,
            event_context,
            "opened",
            None,
            max(0.0, monotonic() - started),
        )
        return session

    def _close_entry(self, key: str, *, status: str) -> None:
        entry = self._entries.pop(key)
        started = monotonic()
        first_error: Exception | None = None
        try:
            self._notify(
                self.on_close_started,
                entry.key,
                entry.selection,
                entry.context,
                status,
            )
        except Exception as exc:  # noqa: BLE001 - cleanup must proceed after callback errors.
            first_error = exc

        close_error: Exception | None = None
        try:
            entry.stack.close()
        except Exception as exc:  # noqa: BLE001 - continue closing remaining sessions.
            close_error = exc
            if first_error is None:
                first_error = exc
        self.stats.closes += 1
        elapsed = max(0.0, monotonic() - started)
        try:
            self._notify(
                self.on_close_finished,
                entry.key,
                entry.selection,
                entry.context,
                status,
                close_error,
                elapsed,
            )
        except Exception as exc:  # noqa: BLE001 - run all close-finished callbacks.
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error

    @staticmethod
    def _notify(callback: SessionCallback | None, *args: Any) -> None:
        if callback is not None:
            callback(*args)

    def close(self, *, status: str = "render_complete") -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        while self._entries:
            newest_key = next(reversed(self._entries))
            try:
                self._close_entry(newest_key, status=status)
            except Exception as exc:  # noqa: BLE001 - continue closing the pool.
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> EngineSessionPool:  # noqa: PYI034 - Python 3.10 support.
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed EngineSessionPool")
        return self

    def __exit__(
        self, exc_type: object, exc: object, _traceback: object
    ) -> Literal[False]:
        status = "failure_cleanup" if exc_type is not None else "render_complete"
        try:
            self.close(status=status)
        except Exception:
            if exc is None:
                raise
        return False


__all__ = [
    "EngineSessionPool",
    "SessionOpenFailure",
    "SessionPoolStats",
    "validate_max_open_sessions",
]
