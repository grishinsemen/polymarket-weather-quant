"""Clock abstraction and temporal-integrity guards for zero data leakage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol


class TemporalIntegrityError(ValueError):
    """Raised when a feature, forecast, or observation violates ts < decision_time."""


class Clock(Protocol):
    """Time source abstraction allowing deterministic VirtualClock replay."""

    def now(self) -> datetime:
        """Return current UTC timestamp."""
        ...


class SystemClock:
    """Live wall-clock UTC time provider."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class VirtualClock:
    """Controllable deterministic clock for tests and Experiment Replay."""

    def __init__(self, initial_utc: datetime) -> None:
        self._now = _ensure_utc(initial_utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        self._now = self._now + delta
        return self._now

    def set(self, new_utc: datetime) -> datetime:
        self._now = _ensure_utc(new_utc)
        return self._now


def ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


_ensure_utc = ensure_utc


def assert_before_decision(
    ts: datetime,
    decision_time: datetime,
    label: str = "timestamp",
) -> None:
    """Enforce strict temporal ordering: ts must be strictly prior to decision_time."""
    ts_utc = _ensure_utc(ts)
    decision_utc = _ensure_utc(decision_time)
    if ts_utc >= decision_utc:
        raise TemporalIntegrityError(
            f"Temporal integrity violation for {label}: "
            f"{ts_utc.isoformat()} >= decision_time {decision_utc.isoformat()}"
        )
