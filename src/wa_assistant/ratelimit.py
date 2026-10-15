"""Per-user fixed-window rate limiting (in memory, thread-safe)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum


class RateDecision(Enum):
    ALLOW = "allow"
    NOTIFY = "notify"  # first message over the limit in this window: tell the user once
    DROP = "drop"  # further messages in the same window: stay silent


@dataclass(slots=True)
class _Window:
    start: float
    count: int = 0
    notified: bool = False


class RateLimiter:
    """At most ``limit`` messages per ``window_seconds`` per key; ``limit=0`` disables it."""

    def __init__(self, limit: int, window_seconds: float = 60.0, max_keys: int = 10_000) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._max_keys = max_keys
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float) -> RateDecision:
        """Count one message from ``key`` at ``now`` (seconds) and decide what to do with it."""
        if self.limit <= 0:
            return RateDecision.ALLOW
        with self._lock:
            window = self._windows.get(key)
            if window is None or now - window.start >= self.window_seconds:
                if len(self._windows) >= self._max_keys:
                    self._evict(now)
                window = self._windows[key] = _Window(start=now)
            window.count += 1
            if window.count <= self.limit:
                return RateDecision.ALLOW
            if not window.notified:
                window.notified = True
                return RateDecision.NOTIFY
            return RateDecision.DROP

    def _evict(self, now: float) -> None:
        expired = [k for k, w in self._windows.items() if now - w.start >= self.window_seconds]
        for key in expired or list(self._windows)[: self._max_keys // 10]:
            del self._windows[key]
