"""UtilizationSlidingWindow: a recent-history view over utilization data.

The future reclamation engine has to tell a GPU that has been quiet
for a sustained period apart from one that just dipped for a second
while loading its next batch. Both look identical if you only check
the current reading - telling them apart needs recent history, and
scanning the *entire* history every time would get slower as a GPU
runs longer. This structure keeps only the observations that still
fall inside a fixed trailing time window, so later phases can inspect
"the last N minutes" without ever touching data older than that.

Built directly against Phase 1's `UtilizationObservation` (rather than
generically) because everything about it - what "recent" means, what
gets evicted - is specifically about timestamped utilization data.
It composes the generic `Queue`: observations always arrive in
increasing timestamp order and are always evicted oldest-first, which
is exactly what a FIFO queue is for.

This structure does NOT decide anything about the <2%/20-30min or
<15%/2-3hr thresholds from the project's reclamation rules - it only
keeps the observations later phases will need to check them.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from engine.dsa.queue import Queue
from engine.models.utilization import UtilizationObservation


class UtilizationSlidingWindow:
    """Keeps only the `UtilizationObservation`s within ``window_duration``.

    Complexity
    ----------
    add_observation      O(1)       - enqueue, plus eviction (see below)
    evict_expired(now)      O(k)       - k = observations expired since the last
                                          call; amortized O(1) per observation
                                          over its whole lifetime in the window
    observations(now)        O(n)       - n = observations currently in the window
    window_span(now)           O(1)       - just look at the two ends
    size / is_empty               O(1)
    """

    def __init__(self, window_duration: timedelta) -> None:
        if window_duration <= timedelta(0):
            raise ValueError("window_duration must be positive")
        self._window_duration = window_duration
        self._queue: Queue[UtilizationObservation] = Queue()
        self._latest: Optional[UtilizationObservation] = None

    def add_observation(self, observation: UtilizationObservation) -> None:
        """Record a new observation and drop anything now out of window. O(1)."""
        self._queue.enqueue(observation)
        if self._latest is None or observation.timestamp >= self._latest.timestamp:
            self._latest = observation
        self.evict_expired(observation.timestamp)

    def evict_expired(self, now: Optional[datetime] = None) -> int:
        """Drop observations older than ``window_duration`` before ``now``.

        Returns how many observations were evicted. Safe to call on
        its own (e.g. periodically, even with no new data arriving) to
        age the window forward.
        """
        if now is None:
            now = self._latest.timestamp if self._latest is not None else None
        if now is None:
            return 0
        cutoff = now - self._window_duration
        evicted = 0
        while True:
            oldest = self._queue.peek()
            if oldest is None or oldest.timestamp >= cutoff:
                break
            self._queue.dequeue()
            evicted += 1
        return evicted

    def observations(self, now: Optional[datetime] = None) -> List[UtilizationObservation]:
        """The observations currently inside the window, oldest first."""
        self.evict_expired(now)
        return self._queue.to_list()

    def latest(self) -> Optional[UtilizationObservation]:
        """The most recently added observation, or None if the window is empty."""
        return self._latest if not self.is_empty() else None

    def window_span(self, now: Optional[datetime] = None) -> Optional[timedelta]:
        """Time between the oldest and newest observation still in the window."""
        current = self.observations(now)
        if not current:
            return None
        return current[-1].timestamp - current[0].timestamp

    def size(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return self._queue.is_empty()
