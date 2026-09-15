"""SimulationClock: the one place simulated "now" comes from.

Every engine in this project already accepts a plain `datetime` for
"now" wherever it needs to reason about elapsed time (Reclamation's
`record_utilization`/`check_timeouts`, Balancing's `route_job`,
Allocation's `finalize_assignment`) - there is no separate clock
abstraction to conflict with, only a convention. This class *is* that
convention's source of truth during a simulation: 20 minutes or 3
hours of "sustained low utilization" history is produced by calling
`advance()` a few times with exact `timedelta`s, never by
`time.sleep`-ing the test process.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class SimulationClock:
    """A monotonically-advancing, entirely synthetic clock.

    ``speed`` is descriptive configuration only - it is not read by
    `advance()` at all. The backend simulator always advances by an
    exact, explicit `timedelta` the caller supplies (deterministic by
    construction); a future real-time-driven frontend could use
    ``speed`` to decide how much simulated time one animation frame of
    *wall* time should correspond to (e.g. "1 real second = `speed`
    simulated minutes"), but that mapping belongs to whatever drives
    the clock, not to the clock itself - see the README's Phase 6
    section for why real sleeping is deliberately never implemented
    here.

    Complexity: every operation is O(1).
    """

    start: datetime
    speed: float = 1.0
    _now: datetime = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        self._now = self.start

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        """Move the clock forward by exactly ``delta``. O(1)."""
        if delta < timedelta(0):
            raise ValueError("a simulated clock cannot advance backwards")
        self._now = self._now + delta
        return self._now

    def set(self, when: datetime) -> None:
        """Jump directly to ``when`` (must not be earlier than now). O(1)."""
        if when < self._now:
            raise ValueError("a simulated clock cannot be set backwards")
        self._now = when

    def reset(self) -> None:
        """Return to the clock's original start time. O(1)."""
        self._now = self.start
