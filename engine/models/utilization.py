"""A single point-in-time reading of a GPU's utilization."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class UtilizationObservation:
    """One sample of how busy a GPU was at a point in time.

    A GPU accumulates a history of these. Later phases will scan that
    history to detect sustained low-utilization windows (e.g. "below
    2% for 20-30 minutes" or "below 15% for 2-3 hours"). Phase 1 only
    needs to guarantee the data is available to scan - no threshold
    or sliding-window logic lives here.
    """

    utilization_percent: float
    timestamp: datetime = None  # type: ignore[assignment]
    memory_used_mb: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if not 0.0 <= self.utilization_percent <= 100.0:
            raise ValueError(
                f"utilization_percent must be within [0, 100], got {self.utilization_percent}"
            )
