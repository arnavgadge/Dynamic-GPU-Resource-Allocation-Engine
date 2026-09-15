"""The GPU model: a state/data object for one GPU in the company pool."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from engine.models.enums import GPUStatus
from engine.models.utilization import UtilizationObservation


@dataclass
class GPU:
    """One physical GPU belonging to the company's shared pool.

    A GPU is a pure state container. It records what is currently
    true (utilization, memory, who/what it is assigned to, its
    status) and keeps a history of utilization observations. It never
    decides to reclaim itself or to hand itself to someone else -
    those decisions belong to the scheduler in later phases.
    """

    gpu_id: str
    total_memory_mb: float
    memory_used_mb: float = 0.0
    utilization_percent: float = 0.0
    status: GPUStatus = GPUStatus.IDLE

    # Current assignment, if any. `None` means the GPU is unassigned
    # and sitting free in the pool - no GPU permanently belongs to a
    # user, so both fields are optional and can change at any time.
    assigned_user_id: Optional[str] = None
    assigned_job_id: Optional[str] = None

    utilization_history: List[UtilizationObservation] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_assigned(self) -> bool:
        """Whether this GPU currently belongs to a user/job."""
        return self.assigned_user_id is not None

    @property
    def memory_free_mb(self) -> float:
        return self.total_memory_mb - self.memory_used_mb

    def record_observation(self, observation: UtilizationObservation) -> None:
        """Append a new utilization sample and sync the "current" fields.

        This only records data - it never changes ``status`` or the
        current assignment. Deciding what a low reading *means* (idle
        warning, reclaim, etc.) is scheduler logic for a later phase.
        """
        self.utilization_history.append(observation)
        self.utilization_percent = observation.utilization_percent
        if observation.memory_used_mb is not None:
            self.memory_used_mb = observation.memory_used_mb
        self.last_updated = observation.timestamp
