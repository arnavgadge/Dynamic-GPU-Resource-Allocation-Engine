"""The Job model: a workload submitted by a user."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from engine.models.enums import JobStatus, Priority


@dataclass
class Job:
    """A workload submitted by a user, waiting for or running on a GPU.

    The engine is workload-agnostic: ``name`` is a free-text label
    ("ML Training", "Excel", "GTA5", ...) that exists purely for
    display in the frontend/event log. Nothing in the model - and
    nothing later phases add - should branch on this string.

    ``gpu_count`` is how many GPUs this job actually needs (1-9 for a
    normal user request; see `api/config.py` for the enforced range -
    this model itself places no upper bound, that is API-layer
    policy). ``assigned_gpu_ids`` is how many it *currently holds*,
    which can be fewer than ``gpu_count`` while a multi-GPU request is
    only partially satisfied: the job stays ``WAITING`` (still
    eligible to be picked by `AllocationEngine.select_next_job` for
    its *next* GPU) until it holds ``gpu_count`` of them, at which
    point it becomes ``RUNNING``. A job that later loses one GPU (a
    partial reclaim) but still holds at least one stays ``RUNNING`` -
    losing one of several GPUs is not the same as the job finishing;
    see `ReclamationEngine._reclaim`.
    """

    job_id: str
    user_id: str
    name: str
    priority: Priority
    estimated_size_minutes: float
    gpu_count: int = 1

    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    status: JobStatus = JobStatus.WAITING
    assigned_gpu_ids: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def assigned_gpu_id(self) -> Optional[str]:
        """Backward-compatible single-GPU view: the first GPU this job
        holds, or ``None`` if it holds none yet. For a ``gpu_count=1``
        job (the default, and every job before this phase) this is
        exactly the old ``assigned_gpu_id`` field's value - read-only,
        because a multi-GPU job's assignment set must change through
        `assigned_gpu_ids` (append/remove one id at a time), never by
        overwriting "the" GPU.
        """
        return self.assigned_gpu_ids[0] if self.assigned_gpu_ids else None

    @property
    def gpus_still_needed(self) -> int:
        """How many more GPUs this job needs before it is fully running."""
        return max(self.gpu_count - len(self.assigned_gpu_ids), 0)

    @property
    def is_fully_allocated(self) -> bool:
        return len(self.assigned_gpu_ids) >= self.gpu_count

    @property
    def waiting_time(self) -> timedelta:
        """How long the job has waited (or waited before it started).

        Uses ``started_at`` once the job is running; otherwise measures
        against now. This is data later phases can use to break ties
        (e.g. FCFS) - it does not itself decide anything.
        """
        end = self.started_at or datetime.now(timezone.utc)
        return end - self.submitted_at
