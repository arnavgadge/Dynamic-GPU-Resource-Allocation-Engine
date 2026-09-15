"""The User model: an employee drawing from the company GPU pool."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from engine.models.enums import Priority


@dataclass
class User:
    """An employee who submits jobs and may hold GPUs from the pool.

    A user's ``assigned_gpu_ids`` can be empty, one GPU, or many -
    there is no fixed ownership. The user never chooses which GPU
    they get; the scheduler decides that and updates this list.
    """

    user_id: str
    name: str
    priority: Priority

    assigned_gpu_ids: List[str] = field(default_factory=list)
    running_job_ids: List[str] = field(default_factory=list)

    # Room for scheduler-relevant extras (e.g. a deadline, a team
    # name) without forcing a schema change on this class every time
    # a later phase needs one more piece of context.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def gpu_count(self) -> int:
        return len(self.assigned_gpu_ids)
