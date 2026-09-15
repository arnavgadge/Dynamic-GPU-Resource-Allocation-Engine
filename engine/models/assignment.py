"""The GPUAssignment model: the record of a GPU being held by a user/job."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class GPUAssignment:
    """A record that GPU X was given to user Y (running job Z) at time T.

    Design note - why a dedicated model instead of only fields on
    GPU/User/Job:

    ``GPU.assigned_user_id`` / ``User.assigned_gpu_ids`` /
    ``Job.assigned_gpu_id`` are enough to answer "what is true right
    now" in O(1). But the project also needs to answer "when was this
    assignment created" and "when did it change", and later phases
    need a history of assignments to reason about (e.g. reclaim
    history, dynamic reallocation). Cramming timestamps onto three
    different objects and keeping them in sync is error-prone, so a
    small, explicit ``GPUAssignment`` record carries that history
    instead. GPU/User/Job stay the fast "current state" pointers;
    ``GPUAssignment`` is the audit trail those pointers are built
    from.

    This is intentionally NOT a lease: there is no expiry field and
    nothing here enforces a time limit. The scheduler ends an
    assignment (sets ``ended_at``) whenever it decides to, for
    whatever reason later phases implement.
    """

    assignment_id: str
    gpu_id: str
    user_id: str
    job_id: Optional[str] = None

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    def end(self, when: Optional[datetime] = None) -> None:
        """Mark this assignment as no longer current."""
        self.ended_at = when or datetime.now(timezone.utc)
