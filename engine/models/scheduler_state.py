"""SchedulerState: a central container for the complete system state."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.history_config import MAX_EVENT_HISTORY
from engine.models.assignment import GPUAssignment
from engine.models.enums import EngineStatus, JobStatus
from engine.models.event import Event
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User


@dataclass
class SchedulerState:
    """Everything the scheduler currently knows, in one place.

    This is a state container, not the scheduler. It holds the
    company's GPU pool, its users, their jobs, the current
    assignments and the event log, plus simple lookup/query helpers
    that answer questions like "which GPUs does this user hold" or
    "which jobs are waiting". None of those helpers *decide*
    anything (no picking a winner, no reclaiming) - that logic
    belongs to the scheduling engine built in later phases on top of
    this state.

    Phase 1 deliberately uses plain ``dict``/``list`` lookups here.
    The HashMap/heap/priority-queue versions of these same lookups
    are a Phase 2 concern (see the README).
    """

    gpus: Dict[str, GPU] = field(default_factory=dict)
    users: Dict[str, User] = field(default_factory=dict)
    jobs: Dict[str, Job] = field(default_factory=dict)
    assignments: Dict[str, GPUAssignment] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)

    engine_status: EngineStatus = EngineStatus.STOPPED

    # ---- registration -----------------------------------------------
    # These only add to the state container; they do not decide
    # anything about who should get what.

    def add_gpu(self, gpu: GPU) -> None:
        self.gpus[gpu.gpu_id] = gpu

    def add_user(self, user: User) -> None:
        self.users[user.user_id] = user

    def add_job(self, job: Job) -> None:
        self.jobs[job.job_id] = job

    def add_assignment(self, assignment: GPUAssignment) -> None:
        self.assignments[assignment.assignment_id] = assignment

    def log_event(self, event: Event) -> None:
        """Append one event, then evict the oldest if the retention
        cap (`MAX_EVENT_HISTORY`, Phase 15 of the 100-scenario fix
        set) is exceeded - deterministic, oldest-first, and never
        touching any *active* state (GPUs/users/jobs/assignments),
        only this historical log.
        """
        self.events.append(event)
        if len(self.events) > MAX_EVENT_HISTORY:
            del self.events[: len(self.events) - MAX_EVENT_HISTORY]

    # ---- queries -------------------------------------------------
    # Read-only helpers answering the relationship questions the
    # scheduler will need. Plain lookups/filters - no scheduling
    # decisions.

    def get_gpu(self, gpu_id: str) -> Optional[GPU]:
        return self.gpus.get(gpu_id)

    def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def get_gpus_for_user(self, user_id: str) -> List[GPU]:
        """Which GPUs does this user currently hold?"""
        return [gpu for gpu in self.gpus.values() if gpu.assigned_user_id == user_id]

    def get_job_on_gpu(self, gpu_id: str) -> Optional[Job]:
        """Which job is running on this GPU?"""
        gpu = self.gpus.get(gpu_id)
        if gpu is None or gpu.assigned_job_id is None:
            return None
        return self.jobs.get(gpu.assigned_job_id)

    def get_waiting_jobs(self) -> List[Job]:
        """Which jobs are waiting for a GPU?"""
        return [job for job in self.jobs.values() if job.status == JobStatus.WAITING]

    def get_running_jobs(self) -> List[Job]:
        return [job for job in self.jobs.values() if job.status == JobStatus.RUNNING]

    def get_active_assignment_for_gpu(self, gpu_id: str) -> Optional[GPUAssignment]:
        """The current (not-yet-ended) assignment for a GPU, if any."""
        for assignment in self.assignments.values():
            if assignment.gpu_id == gpu_id and assignment.is_active:
                return assignment
        return None

    def get_events_for_gpu(self, gpu_id: str) -> List[Event]:
        """What events happened to this GPU, oldest first?"""
        return [event for event in self.events if event.gpu_id == gpu_id]

    def get_events_for_user(self, user_id: str) -> List[Event]:
        return [event for event in self.events if event.user_id == user_id]
