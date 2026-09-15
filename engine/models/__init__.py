"""Domain model package for the GPU Resource Allocation Engine.

Everything in this package is a plain data model: enums and
dataclasses describing GPUs, users, jobs, assignments, events and the
overall scheduler state. Nothing here decides *how* GPUs should be
allocated - that is the job of the scheduling engine built in later
phases on top of these models.
"""

from engine.models.enums import (
    EngineStatus,
    EventType,
    GPUStatus,
    JobStatus,
    Priority,
)
from engine.models.utilization import UtilizationObservation
from engine.models.gpu import GPU
from engine.models.user import User
from engine.models.job import Job
from engine.models.assignment import GPUAssignment
from engine.models.event import Event
from engine.models.scheduler_state import SchedulerState

__all__ = [
    "EngineStatus",
    "EventType",
    "GPUStatus",
    "JobStatus",
    "Priority",
    "UtilizationObservation",
    "GPU",
    "User",
    "Job",
    "GPUAssignment",
    "Event",
    "SchedulerState",
]
