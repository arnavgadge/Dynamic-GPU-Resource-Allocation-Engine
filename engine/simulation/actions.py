"""Scenario actions: generic, declarative inputs a scenario schedules
at a simulated time offset.

Each action describes *what input to feed the real Scheduler, and
when* - never what should happen as a result. `Simulator._execute`
turns an action into exactly one call against `Scheduler` (which
itself only calls into the real Allocation/Reclamation/Balancing
engines); no action type here is allowed to pick a job, pick a GPU,
or decide a reclamation outcome. That is the line this project's "no
fake decisions" rule draws, and every action below stays on the
correct side of it.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Union

from engine.models.job import Job
from engine.reclamation.policy import ConfirmationResponse


@dataclass(frozen=True)
class AddJobAction:
    """At ``offset``, submit ``job`` to the scheduler's waiting queue."""

    offset: timedelta
    job: Job


@dataclass(frozen=True)
class UtilizationAction:
    """At ``offset``, report one utilization reading for a GPU.

    This is the *only* way a scenario supplies utilization data - it
    never tells the Reclamation Engine what that reading means; it
    just reports the number, exactly like a real monitoring layer would.
    """

    offset: timedelta
    gpu_id: str
    utilization_percent: float
    memory_used_mb: Optional[float] = None


@dataclass(frozen=True)
class UserResponseAction:
    """At ``offset``, answer a pending reclamation confirmation prompt."""

    offset: timedelta
    gpu_id: str
    response: ConfirmationResponse


@dataclass(frozen=True)
class CompleteJobAction:
    """At ``offset``, mark a running job as finished (via
    `Scheduler.complete_job`), releasing its GPU back to the pool."""

    offset: timedelta
    job_id: str


#: Every action type the simulator knows how to execute - as a tuple,
#: so `isinstance(action, ScenarioActionTypes)` can validate a
#: scenario's action list, and as a `Union` for type hints.
ScenarioActionTypes = (AddJobAction, UtilizationAction, UserResponseAction, CompleteJobAction)
ScenarioAction = Union[AddJobAction, UtilizationAction, UserResponseAction, CompleteJobAction]
