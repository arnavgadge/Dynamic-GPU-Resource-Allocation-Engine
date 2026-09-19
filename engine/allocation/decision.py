"""What the allocation engine can tell you about a decision it made."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from engine.models.enums import Priority
from engine.models.event import Event


class AllocationPolicy(Enum):
    """Which of the project's two candidate-selection behaviors was used.

    Exactly the two policies the project defines - there is no third
    "default" or "fallback" policy. When only one job is eligible,
    that is still reported as ``FCFS`` (trivially: the only candidate
    has, by definition, waited the longest of the eligible set).
    """

    FCFS = "FCFS"
    SCORE_BASED = "SCORE_BASED"


@dataclass(frozen=True)
class CandidateInfo:
    """One job's numbers, as they were considered for one decision.

    ``score`` is ``None`` when the decision used FCFS - the project's
    score formula was never computed for that job in that round, so
    reporting a number here would imply it influenced a decision it
    did not.
    """

    job_id: str
    user_id: str
    priority: Priority
    size_minutes: float
    waiting_time: timedelta
    score: Optional[float]

    #: Score breakdown (Phase 4 of the 100-scenario fix set) - all
    #: `None` together whenever `score` is (FCFS never computed any of
    #: this); otherwise `base_score + aging_component == score`
    #: exactly. Exposed so a decision trace can show the real
    #: computation instead of just its final number.
    base_score: Optional[float] = None
    aging_component: Optional[float] = None
    waiting_minutes: Optional[float] = None

    #: The two normalized terms behind `base_score` (Day 5) - so a
    #: candidate's full explanation (priority component, size
    #: component, aging, final score) is available on the decision
    #: itself, not only via the separate `ScoreBreakdown`. `None`
    #: whenever `score` is, for the same reason.
    priority_component: Optional[float] = None
    size_component: Optional[float] = None


@dataclass
class AllocationDecision:
    """A complete, explainable record of one GPU being handed to one job.

    Everything a viva (or a future frontend event log) would need to
    justify the outcome: which GPU, which job won, every candidate
    that was considered and their numbers, which policy decided it,
    and a human-readable reason. `engine` is the `Event` that was
    also written to `SchedulerState` for this same decision.
    """

    timestamp: datetime
    gpu_id: str
    job_id: str
    user_id: str
    policy: AllocationPolicy
    reason: str
    candidates: List[CandidateInfo]
    event: Event
