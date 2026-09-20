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

    @property
    def label(self) -> str:
        """Human-readable scheduling mode for a decision trace."""
        return "FCFS" if self is AllocationPolicy.FCFS else "SJF / Weighted"


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

    #: 1 = earliest-submitted among the candidates of this decision
    #: (ties keep queue order) - the FCFS ordering, shown in the
    #: trace for both modes so a reader can see arrival order next to
    #: the score that may have overridden it.
    arrival_position: Optional[int] = None


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

    #: The relative size spread of the candidates and the threshold it
    #: was judged against (Day 6) - the numbers behind the FCFS vs.
    #: SJF/weighted choice, structured rather than only inside `reason`.
    size_spread: Optional[float] = None
    similarity_threshold: Optional[float] = None

    def explain(self) -> str:
        """The decision as a readable trace: mode, why, every
        candidate (arrival order, priority, size, wait, score
        breakdown), and the winner."""
        lines = [f"Policy: {self.policy.label}"]
        if self.size_spread is not None and self.similarity_threshold is not None:
            relation = "within" if self.size_spread <= self.similarity_threshold else "beyond"
            lines.append(
                f"Size spread: {self.size_spread:.1%} ({relation} the {self.similarity_threshold:.0%} threshold)"
            )
        lines.append(f"Reason: {self.reason}")
        lines.append("Candidates:")
        for c in sorted(self.candidates, key=lambda c: c.arrival_position or 0):
            if c.score is None:
                score = "score n/a (FCFS - not computed)"
            else:
                score = (
                    f"score {c.score:.3f} = base {c.base_score:.3f} (priority component "
                    f"{c.priority_component:.3f}, size component {c.size_component:.3f}) "
                    f"+ aging {c.aging_component:.3f}"
                )
            lines.append(
                f"  #{c.arrival_position} {c.job_id} user={c.user_id} {c.priority.name} "
                f"size={c.size_minutes:g}min waited={c.waiting_time} {score}"
            )
        lines.append(f"Selected: {self.job_id} -> {self.gpu_id}")
        return "\n".join(lines)
