"""What the load-balancing router can tell you about a routing decision."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from engine.models.event import Event


class RoutingOutcome(Enum):
    """What the router actually did with a job it was asked to place."""

    #: A genuinely available GPU was found and selected.
    ROUTED = "ROUTED"
    #: No GPU in the pool was available; the job is left for the
    #: caller to keep as WAITING - the router never invents a fake
    #: allocation.
    NO_GPU_AVAILABLE = "NO_GPU_AVAILABLE"


@dataclass(frozen=True)
class GPUCandidateInfo:
    """One GPU's numbers, as they were considered for one routing decision."""

    gpu_id: str
    utilization_percent: float
    available: bool


@dataclass
class RoutingDecision:
    """A complete, explainable record of one "where should this job go"
    decision - `engine.allocation.decision.AllocationDecision`'s
    counterpart for "which GPU" instead of "which job".

    ``candidates`` lists every GPU the router looked at, available or
    not - useful for a viva or a frontend event log to show *why* an
    apparently-idle GPU (e.g. one at 4% but already assigned) was
    correctly skipped, not just which GPU won.
    """

    timestamp: datetime
    job_id: str
    candidates: List[GPUCandidateInfo]
    available_candidates: List[GPUCandidateInfo]
    selected_gpu_id: Optional[str]
    imbalance_detected: bool
    outcome: RoutingOutcome
    reason: str
    event: Event


class ReallocationPath(Enum):
    """Which of `Scheduler._request_additional_gpus_if_needed`'s two
    reallocation paths a `LoadBalancingTrace` is for (Day 8) - never a
    second scheduling policy, just which existing eligibility rule was
    being applied when the candidates below were evaluated."""

    #: A multi-GPU request asking another user's *underutilized* GPU.
    EXCESS_CAPACITY = "EXCESS_CAPACITY"
    #: A higher-priority arrival asking a genuinely lower-priority holder.
    PRIORITY_PREEMPTION = "PRIORITY_PREEMPTION"


@dataclass(frozen=True)
class LoadBalancingCandidate:
    """One already-assigned GPU's numbers, as considered for one
    reallocation ask (Day 8) - the same idea as `GPUCandidateInfo`,
    for the "ask another user to release theirs" path rather than the
    router's "place new work" path. Purely observational: recorded
    *after* the real eligibility check every ask already makes, never
    a second copy of that logic.
    """

    gpu_id: str
    utilization_percent: float
    holder_user_id: Optional[str]
    eligible: bool
    #: Why this candidate was *not* asked, or `None` if it was eligible.
    #: One of: "not assigned", "held by the requester", "prompt already
    #: pending", "already declined for this job", "in cooldown", "not
    #: underutilized enough" (path EXCESS_CAPACITY only), "priority not
    #: outranked" / "does not out-score the current holder" (path
    #: PRIORITY_PREEMPTION only).
    skip_reason: Optional[str]
    #: Whether a resource-request/preemption cooldown is currently
    #: active on this GPU, regardless of whether that is *why* it was
    #: skipped (surfaced separately - see `docs/dsa.md` for why
    #: cooldown deliberately never affects `RoutingDecision`, only this
    #: reallocation path).
    in_cooldown: bool
    selected: bool


@dataclass(frozen=True)
class LoadBalancingTrace:
    """A complete, explainable record of one reallocation ask made on
    behalf of one waiting job's remaining GPU deficit - candidates,
    eligibility, cooldown status, and which (if any) were actually
    asked. `Scheduler.last_balancing_traces` holds every trace built
    during the most recent `try_allocate_all` call.
    """

    timestamp: datetime
    job_id: str
    path: ReallocationPath
    deficit: int
    candidates: List[LoadBalancingCandidate]
    selected_gpu_ids: List[str]
    reason: str
