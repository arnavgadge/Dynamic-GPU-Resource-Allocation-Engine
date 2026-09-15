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
