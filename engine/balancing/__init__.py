"""Phase 5: load balancing and dynamic task routing.

Answers "given a job Phase 3 already decided to schedule, which
currently-available GPU should receive it?" - routing *new* work
toward underutilized GPUs, never migrating or interrupting work that
is already legitimately running. Reclaiming a GPU because it looks
idle stays Phase 4's job; this package only ever chooses among GPUs
that are already, genuinely available.
"""

from engine.balancing.availability import has_valid_utilization, is_gpu_available
from engine.balancing.config import BalancingPolicy, DEFAULT_BALANCING_POLICY
from engine.balancing.decision import GPUCandidateInfo, RoutingDecision, RoutingOutcome
from engine.balancing.detector import is_pool_imbalanced, utilization_spread
from engine.balancing.router import LoadBalancingRouter

__all__ = [
    "has_valid_utilization",
    "is_gpu_available",
    "BalancingPolicy",
    "DEFAULT_BALANCING_POLICY",
    "GPUCandidateInfo",
    "RoutingDecision",
    "RoutingOutcome",
    "is_pool_imbalanced",
    "utilization_spread",
    "LoadBalancingRouter",
]
