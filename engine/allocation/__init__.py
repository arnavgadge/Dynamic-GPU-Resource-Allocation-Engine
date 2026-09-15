"""Phase 3: the allocation engine.

The first phase allowed to make a decision - "which waiting job gets
this available GPU?" - under the project's fixed policy (allocation
score, 20% size-similarity threshold, FCFS, critical-job precedence).
Everything here is built on Phase 1's models and Phase 2's DSA
structures; nothing here reclaims a GPU, load-balances, or leases one.
"""

from engine.allocation.config import (
    JOB_SIZE_SIMILARITY_THRESHOLD,
    PRIORITY_WEIGHT,
    SIZE_WEIGHT,
)
from engine.allocation.decision import AllocationDecision, AllocationPolicy, CandidateInfo
from engine.allocation.engine import AllocationEngine
from engine.allocation.scoring import allocation_score, job_size_inverse, normalized_priority
from engine.allocation.similarity import relative_size_spread, sizes_are_similar

__all__ = [
    "PRIORITY_WEIGHT",
    "SIZE_WEIGHT",
    "JOB_SIZE_SIMILARITY_THRESHOLD",
    "AllocationDecision",
    "AllocationPolicy",
    "CandidateInfo",
    "AllocationEngine",
    "allocation_score",
    "job_size_inverse",
    "normalized_priority",
    "relative_size_spread",
    "sizes_are_similar",
]
