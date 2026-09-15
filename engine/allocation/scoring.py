"""The project's allocation-score formula.

    Allocation Score = (PRIORITY_WEIGHT x normalized priority)
                      + (SIZE_WEIGHT    x job size inverse)

Both components are normalized onto ``[0, 1]`` *before* the formula
weighs them, for two different reasons:

- **Priority** already has a small, fixed set of levels
  (`Priority.LOW` .. `Priority.CRITICAL`, i.e. 1..4). Normalizing
  against those fixed bounds turns "priority level" into "how
  priority this job is, as a fraction of the highest possible
  priority" - a stable meaning that does not depend on which other
  jobs happen to be waiting right now.

- **Job size** has no such fixed bound - a job can reasonably be 5
  minutes or 5 days, and those units are only meaningful relative to
  each other. So job size is normalized *relative to the other
  candidates in the current decision* (min-max normalization): the
  smallest candidate scores 1.0 on size, the largest scores 0.0, and
  everything else falls linearly in between. This is what keeps the
  formula numerically safe - see `job_size_inverse` below.

Neither normalization ever divides by a quantity that can be zero
except the case that is explicitly guarded (`max_size <= min_size`),
so this module cannot produce a divide-by-zero, NaN, or infinity.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from engine.allocation.config import AGING_MAX_CONTRIBUTION, AGING_RATE_PER_MINUTE, PRIORITY_WEIGHT, SIZE_WEIGHT
from engine.models.enums import Priority
from engine.models.job import Job

_MIN_PRIORITY_VALUE = Priority.LOW.value
_MAX_PRIORITY_VALUE = Priority.CRITICAL.value
_PRIORITY_SPAN = _MAX_PRIORITY_VALUE - _MIN_PRIORITY_VALUE  # fixed, always 3 - never zero


@dataclass(frozen=True)
class ScoreBreakdown:
    """Every term behind one `calculate_allocation_score` result, for
    the decision trace (Phase 4 of the 100-scenario fix set) - never
    recomputed or re-derived by a caller; this *is* the computation.
    """

    priority_component: float
    size_component: float
    aging_component: float
    base_score: float   # priority_component*WEIGHT + size_component*WEIGHT, before aging
    final_score: float  # base_score + aging_component
    waiting_minutes: float


def normalized_priority(priority: Priority) -> float:
    """Priority as a fraction of the highest possible priority level.

    ``LOW`` -> 0.0, ``CRITICAL`` -> 1.0, evenly spaced in between.
    The span (`_PRIORITY_SPAN`) is a compile-time constant derived
    from the `Priority` enum's own bounds, so this can never divide
    by zero.
    """
    return (priority.value - _MIN_PRIORITY_VALUE) / _PRIORITY_SPAN


def calculate_priority_component(priority: Priority) -> float:
    """Named alias of `normalized_priority` - the allocation score's
    priority term, exposed under the name the decision trace/Phase 4
    refactor calls for. Kept as a thin wrapper (not a duplicate
    implementation) so there is exactly one normalization rule."""
    return normalized_priority(priority)


def job_size_inverse(size_minutes: float, min_size_minutes: float, max_size_minutes: float) -> float:
    """How small ``size_minutes`` is, relative to the current candidates.

    Returns 1.0 for the smallest candidate, 0.0 for the largest, and
    a linear interpolation in between - deliberately *not*
    ``1 / size_minutes``, which would blow up (division by zero,
    or a meaningless huge number) for a very small or zero-length
    job, and whose scale would depend on whatever unit "size" happens
    to be measured in.

    If every candidate is the same size (``max_size_minutes <=
    min_size_minutes``), there is nothing to differentiate on, so
    every candidate gets the maximum size score (1.0) - being "the
    smallest" is not a meaningful distinction when everyone is
    the same size.

    ``size_minutes`` is clamped into ``[min_size_minutes,
    max_size_minutes]`` first, so a pathological or invalid value
    (zero, negative) simply behaves as "at least as small as the
    smallest real candidate" instead of producing an out-of-range
    result.
    """
    if max_size_minutes <= min_size_minutes:
        return 1.0
    clamped = min(max(size_minutes, min_size_minutes), max_size_minutes)
    return (max_size_minutes - clamped) / (max_size_minutes - min_size_minutes)


def calculate_size_component(size_minutes: float, min_size_minutes: float, max_size_minutes: float) -> float:
    """Named alias of `job_size_inverse` - the allocation score's size
    term, under the Phase 4 naming convention."""
    return job_size_inverse(size_minutes, min_size_minutes, max_size_minutes)


def calculate_aging_component(waiting_minutes: float) -> float:
    """Starvation-prevention bonus for how long a job has already
    waited (Phase 2 of the 100-scenario fix set).

    Grows linearly at `AGING_RATE_PER_MINUTE` per minute waited, capped
    at `AGING_MAX_CONTRIBUTION` - both plain, configurable constants
    in `engine/allocation/config.py`, where the exact reasoning behind
    the cap's value (deliberately above the maximum possible base
    score of 1.0) is documented. A job that has waited long enough
    always eventually out-scores any possible fresh arrival, which is
    what actually prevents indefinite starvation by a continuous
    stream of new work - deterministically, not just usually. A job
    that has *just* started waiting gets essentially no bonus, so
    aging never overturns a fresh priority/size decision immediately.
    """
    if waiting_minutes <= 0:
        return 0.0
    return min(waiting_minutes * AGING_RATE_PER_MINUTE, AGING_MAX_CONTRIBUTION)


def calculate_allocation_score(
    job: Job, min_size_minutes: float, max_size_minutes: float, now: Optional[datetime] = None,
) -> ScoreBreakdown:
    """The project's full weighted allocation score, with every term
    broken out (Phase 4) - this is the one real computation; nothing
    else in the codebase re-derives priority/size/aging arithmetic.

    ``min_size_minutes``/``max_size_minutes`` must come from the same
    set of candidates being compared in this decision - the score is
    only meaningful relative to that set, not as an absolute number
    comparable across different decisions. ``now`` defaults to the
    job's own `submitted_at` (zero wait) when omitted, so a caller
    that genuinely has no clock available still gets a valid,
    zero-aging score rather than an error.
    """
    priority_component = calculate_priority_component(job.priority)
    size_component = calculate_size_component(job.estimated_size_minutes, min_size_minutes, max_size_minutes)
    base_score = PRIORITY_WEIGHT * priority_component + SIZE_WEIGHT * size_component

    reference_now = now if now is not None else job.submitted_at
    waiting_minutes = max((reference_now - job.submitted_at).total_seconds() / 60.0, 0.0)
    aging_component = calculate_aging_component(waiting_minutes)

    return ScoreBreakdown(
        priority_component=priority_component,
        size_component=size_component,
        aging_component=aging_component,
        base_score=base_score,
        final_score=base_score + aging_component,
        waiting_minutes=waiting_minutes,
    )


def allocation_score(
    job: Job, min_size_minutes: float, max_size_minutes: float, now: Optional[datetime] = None,
) -> float:
    """Backward-compatible entry point: the final (aging-inclusive)
    score as a plain float, for every existing call site that only
    ever wanted the one number. `calculate_allocation_score` above is
    the real computation and the one to use when the components
    themselves (for a decision trace) matter."""
    return calculate_allocation_score(job, min_size_minutes, max_size_minutes, now).final_score
