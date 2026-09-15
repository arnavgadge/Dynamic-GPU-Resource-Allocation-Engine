"""Load-imbalance detection over the *available* GPU pool.

Deliberately not `max(utilization) != min(utilization)` - the brief
is explicit that small, ordinary differences must not be treated as
"imbalanced". Imbalance is measured as the utilization spread among
GPUs that are actually available for new work, compared against the
single configured threshold in `engine/balancing/config.py`.
"""

from typing import Sequence

from engine.balancing.availability import has_valid_utilization
from engine.balancing.config import BalancingPolicy
from engine.models.gpu import GPU


def utilization_spread(gpus: Sequence[GPU]) -> float:
    """The gap, in percentage points, between the busiest and quietest
    GPU in ``gpus``. 0.0 for an empty or single-GPU sequence - there is
    nothing to spread across.

    GPUs with an invalid utilization reading (see
    `engine.balancing.availability.has_valid_utilization`) are ignored
    here for the same reason the router never selects one: a `NaN` or
    out-of-range value must never be allowed to silently participate
    in a `max`/`min` comparison.
    """
    valid = [gpu.utilization_percent for gpu in gpus if has_valid_utilization(gpu)]
    if len(valid) < 2:
        return 0.0
    return max(valid) - min(valid)


def is_pool_imbalanced(gpus: Sequence[GPU], policy: BalancingPolicy) -> bool:
    """Whether ``gpus`` (expected to already be filtered to available
    ones) shows a meaningful utilization spread under ``policy``.

    This does not change *what* gets selected - the router always
    prefers whichever available GPU is least utilized regardless of
    this result. It only controls whether that choice gets reported
    as "a real imbalance" (worth a note in the decision/event log) or
    just "the obvious pick among GPUs that were already close
    together".
    """
    return utilization_spread(gpus) >= policy.imbalance_threshold_percent
