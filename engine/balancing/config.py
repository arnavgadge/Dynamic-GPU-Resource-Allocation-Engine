"""The load-balancing policy's one configurable value: how big a
utilization spread among *available* GPUs counts as a meaningful
imbalance worth calling out, rather than ordinary noise.

Kept in one place, like every other phase's policy constants
(`engine/allocation/config.py`, `engine/reclamation/policy.py`), so
"how different is different enough" is never re-typed as a literal
somewhere else in the balancing code.
"""

from dataclasses import dataclass, field
from datetime import timedelta


@dataclass(frozen=True)
class BalancingPolicy:
    """Configuration for load-imbalance detection and cross-user
    resource-request hysteresis (100-scenario validation, Phase 12)."""

    #: Minimum utilization-percentage-point spread among currently
    #: *available* GPUs before the pool is considered meaningfully
    #: imbalanced. Below this, differences are treated as normal
    #: variation - the router still always prefers the least-utilized
    #: available GPU either way; this threshold only controls whether
    #: that choice gets flagged/logged as "a real imbalance" or not.
    imbalance_threshold_percent: float

    #: Minimum real time that must pass after a resource-request/
    #: preemption prompt on a GPU is *resolved* (YES or NO) before
    #: that same GPU can be asked again for a *different* job. This is
    #: the actual anti-thrashing guard: utilization crossing a
    #: threshold back and forth (4% -> 8% -> 5% -> 7% -> 4%) cannot by
    #: itself cause repeated reallocation asks on the same GPU, because
    #: `Scheduler._request_additional_gpus_if_needed` checks this
    #: cooldown before ever raising a new ask - see `ReclamationEngine.
    #: is_in_cooldown`. A *different* GPU is never affected by another
    #: GPU's cooldown.
    preemption_cooldown: timedelta = field(default_factory=lambda: timedelta(minutes=10))


#: Default: a 30 percentage-point spread (e.g. 5% vs. 40%+) between
#: the busiest and quietest *available* GPU is treated as a real
#: imbalance worth noting. Smaller gaps (a 20% GPU next to a 25% one)
#: are ordinary variation, not something the system should call
#: special attention to - the brief explicitly warns against treating
#: `max != min` itself as "imbalanced". Like every other policy value
#: in this project, it is a constructor argument, not a hardcoded
#: constant. `preemption_cooldown` defaults to 10 minutes - long enough
#: that a GPU whose utilization is merely oscillating near a threshold
#: cannot be re-asked for on every tick, short enough that a genuinely
#: still-needed reallocation isn't stuck waiting unreasonably long.
DEFAULT_BALANCING_POLICY = BalancingPolicy(imbalance_threshold_percent=30.0)
