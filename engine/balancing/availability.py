"""What "available" actually means, as one predicate - not five
different ad-hoc checks scattered across the balancing code.

This project deliberately keeps several related-but-different GPU
concepts distinct:

- **Utilization** - a raw number (`GPU.utilization_percent`). Says
  nothing about whether the GPU can be handed to someone else.
- **Assigned** - `GPU.assigned_user_id is not None`
  (`GPU.is_assigned`). A GPU can be assigned and nearly idle at the
  same time (kept as a memory buffer, an intermittent process, a
  user between computation stages) - that is exactly the scenario
  this phase's brief calls out as "do not steal this GPU".
- **Available** - genuinely free to hand to a new job right now. This
  module's `is_gpu_available`.
- **Reclaimable** - a *separate* judgment Phase 4 makes over time
  (sustained low utilization); this module does not decide it and
  does not read it. A GPU only becomes available *after* Phase 4
  actually reclaims it and the GPU model reflects that
  (unassigned + `GPUStatus.IDLE`) - never before.
- **Running** - `GPU.assigned_job_id is not None`; a GPU can be
  running and available at the same time is a contradiction this
  project's model does not allow (`GPU.is_assigned` covers both).

`GPUStatus.IDLE` is the only status this module treats as available.
`ACTIVE` (in use) and `IDLE_WARNING` (Phase 4 has a pending
confirmation prompt on it) are excluded because the GPU is still
assigned. `RECLAIMING` and `REALLOCATING` are excluded even though a
future engine might one day set them on an *unassigned* GPU mid
state-transition - a GPU is not safe to route new work onto until
that transition is actually finished and it is sitting plainly IDLE.
"""

import math

from engine.models.enums import GPUStatus
from engine.models.gpu import GPU


def is_gpu_available(gpu: GPU) -> bool:
    """Whether ``gpu`` may be routed a brand-new job right now.

    Both conditions must hold: nothing currently holds it
    (``not gpu.is_assigned``), and its status has actually settled at
    ``IDLE`` - not merely "not ACTIVE". ``ACTIVE``, ``IDLE_WARNING``,
    ``RECLAIMING`` and ``REALLOCATING`` are all excluded.
    """
    return (not gpu.is_assigned) and gpu.status == GPUStatus.IDLE


def has_valid_utilization(gpu: GPU) -> bool:
    """Whether ``gpu.utilization_percent`` is a number the router can
    safely compare and rank.

    A monitoring glitch (`NaN`, `inf`, a negative reading, a reading
    over 100) must never be allowed to silently win a "lowest
    utilization" comparison - `NaN` in particular compares as neither
    less nor greater than anything, which would make heap ordering
    (and `min`/`max`) behave inconsistently depending on insertion
    order. A GPU that fails this check is excluded from selection
    entirely rather than risk an unsafe routing decision.
    """
    value = gpu.utilization_percent
    return isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 100.0
