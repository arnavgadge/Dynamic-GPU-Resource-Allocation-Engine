"""What the reclamation engine can tell you about a decision it made."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from engine.models.event import Event
from engine.reclamation.policy import ReclamationTier


class ReclamationAction(Enum):
    """What actually happened as a result of one reclamation decision."""

    #: A sustained breach was confirmed; the user was asked to confirm.
    PROMPTED = "PROMPTED"
    #: The user answered "yes, still using it" - the timer was reset.
    BACKED_OFF = "BACKED_OFF"
    #: The GPU was actually taken back (a "no", or silence past the
    #: grace period, or - not used by this engine, but available for a
    #: caller - an explicit reclaim).
    RECLAIMED = "RECLAIMED"


@dataclass
class ReclamationDecision:
    """A complete, explainable record of one reclamation-related action.

    Mirrors `engine.allocation.decision.AllocationDecision`'s role for
    Phase 3: enough information for a viva - or a future frontend
    event log - to justify what happened, without having to re-derive
    it from raw utilization numbers. ``event`` is the same `Event`
    that was also written to `SchedulerState`.
    """

    timestamp: datetime
    gpu_id: str
    tier: Optional[ReclamationTier]
    action: ReclamationAction
    reason: str
    event: Event

    #: Set only for a resource-request prompt (Phase 10's "another
    #: user's request needs this GPU" flow, `ReclamationEngine.
    #: request_gpu_for_reallocation`) - who is asking, and for which
    #: waiting job. ``None`` for every ordinary utilization-tier or
    #: estimated-completion prompt, which are not made on anyone's
    #: behalf.
    requesting_user_id: Optional[str] = None
    requesting_job_id: Optional[str] = None
