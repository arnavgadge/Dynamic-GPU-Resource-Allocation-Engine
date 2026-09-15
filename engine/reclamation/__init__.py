"""Phase 4: the reclamation engine.

Detects a GPU that has been genuinely idle for a sustained period
(not just briefly quiet) and, through a confirm-then-reclaim flow,
returns it to the shared pool via `AllocationEngine.mark_gpu_available`
- the same pool every other GPU comes from. There are no GPU leases
anywhere in this module, and nothing here touches load balancing,
the frontend, or real GPU monitoring integration.
"""

from engine.reclamation.decision import ReclamationAction, ReclamationDecision
from engine.reclamation.engine import ReclamationEngine
from engine.reclamation.monitor import is_sustained_breach
from engine.reclamation.policy import (
    DEFAULT_RECLAMATION_POLICY,
    ConfirmationResponse,
    ReclamationPolicy,
    ReclamationTier,
    ReclamationTierPolicy,
)

__all__ = [
    "ReclamationAction",
    "ReclamationDecision",
    "ReclamationEngine",
    "is_sustained_breach",
    "DEFAULT_RECLAMATION_POLICY",
    "ConfirmationResponse",
    "ReclamationPolicy",
    "ReclamationTier",
    "ReclamationTierPolicy",
]
