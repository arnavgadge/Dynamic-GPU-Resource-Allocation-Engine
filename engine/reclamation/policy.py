"""The project's fixed reclamation policy: two utilization tiers plus
the confirmation-prompt grace period, defined once and configurable.

Every threshold/duration a reclamation decision depends on lives
here - `monitor.py` and `engine.py` import these values rather than
repeating the numbers, exactly like Phase 3's `allocation/config.py`
centralizes the allocation-score weights and the 20% threshold.
"""

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum


class ReclamationTier(Enum):
    """Which of the project's two reclamation conditions applies.

    Tier 1 (very low utilization, short sustained window) and Tier 2
    (moderately low utilization, long sustained window) are checked
    in that order - see `ReclamationEngine._detect_tier` - because
    Tier 1 is the stronger, faster signal: anything quiet enough to
    trip Tier 1 is, by definition, also under Tier 2's higher
    threshold, so there is no point waiting for Tier 2's much longer
    window once Tier 1 has already fired.
    """

    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"


@dataclass(frozen=True)
class ReclamationTierPolicy:
    """One tier's threshold, required sustained duration, and what a
    breach of it is interpreted to mean."""

    tier: ReclamationTier
    label: str
    utilization_threshold_percent: float
    sustained_duration: timedelta


@dataclass(frozen=True)
class ReclamationPolicy:
    """The complete, configurable reclamation policy."""

    tier1: ReclamationTierPolicy
    tier2: ReclamationTierPolicy

    #: How long a confirmation prompt ("are you still using this GPU?")
    #: waits for a response before being treated the same as a "no".
    no_response_grace_period: timedelta

    @property
    def monitoring_window_duration(self) -> timedelta:
        """How much utilization history must be kept per GPU.

        The larger of the two tiers' sustained durations - a single
        `UtilizationSlidingWindow` sized to this covers both tiers'
        checks, so there is no need to keep two separate windows (and
        therefore two copies of the same observations) per GPU.
        """
        return max(self.tier1.sustained_duration, self.tier2.sustained_duration)


class ConfirmationResponse(Enum):
    """A user's answer to "are you still using this GPU?"."""

    YES = "YES"
    NO = "NO"


#: Default policy, with values chosen at the midpoint of each range the
#: project brief specifies:
#:
#: - Tier 1: below 2% utilization, sustained 20-30 minutes -> 25 minutes.
#: - Tier 2: below 15% utilization, sustained 2-3 hours -> 2 hours 30 minutes.
#: - No-response grace period: 5 minutes. The brief does not pin this one
#:   to a range; 5 minutes is a reasonable window for a user to notice and
#:   answer a prompt without leaving a GPU sitting idle-but-unreclaimed
#:   for long, and - like every other value here - it is a constructor
#:   argument, not a hardcoded constant, so a demo can shrink it to
#:   seconds without touching any other code.
DEFAULT_RECLAMATION_POLICY = ReclamationPolicy(
    tier1=ReclamationTierPolicy(
        tier=ReclamationTier.TIER_1,
        label="likely completed job",
        utilization_threshold_percent=2.0,
        sustained_duration=timedelta(minutes=25),
    ),
    tier2=ReclamationTierPolicy(
        tier=ReclamationTier.TIER_2,
        label="likely inactive user",
        utilization_threshold_percent=15.0,
        sustained_duration=timedelta(hours=2, minutes=30),
    ),
    no_response_grace_period=timedelta(minutes=5),
)
