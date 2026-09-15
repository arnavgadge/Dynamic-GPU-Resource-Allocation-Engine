"""Sustained-breach detection over a GPU's utilization history.

Answers exactly one question: has utilization stayed continuously
below a threshold for at least a given duration, right up through the
most recent observation? A brief dip that recovers must NOT count -
only a breach that is still ongoing *right now*, and has already
lasted at least `duration`, counts as "sustained". This module makes
no reclamation decision itself - it only tells the engine whether a
tier's condition currently holds.
"""

from datetime import timedelta
from typing import Sequence

from engine.models.utilization import UtilizationObservation


def is_sustained_breach(
    observations: Sequence[UtilizationObservation],
    threshold_percent: float,
    duration: timedelta,
) -> bool:
    """Whether utilization has been below ``threshold_percent`` for at
    least ``duration``, continuously, ending at the latest observation.

    Three conditions all have to hold, checked against the window
    ``[latest.timestamp - duration, latest.timestamp]``:

    1. **Some history exists further back than the window.** If the
       *earliest observation ever recorded* is more recent than
       ``latest.timestamp - duration``, monitoring plainly hasn't been
       running long enough yet to confirm the full duration.
    2. **The window itself is actually covered by more than one
       reading.** A single observation sitting inside the window,
       backed only by an old, otherwise-irrelevant reading from *before*
       the window started, is not evidence of sustained anything - it
       is one snapshot. Concretely: if there was a monitoring gap that
       swallowed the transition into the window (e.g. a reading at
       T-40min, then silence, then the next reading only at T-2min for
       a 25-minute window), condition 1 alone would be satisfied by
       the stale T-40min point even though the window contains just
       one real sample. Requiring at least two observations inside the
       window closes that gap - see `test_a_lone_recent_reading_
       backed_only_by_a_stale_pre_window_observation_is_not_sustained`.
    3. **No observation in that window is at or above the threshold.**
       A single reading back at/above the threshold - even one, even
       for a moment - breaks the streak and resets what "sustained"
       means going forward.

    ``observations`` must be sorted oldest-first (exactly what
    `UtilizationSlidingWindow.observations()` returns).

    This is still not a perfect model of "monitoring never gapped" -
    two observations clustered right next to each other at the very
    end of the window would technically satisfy condition 2 without
    truly covering the window either. It is a deliberately minimal
    guard against the concrete false-positive described above, not a
    general-purpose gap detector; see the README's Phase 9 audit
    section for why a fuller fix (an explicit expected-sampling-rate
    parameter) was left out.
    """
    if not observations:
        return False

    latest_timestamp = observations[-1].timestamp
    window_start = latest_timestamp - duration

    if observations[0].timestamp > window_start:
        return False

    in_window = [o for o in observations if o.timestamp >= window_start]
    if len(in_window) < 2:
        return False

    for observation in in_window:
        if observation.utilization_percent >= threshold_percent:
            return False

    return True
