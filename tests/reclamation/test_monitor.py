from datetime import datetime, timedelta, timezone

from engine.reclamation.monitor import is_sustained_breach
from engine.models.utilization import UtilizationObservation

BASE = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
THRESHOLD = 15.0
DURATION = timedelta(minutes=20)


def obs(minute: float, percent: float) -> UtilizationObservation:
    return UtilizationObservation(utilization_percent=percent, timestamp=BASE + timedelta(minutes=minute))


def test_no_observations_is_not_a_breach():
    assert is_sustained_breach([], THRESHOLD, DURATION) is False


def test_insufficient_history_is_not_confirmed_sustained():
    # Only 5 minutes of (low) history when 20 are required.
    observations = [obs(0, 1.0), obs(5, 1.0)]
    assert is_sustained_breach(observations, THRESHOLD, DURATION) is False


def test_continuous_low_utilization_for_the_full_duration_is_sustained():
    observations = [obs(m, 1.0) for m in range(0, 21)]
    assert is_sustained_breach(observations, THRESHOLD, DURATION) is True


def test_brief_dip_that_recovers_is_not_sustained():
    # A few minutes of low utilization (a batch-loading pause), then a
    # genuine recovery well above the threshold for the rest of the
    # window. The dip alone would look identical to a real breach if
    # only those few readings were examined - the recovery covering
    # most of the required window is what correctly rules it out.
    observations = [obs(0, 10.0), obs(1, 9.0), obs(2, 4.0), obs(3, 1.0), obs(4, 8.0)]
    observations += [obs(m, 50.0) for m in range(5, 21)]
    assert is_sustained_breach(observations, THRESHOLD, DURATION) is False


def test_a_single_reading_at_the_threshold_breaks_the_streak():
    observations = [obs(m, 1.0) for m in range(0, 20)] + [obs(20, THRESHOLD)]
    assert is_sustained_breach(observations, THRESHOLD, DURATION) is False


def test_a_spike_that_happened_before_the_required_window_does_not_matter():
    # High utilization 25 minutes ago is outside the 20-minute lookback.
    observations = [obs(-25, 90.0)] + [obs(m, 1.0) for m in range(0, 21)]
    assert is_sustained_breach(observations, THRESHOLD, DURATION) is True


def test_boundary_exact_duration_counts_as_sustained():
    observations = [obs(0, 1.0), obs(20, 1.0)]
    assert is_sustained_breach(observations, THRESHOLD, DURATION) is True


def test_readings_exactly_at_threshold_are_a_breach_not_below_it():
    # ">= threshold" counts as NOT low enough - threshold itself is
    # the boundary of "below", not included in "below".
    observations = [obs(m, THRESHOLD) for m in range(0, 21)]
    assert is_sustained_breach(observations, THRESHOLD, DURATION) is False


def test_tier1_and_tier2_use_independent_thresholds_and_durations():
    # A GPU at 10% utilization breaches Tier 2 (< 15%) but not Tier 1 (< 2%).
    observations = [obs(m, 10.0) for m in range(0, 200)]
    assert is_sustained_breach(observations, 2.0, timedelta(minutes=25)) is False
    assert is_sustained_breach(observations, 15.0, timedelta(hours=2, minutes=30)) is True


def test_a_lone_recent_reading_backed_only_by_a_stale_pre_window_observation_is_not_sustained():
    """Audit-found regression: a monitoring gap must not be mistaken
    for sustained coverage just because *some* old reading happens to
    predate the window.

    Sequence: normal utilization, a brief dip, back to normal (all
    well before the window) - then silence for 28 minutes - then a
    single low reading. For a 25-minute window, that one reading is
    backed only by the stale pre-window data; it is not proof the GPU
    was actually low for the preceding 25 minutes.
    """
    observations = [
        obs(0, 70.0), obs(1, 5.0), obs(2, 70.0),  # a temporary dip, long resolved
        obs(30, 1.0),  # the only observation anywhere near "now" - a single snapshot
    ]
    assert is_sustained_breach(observations, THRESHOLD, DURATION) is False


def test_two_observations_actually_spanning_the_window_after_a_pre_window_gap_is_sustained():
    # Same shape as above, but this time the window itself (the last
    # 20 minutes before "now") is covered by two real readings, not
    # backed solely by stale data from outside it.
    observations = [
        obs(0, 70.0), obs(1, 5.0), obs(2, 70.0),
        obs(10, 1.0), obs(30, 1.0),  # both within [10, 30] - the actual 20-minute window
    ]
    assert is_sustained_breach(observations, THRESHOLD, DURATION) is True
