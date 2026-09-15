from datetime import datetime, timedelta, timezone

import pytest

from engine.dsa.sliding_window import UtilizationSlidingWindow
from engine.models.utilization import UtilizationObservation

BASE = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


def obs_at(minutes: int, percent: float) -> UtilizationObservation:
    return UtilizationObservation(utilization_percent=percent, timestamp=BASE + timedelta(minutes=minutes))


def test_add_observations_within_window():
    window = UtilizationSlidingWindow(window_duration=timedelta(minutes=30))
    window.add_observation(obs_at(0, 2.0))
    window.add_observation(obs_at(1, 1.0))

    assert window.size() == 2
    assert [o.utilization_percent for o in window.observations()] == [2.0, 1.0]


def test_time_progression_expires_old_observations():
    window = UtilizationSlidingWindow(window_duration=timedelta(minutes=5))
    window.add_observation(obs_at(0, 2.0))
    window.add_observation(obs_at(1, 1.0))
    window.add_observation(obs_at(10, 3.0))  # 10 minutes later - outside the 5 min window

    remaining = window.observations()
    assert [o.utilization_percent for o in remaining] == [3.0]
    assert window.size() == 1


def test_maintaining_only_relevant_observations_as_stream_progresses():
    window = UtilizationSlidingWindow(window_duration=timedelta(minutes=3))
    for minute in range(10):
        window.add_observation(obs_at(minute, float(minute)))

    remaining = window.observations()
    # Only observations within the last 3 minutes of the latest (minute 9) survive.
    assert [o.utilization_percent for o in remaining] == [6.0, 7.0, 8.0, 9.0]


def test_boundary_timestamp_is_inclusive():
    window = UtilizationSlidingWindow(window_duration=timedelta(minutes=5))
    window.add_observation(obs_at(0, 1.0))
    # Exactly at the boundary (now - window_duration) should still be kept.
    window.evict_expired(now=BASE + timedelta(minutes=5))

    assert window.size() == 1


def test_boundary_timestamp_just_past_is_expired():
    window = UtilizationSlidingWindow(window_duration=timedelta(minutes=5))
    window.add_observation(obs_at(0, 1.0))
    window.evict_expired(now=BASE + timedelta(minutes=5, seconds=1))

    assert window.size() == 0


def test_window_span_reflects_oldest_and_newest():
    window = UtilizationSlidingWindow(window_duration=timedelta(minutes=30))
    window.add_observation(obs_at(0, 2.0))
    window.add_observation(obs_at(5, 1.0))
    window.add_observation(obs_at(12, 3.0))

    assert window.window_span() == timedelta(minutes=12)


def test_empty_window_has_no_span_or_latest():
    window = UtilizationSlidingWindow(window_duration=timedelta(minutes=30))
    assert window.window_span() is None
    assert window.latest() is None
    assert window.is_empty() is True


def test_rejects_non_positive_window_duration():
    with pytest.raises(ValueError):
        UtilizationSlidingWindow(window_duration=timedelta(0))
