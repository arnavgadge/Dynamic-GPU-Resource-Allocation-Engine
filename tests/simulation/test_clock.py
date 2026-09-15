from datetime import datetime, timedelta, timezone

import pytest

from engine.simulation.clock import SimulationClock

START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def test_clock_starts_at_the_given_time():
    clock = SimulationClock(start=START)
    assert clock.now() == START


def test_advance_moves_time_forward_by_exactly_the_given_delta():
    clock = SimulationClock(start=START)
    clock.advance(timedelta(minutes=20))
    assert clock.now() == START + timedelta(minutes=20)

    clock.advance(timedelta(hours=3))
    assert clock.now() == START + timedelta(minutes=20) + timedelta(hours=3)


def test_advance_never_sleeps_and_handles_large_durations_instantly():
    import time as real_time

    clock = SimulationClock(start=START)
    started = real_time.monotonic()
    clock.advance(timedelta(hours=3))
    elapsed_real_seconds = real_time.monotonic() - started

    assert clock.now() == START + timedelta(hours=3)
    assert elapsed_real_seconds < 1.0  # nowhere close to 3 real hours


def test_advance_rejects_negative_delta():
    clock = SimulationClock(start=START)
    with pytest.raises(ValueError):
        clock.advance(timedelta(minutes=-1))


def test_set_moves_directly_to_a_later_time():
    clock = SimulationClock(start=START)
    target = START + timedelta(hours=1)
    clock.set(target)
    assert clock.now() == target


def test_set_rejects_moving_backwards():
    clock = SimulationClock(start=START)
    clock.advance(timedelta(minutes=10))
    with pytest.raises(ValueError):
        clock.set(START)


def test_reset_returns_to_the_original_start_time():
    clock = SimulationClock(start=START)
    clock.advance(timedelta(hours=5))
    clock.reset()
    assert clock.now() == START


def test_speed_must_be_positive():
    with pytest.raises(ValueError):
        SimulationClock(start=START, speed=0)
    with pytest.raises(ValueError):
        SimulationClock(start=START, speed=-1)
