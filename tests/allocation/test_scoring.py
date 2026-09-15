import math

import pytest

from engine.allocation.config import PRIORITY_WEIGHT, SIZE_WEIGHT
from engine.allocation.scoring import allocation_score, job_size_inverse, normalized_priority
from engine.models.enums import Priority
from engine.models.job import Job


def make_job(size_minutes: float, priority: Priority = Priority.MEDIUM) -> Job:
    return Job(job_id="J", user_id="U", name="job", priority=priority, estimated_size_minutes=size_minutes)


# -- normalized_priority --------------------------------------------

def test_normalized_priority_spans_zero_to_one():
    assert normalized_priority(Priority.LOW) == 0.0
    assert normalized_priority(Priority.CRITICAL) == 1.0


def test_normalized_priority_is_evenly_spaced():
    medium = normalized_priority(Priority.MEDIUM)
    high = normalized_priority(Priority.HIGH)
    assert medium == pytest.approx(1 / 3)
    assert high == pytest.approx(2 / 3)


# -- job_size_inverse -------------------------------------------------

def test_smallest_job_gets_maximum_size_score():
    assert job_size_inverse(10, min_size_minutes=10, max_size_minutes=100) == 1.0


def test_largest_job_gets_minimum_size_score():
    assert job_size_inverse(100, min_size_minutes=10, max_size_minutes=100) == 0.0


def test_midpoint_job_gets_half_size_score():
    assert job_size_inverse(55, min_size_minutes=10, max_size_minutes=100) == pytest.approx(0.5)


def test_identical_sizes_all_score_maximum():
    # No differentiation possible -> nobody is penalized for "being large".
    assert job_size_inverse(50, min_size_minutes=50, max_size_minutes=50) == 1.0


def test_zero_size_does_not_raise_and_is_bounded():
    result = job_size_inverse(0, min_size_minutes=10, max_size_minutes=100)
    assert 0.0 <= result <= 1.0
    assert math.isfinite(result)


def test_negative_size_is_clamped_not_a_crash():
    result = job_size_inverse(-50, min_size_minutes=10, max_size_minutes=100)
    assert result == 1.0  # clamped to the minimum -> treated as "smallest"
    assert math.isfinite(result)


def test_never_divides_by_zero_when_min_equals_max():
    # max_size_minutes <= min_size_minutes is the only branch that could
    # divide by zero; it must be fully guarded.
    assert job_size_inverse(5, min_size_minutes=5, max_size_minutes=5) == 1.0
    assert job_size_inverse(5, min_size_minutes=5, max_size_minutes=0) == 1.0


# -- allocation_score --------------------------------------------------

def test_allocation_score_is_weighted_combination():
    job = make_job(size_minutes=10, priority=Priority.HIGH)
    score = allocation_score(job, min_size_minutes=10, max_size_minutes=100)

    expected = PRIORITY_WEIGHT * normalized_priority(Priority.HIGH) + SIZE_WEIGHT * 1.0
    assert score == pytest.approx(expected)


def test_allocation_score_is_bounded_between_zero_and_one():
    for priority in Priority:
        for size in (0, 1, 60, 10_000):
            job = make_job(size_minutes=size, priority=priority)
            score = allocation_score(job, min_size_minutes=0, max_size_minutes=10_000)
            assert 0.0 <= score <= 1.0
            assert math.isfinite(score)


def test_higher_priority_increases_score_at_equal_size():
    low = make_job(size_minutes=50, priority=Priority.LOW)
    critical = make_job(size_minutes=50, priority=Priority.CRITICAL)

    low_score = allocation_score(low, min_size_minutes=50, max_size_minutes=50)
    critical_score = allocation_score(critical, min_size_minutes=50, max_size_minutes=50)

    assert critical_score > low_score


def test_smaller_size_increases_score_at_equal_priority():
    small = make_job(size_minutes=10, priority=Priority.MEDIUM)
    large = make_job(size_minutes=100, priority=Priority.MEDIUM)

    small_score = allocation_score(small, min_size_minutes=10, max_size_minutes=100)
    large_score = allocation_score(large, min_size_minutes=10, max_size_minutes=100)

    assert small_score > large_score


def test_priority_can_outweigh_a_large_size_disadvantage():
    # This is the whole point of the 60/40 split: a high-enough-priority
    # large job can still beat a low-priority small job.
    critical_large = make_job(size_minutes=100, priority=Priority.CRITICAL)
    low_priority_small = make_job(size_minutes=10, priority=Priority.LOW)

    score_critical = allocation_score(critical_large, min_size_minutes=10, max_size_minutes=100)
    score_low = allocation_score(low_priority_small, min_size_minutes=10, max_size_minutes=100)

    assert score_critical > score_low


def test_not_pure_sjf_smallest_job_does_not_always_win():
    # A LOW-priority tiny job should not automatically beat a
    # CRITICAL-priority job of moderate size - priority still counts.
    tiny_low = make_job(size_minutes=1, priority=Priority.LOW)
    moderate_critical = make_job(size_minutes=50, priority=Priority.CRITICAL)

    score_tiny = allocation_score(tiny_low, min_size_minutes=1, max_size_minutes=50)
    score_moderate = allocation_score(moderate_critical, min_size_minutes=1, max_size_minutes=50)

    assert score_moderate > score_tiny
