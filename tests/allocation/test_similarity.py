import pytest

from engine.allocation.config import JOB_SIZE_SIMILARITY_THRESHOLD
from engine.allocation.similarity import relative_size_spread, sizes_are_similar


def test_identical_sizes_have_zero_spread():
    assert relative_size_spread([10, 10, 10]) == 0.0
    assert sizes_are_similar([10, 10, 10]) is True


def test_report_example_close_sizes_are_similar():
    # 10 / 12 / 11 minutes, from the project's own scenario description.
    assert sizes_are_similar([10, 12, 11]) is True


def test_report_example_wildly_different_sizes_are_not_similar():
    # 15 minutes / 1 hour / 3 hours.
    assert sizes_are_similar([15, 60, 180]) is False


def test_spread_is_measured_against_the_largest_value():
    # (12 - 10) / 12, not / 10.
    assert relative_size_spread([10, 12]) == pytest.approx((12 - 10) / 12)


def test_boundary_exactly_at_threshold_counts_as_similar():
    largest = 100.0
    smallest = largest * (1 - JOB_SIZE_SIMILARITY_THRESHOLD)
    assert relative_size_spread([smallest, largest]) == pytest.approx(JOB_SIZE_SIMILARITY_THRESHOLD)
    assert sizes_are_similar([smallest, largest]) is True


def test_boundary_just_past_threshold_is_not_similar():
    largest = 100.0
    smallest = largest * (1 - JOB_SIZE_SIMILARITY_THRESHOLD) - 1
    assert sizes_are_similar([smallest, largest]) is False


def test_single_size_has_zero_spread():
    assert relative_size_spread([42]) == 0.0
    assert sizes_are_similar([42]) is True


def test_non_positive_largest_size_is_treated_as_no_spread():
    assert relative_size_spread([0, -5, -1]) == 0.0


def test_empty_sizes_raises():
    with pytest.raises(ValueError):
        relative_size_spread([])
