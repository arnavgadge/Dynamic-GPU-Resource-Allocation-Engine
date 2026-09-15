import math

import pytest

from engine.balancing.availability import has_valid_utilization, is_gpu_available
from engine.models.enums import GPUStatus
from engine.models.gpu import GPU


def make_gpu(status: GPUStatus, assigned: bool, utilization: float = 10.0) -> GPU:
    return GPU(
        gpu_id="GPU0", total_memory_mb=24_576, utilization_percent=utilization, status=status,
        assigned_user_id="U1" if assigned else None,
        assigned_job_id="J1" if assigned else None,
    )


@pytest.mark.parametrize(
    "status,assigned,expected",
    [
        (GPUStatus.IDLE, False, True),          # genuinely free
        (GPUStatus.ACTIVE, True, False),         # in use
        (GPUStatus.IDLE_WARNING, True, False),   # still assigned, just flagged
        (GPUStatus.RECLAIMING, False, False),    # mid reclaim transition
        (GPUStatus.REALLOCATING, False, False),  # mid reallocation transition
        (GPUStatus.ACTIVE, False, False),        # inconsistent data - status says busy
        (GPUStatus.IDLE, True, False),           # inconsistent data - still assigned
    ],
)
def test_is_gpu_available_matrix(status, assigned, expected):
    gpu = make_gpu(status, assigned)
    assert is_gpu_available(gpu) is expected


def test_low_utilization_alone_does_not_make_a_gpu_available():
    gpu = make_gpu(GPUStatus.ACTIVE, assigned=True, utilization=4.0)
    assert is_gpu_available(gpu) is False


@pytest.mark.parametrize("value", [0.0, 50.0, 100.0])
def test_valid_utilization_values(value):
    gpu = make_gpu(GPUStatus.IDLE, assigned=False, utilization=value)
    assert has_valid_utilization(gpu) is True


@pytest.mark.parametrize("value", [-1.0, 100.1, float("nan"), float("inf"), float("-inf")])
def test_invalid_utilization_values(value):
    gpu = make_gpu(GPUStatus.IDLE, assigned=False, utilization=value)
    assert has_valid_utilization(gpu) is False
