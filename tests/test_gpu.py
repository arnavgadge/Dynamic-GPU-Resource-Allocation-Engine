from datetime import datetime, timezone

import pytest

from engine.models.enums import GPUStatus
from engine.models.gpu import GPU
from engine.models.utilization import UtilizationObservation


def test_gpu_can_be_created_with_defaults():
    gpu = GPU(gpu_id="GPU0", total_memory_mb=24_576)

    assert gpu.gpu_id == "GPU0"
    assert gpu.total_memory_mb == 24_576
    assert gpu.memory_used_mb == 0.0
    assert gpu.utilization_percent == 0.0
    assert gpu.status == GPUStatus.IDLE
    assert gpu.assigned_user_id is None
    assert gpu.assigned_job_id is None
    assert gpu.utilization_history == []
    assert gpu.is_assigned is False


def test_gpu_status_can_represent_every_defined_state():
    for status in GPUStatus:
        gpu = GPU(gpu_id="GPU0", total_memory_mb=24_576, status=status)
        assert gpu.status is status


def test_gpu_assignment_fields_reflect_current_holder():
    gpu = GPU(gpu_id="GPU0", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
              assigned_user_id="U1", assigned_job_id="J1")

    assert gpu.is_assigned is True
    assert gpu.assigned_user_id == "U1"
    assert gpu.assigned_job_id == "J1"


def test_recording_an_observation_updates_current_reading_and_history():
    gpu = GPU(gpu_id="GPU0", total_memory_mb=24_576)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    observation = UtilizationObservation(utilization_percent=42.0, memory_used_mb=1024.0, timestamp=ts)

    gpu.record_observation(observation)

    assert gpu.utilization_percent == 42.0
    assert gpu.memory_used_mb == 1024.0
    assert gpu.last_updated == ts
    assert gpu.utilization_history == [observation]


def test_memory_free_is_derived_from_total_and_used():
    gpu = GPU(gpu_id="GPU0", total_memory_mb=24_576, memory_used_mb=22_528)
    assert gpu.memory_free_mb == pytest.approx(2_048)
