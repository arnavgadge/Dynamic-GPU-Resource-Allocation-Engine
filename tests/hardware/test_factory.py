import pytest

from engine.hardware.factory import detect_gpu_monitor
from engine.hardware.metrics import GPUMetrics
from engine.hardware.monitor import GPUMonitor, MonitorUnavailableError


class AlwaysFails(GPUMonitor):
    def __init__(self):
        raise MonitorUnavailableError("first candidate never available")

    def get_gpu_metrics(self):
        return []


class FailsOnlyWhenProbed(GPUMonitor):
    """Constructs fine (like NvidiaSMIMonitor without the binary) but
    fails only once actually asked for metrics."""

    def get_gpu_metrics(self):
        raise MonitorUnavailableError("tool not installed")


class Works(GPUMonitor):
    def get_gpu_metrics(self):
        return [GPUMetrics(gpu_id="GPU-0", utilization_percent=1.0, memory_used_mb=None, memory_total_mb=None,
                             timestamp=None)]


def test_returns_the_first_candidate_that_both_constructs_and_probes_successfully():
    monitor = detect_gpu_monitor(candidates=[AlwaysFails, FailsOnlyWhenProbed, Works])
    assert isinstance(monitor, Works)


def test_construction_only_failure_is_not_enough_to_reject_a_candidate_falsely():
    # A monitor that constructs fine but fails when actually probed
    # must be skipped too, not returned just because __init__ worked.
    monitor = detect_gpu_monitor(candidates=[FailsOnlyWhenProbed, Works])
    assert isinstance(monitor, Works)


def test_raises_with_every_candidates_reason_when_none_are_available():
    with pytest.raises(MonitorUnavailableError) as exc_info:
        detect_gpu_monitor(candidates=[AlwaysFails, FailsOnlyWhenProbed])

    message = str(exc_info.value)
    assert "AlwaysFails" in message
    assert "FailsOnlyWhenProbed" in message
