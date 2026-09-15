from datetime import datetime, timezone

import pytest

from engine.hardware.metrics import GPUMetrics, GPUProcessInfo
from engine.hardware.monitor import GPUMonitor

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeMonitor(GPUMonitor):
    """A minimal concrete GPUMonitor - exercises the base class's
    get_utilization/get_memory/get_processes, which are all built on
    top of a single get_gpu_metrics() implementation."""

    def __init__(self, metrics):
        self._metrics = metrics

    def get_gpu_metrics(self):
        return self._metrics


def test_gpu_metrics_holds_every_field():
    metric = GPUMetrics(
        gpu_id="GPU-0", utilization_percent=42.0, memory_used_mb=1024.0,
        memory_total_mb=8192.0, timestamp=NOW, processes=[GPUProcessInfo(pid=123, memory_used_mb=512.0)],
    )
    assert metric.gpu_id == "GPU-0"
    assert metric.utilization_percent == 42.0
    assert metric.processes[0].pid == 123


def test_base_class_lookups_are_derived_from_get_gpu_metrics():
    metrics = [
        GPUMetrics(gpu_id="GPU-0", utilization_percent=10.0, memory_used_mb=100.0, memory_total_mb=1000.0,
                    timestamp=NOW, processes=[GPUProcessInfo(pid=1)]),
        GPUMetrics(gpu_id="GPU-1", utilization_percent=90.0, memory_used_mb=900.0, memory_total_mb=1000.0,
                    timestamp=NOW),
    ]
    monitor = _FakeMonitor(metrics)

    assert monitor.get_utilization("GPU-1") == 90.0
    assert monitor.get_memory("GPU-0") == (100.0, 1000.0)
    assert [p.pid for p in monitor.get_processes("GPU-0")] == [1]


def test_unknown_gpu_id_raises_key_error():
    monitor = _FakeMonitor([GPUMetrics(gpu_id="GPU-0", utilization_percent=1.0,
                                          memory_used_mb=None, memory_total_mb=None, timestamp=NOW)])
    with pytest.raises(KeyError):
        monitor.get_utilization("GPU-99")
