import pytest

from engine.hardware.monitor import MonitorUnavailableError
from engine.hardware.nvml_monitor import NVMLMonitor


class _FakeUtilization:
    def __init__(self, gpu):
        self.gpu = gpu


class _FakeMemory:
    def __init__(self, used_bytes, total_bytes):
        self.used = used_bytes
        self.total = total_bytes


class FakePynvml:
    """Stands in for the real `pynvml` package - enough of its surface
    for NVMLMonitor to exercise its real parsing logic without needing
    the actual library or a GPU installed."""

    def __init__(self, readings):
        self._readings = readings  # list of (utilization_percent, used_mb, total_mb)
        self.shutdown_called = False

    def nvmlInit(self):
        pass

    def nvmlDeviceGetCount(self):
        return len(self._readings)

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUtilizationRates(self, handle):
        return _FakeUtilization(self._readings[handle][0])

    def nvmlDeviceGetMemoryInfo(self, handle):
        _, used_mb, total_mb = self._readings[handle]
        return _FakeMemory(used_mb * 1024 * 1024, total_mb * 1024 * 1024)

    def nvmlShutdown(self):
        self.shutdown_called = True


class FailingPynvml:
    def nvmlInit(self):
        raise RuntimeError("driver not loaded")


def test_reads_metrics_through_the_injected_pynvml_module():
    fake = FakePynvml([(92.0, 22528.0, 24576.0), (7.0, 2048.0, 24576.0)])
    monitor = NVMLMonitor(pynvml_module=fake)

    metrics = monitor.get_gpu_metrics()

    assert len(metrics) == 2
    assert metrics[0].gpu_id == "GPU-0"
    assert metrics[0].utilization_percent == 92.0
    assert metrics[0].memory_used_mb == pytest.approx(22528.0)
    assert metrics[0].memory_total_mb == pytest.approx(24576.0)
    assert metrics[1].gpu_id == "GPU-1"


def test_raises_monitor_unavailable_when_pynvml_is_not_installed():
    with pytest.raises(MonitorUnavailableError):
        NVMLMonitor()  # no injected module, and pynvml genuinely isn't installed here


def test_raises_monitor_unavailable_when_nvml_init_fails():
    with pytest.raises(MonitorUnavailableError, match="driver not loaded"):
        NVMLMonitor(pynvml_module=FailingPynvml())


def test_shutdown_calls_through_to_the_underlying_module():
    fake = FakePynvml([(1.0, 1.0, 1.0)])
    monitor = NVMLMonitor(pynvml_module=fake)
    monitor.shutdown()
    assert fake.shutdown_called is True


def test_shutdown_never_raises_even_if_the_module_misbehaves():
    class BrokenShutdown(FakePynvml):
        def nvmlShutdown(self):
            raise RuntimeError("already shut down")

    monitor = NVMLMonitor(pynvml_module=BrokenShutdown([(1.0, 1.0, 1.0)]))
    monitor.shutdown()  # must not raise
