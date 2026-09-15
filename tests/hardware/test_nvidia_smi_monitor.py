import subprocess

import pytest

from engine.hardware.monitor import MonitorUnavailableError
from engine.hardware.nvidia_smi_monitor import NvidiaSMIMonitor


def make_result(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_parses_real_shaped_nvidia_smi_csv_output():
    csv_output = "0, 92, 22528, 24576\n1, 7, 2048, 24576\n"
    monitor = NvidiaSMIMonitor(runner=lambda *a, **k: make_result(stdout=csv_output))

    metrics = monitor.get_gpu_metrics()

    assert len(metrics) == 2
    assert metrics[0].gpu_id == "GPU-0"
    assert metrics[0].utilization_percent == 92.0
    assert metrics[0].memory_used_mb == 22528.0
    assert metrics[0].memory_total_mb == 24576.0
    assert metrics[1].gpu_id == "GPU-1"
    assert metrics[1].utilization_percent == 7.0


def test_raises_monitor_unavailable_when_the_binary_is_missing():
    def missing_binary(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monitor = NvidiaSMIMonitor(runner=missing_binary)
    with pytest.raises(MonitorUnavailableError):
        monitor.get_gpu_metrics()


def test_raises_monitor_unavailable_on_nonzero_exit_code():
    monitor = NvidiaSMIMonitor(runner=lambda *a, **k: make_result(returncode=1, stderr="no devices found"))
    with pytest.raises(MonitorUnavailableError, match="no devices found"):
        monitor.get_gpu_metrics()


def test_raises_monitor_unavailable_on_malformed_output():
    monitor = NvidiaSMIMonitor(runner=lambda *a, **k: make_result(stdout="not,valid\n"))
    with pytest.raises(MonitorUnavailableError):
        monitor.get_gpu_metrics()


def test_blank_lines_in_output_are_ignored():
    monitor = NvidiaSMIMonitor(runner=lambda *a, **k: make_result(stdout="0, 50, 1000, 2000\n\n"))
    metrics = monitor.get_gpu_metrics()
    assert len(metrics) == 1


def test_gpu_id_prefix_is_configurable():
    monitor = NvidiaSMIMonitor(runner=lambda *a, **k: make_result(stdout="3, 10, 100, 200\n"), gpu_id_prefix="CARD-")
    metrics = monitor.get_gpu_metrics()
    assert metrics[0].gpu_id == "CARD-3"
