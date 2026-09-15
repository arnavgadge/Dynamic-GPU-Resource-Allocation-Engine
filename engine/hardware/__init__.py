"""Phase 8 Part A: the GPU monitoring abstraction.

    Scheduler
        v
    GPUMonitor (interface)
        v
    +--------------------+------------------+----------------+
    SimulatorGPUMonitor    NvidiaSMIMonitor    NVMLMonitor

`Scheduler` never calls `subprocess`, NVML, or a `Simulator` directly
for utilization data - every source implements `GPUMonitor` and is
fed into the scheduler through the same `feed_metrics`/`MonitorPoller`
bridge, so the scheduler cannot tell (and never needs to know) which
one produced a given reading.
"""

from engine.hardware.factory import DEFAULT_CANDIDATES, detect_gpu_monitor
from engine.hardware.metrics import GPUMetrics, GPUProcessInfo
from engine.hardware.monitor import GPUMonitor, MonitorUnavailableError
from engine.hardware.nvidia_smi_monitor import NvidiaSMIMonitor
from engine.hardware.nvml_monitor import NVMLMonitor
from engine.hardware.poller import MonitorPoller, feed_metrics
from engine.hardware.simulator_monitor import SimulatorGPUMonitor

__all__ = [
    "GPUMetrics",
    "GPUProcessInfo",
    "GPUMonitor",
    "MonitorUnavailableError",
    "SimulatorGPUMonitor",
    "NvidiaSMIMonitor",
    "NVMLMonitor",
    "detect_gpu_monitor",
    "DEFAULT_CANDIDATES",
    "feed_metrics",
    "MonitorPoller",
]
