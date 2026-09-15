"""GPUMonitor: the interface the scheduler-facing code depends on,
instead of calling `subprocess`/NVML/the simulator directly.

    Scheduler
        v
    GPU Monitor Interface (this module)
        v
    +-----------------+-------------------+-----------------+
    SimulatorGPUMonitor  NvidiaSMIMonitor    NVMLMonitor

Every implementation returns the exact same `GPUMetrics` shape - the
whole point of this abstraction is that nothing above it (the bridge
into `Scheduler.record_utilization`, a poller, the scheduler itself)
needs a single line of source-specific logic to work with any of them.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from engine.hardware.metrics import GPUMetrics, GPUProcessInfo


class MonitorUnavailableError(RuntimeError):
    """Raised when a monitor's underlying tool/library/hardware isn't
    available - e.g. `nvidia-smi` isn't installed, `pynvml` isn't
    installed, or NVML failed to initialize because there is no GPU.
    Never raised for "the reading was momentarily off"; that's just a
    normal `GPUMetrics` value.
    """


class GPUMonitor(ABC):
    """Read-only source of current GPU metrics.

    Subclasses only need to implement `get_gpu_metrics()` - the three
    per-GPU convenience lookups below are all built on it, so every
    implementation (including a future one) gets them for free and
    can't accidentally disagree with `get_gpu_metrics()` about a given
    GPU's numbers.
    """

    @abstractmethod
    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """Every GPU this monitor currently knows about."""

    def get_utilization(self, gpu_id: str) -> float:
        return self._find(gpu_id).utilization_percent

    def get_memory(self, gpu_id: str) -> Tuple[Optional[float], Optional[float]]:
        metric = self._find(gpu_id)
        return metric.memory_used_mb, metric.memory_total_mb

    def get_processes(self, gpu_id: str) -> List[GPUProcessInfo]:
        return list(self._find(gpu_id).processes)

    def _find(self, gpu_id: str) -> GPUMetrics:
        for metric in self.get_gpu_metrics():
            if metric.gpu_id == gpu_id:
                return metric
        raise KeyError(f"unknown GPU {gpu_id!r}")
