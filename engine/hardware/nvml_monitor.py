"""NVMLMonitor: reads real GPU metrics via NVML - the direct C API
into the NVIDIA kernel driver - through the `pynvml` bindings.

This is the "actual kernel level" layer the project report describes.
It sits behind the exact same `GPUMonitor` interface as the simulator
and `nvidia-smi` adapters; nothing about how a poller or the bridge
into `Scheduler` uses this class differs from the others.
"""

from datetime import datetime, timezone
from typing import Any, List, Optional

from engine.hardware.metrics import GPUMetrics
from engine.hardware.monitor import GPUMonitor, MonitorUnavailableError


class NVMLMonitor(GPUMonitor):
    """Reads GPU metrics through NVML.

    ``pynvml_module`` is injectable for testing - a fake object
    exposing the handful of NVML functions used below stands in for
    the real `pynvml` package, so this class is fully testable on a
    machine with no GPU and no `pynvml` installed.
    """

    def __init__(self, pynvml_module: Optional[Any] = None, gpu_id_prefix: str = "GPU-") -> None:
        if pynvml_module is None:
            try:
                import pynvml as pynvml_module  # type: ignore
            except ImportError as exc:
                raise MonitorUnavailableError(
                    "pynvml is not installed - pip install nvidia-ml-py, or use a different GPUMonitor"
                ) from exc

        self._pynvml = pynvml_module
        self._gpu_id_prefix = gpu_id_prefix
        try:
            self._pynvml.nvmlInit()
        except Exception as exc:
            raise MonitorUnavailableError(f"NVML failed to initialize: {exc}") from exc

    def get_gpu_metrics(self) -> List[GPUMetrics]:
        pynvml = self._pynvml
        now = datetime.now(timezone.utc)
        metrics: List[GPUMetrics] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics.append(GPUMetrics(
                gpu_id=f"{self._gpu_id_prefix}{index}",
                utilization_percent=float(utilization),
                memory_used_mb=memory.used / (1024 * 1024),
                memory_total_mb=memory.total / (1024 * 1024),
                timestamp=now,
            ))
        return metrics

    def shutdown(self) -> None:
        """Release NVML. Best-effort - a failure here shouldn't mask
        whatever the caller was already doing."""
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass
