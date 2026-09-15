"""NvidiaSMIMonitor: reads real GPU metrics by shelling out to
`nvidia-smi` - the "kernel-aware" layer the project report describes,
one step removed from the raw driver.

Nothing else in this project calls `subprocess` to talk to a GPU;
this is the one place that happens, behind the same `GPUMonitor`
interface `SimulatorGPUMonitor` and `NVMLMonitor` implement.
"""

import subprocess
from datetime import datetime, timezone
from typing import Callable, List, Optional

from engine.hardware.metrics import GPUMetrics
from engine.hardware.monitor import GPUMonitor, MonitorUnavailableError

_QUERY_FIELDS = "index,utilization.gpu,memory.used,memory.total"
_COMMAND = ["nvidia-smi", f"--query-gpu={_QUERY_FIELDS}", "--format=csv,noheader,nounits"]


class NvidiaSMIMonitor(GPUMonitor):
    """Parses ``nvidia-smi --query-gpu=... --format=csv`` output.

    ``runner`` defaults to `subprocess.run` and is injectable purely
    for testing without a real GPU or the `nvidia-smi` binary present
    - the CSV-parsing logic below is exactly what would run against
    real hardware either way.
    """

    def __init__(self, runner: Optional[Callable[..., "subprocess.CompletedProcess"]] = None,
                 gpu_id_prefix: str = "GPU-") -> None:
        self._runner = runner or subprocess.run
        self._gpu_id_prefix = gpu_id_prefix

    def get_gpu_metrics(self) -> List[GPUMetrics]:
        try:
            result = self._runner(_COMMAND, capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, OSError) as exc:
            raise MonitorUnavailableError("nvidia-smi is not available on this system") from exc

        if result.returncode != 0:
            raise MonitorUnavailableError(
                f"nvidia-smi exited with status {result.returncode}: {result.stderr.strip()}"
            )

        now = datetime.now(timezone.utc)
        metrics: List[GPUMetrics] = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                index, utilization, memory_used, memory_total = (part.strip() for part in line.split(","))
            except ValueError as exc:
                raise MonitorUnavailableError(f"unexpected nvidia-smi output line: {line!r}") from exc

            metrics.append(GPUMetrics(
                gpu_id=f"{self._gpu_id_prefix}{index}",
                utilization_percent=float(utilization),
                memory_used_mb=float(memory_used),
                memory_total_mb=float(memory_total),
                timestamp=now,
            ))
        return metrics
