"""detect_gpu_monitor: pick the best available *real* GPU monitor.

Tries each candidate in preference order - NVML first (the most
direct route into the kernel driver), then `nvidia-smi` - and returns
the first one that actually initializes. Deliberately does not fall
back to `SimulatorGPUMonitor` itself: whether "no real GPU available"
should mean "run the simulator instead" is a decision for whatever
starts the process (a script, a test, a future CLI flag), not for
this factory to make silently.
"""

from typing import List, Optional, Type

from engine.hardware.monitor import GPUMonitor, MonitorUnavailableError
from engine.hardware.nvidia_smi_monitor import NvidiaSMIMonitor
from engine.hardware.nvml_monitor import NVMLMonitor

DEFAULT_CANDIDATES: List[Type[GPUMonitor]] = [NVMLMonitor, NvidiaSMIMonitor]


def detect_gpu_monitor(candidates: Optional[List[Type[GPUMonitor]]] = None) -> GPUMonitor:
    """Return the first real monitor that successfully initializes
    *and* can actually produce a reading.

    Construction alone isn't a reliable availability check: some
    monitors (`NVMLMonitor`) fail immediately if their backing
    library is missing, but others (`NvidiaSMIMonitor`) only discover
    the tool is missing when they actually try to run it. So every
    candidate is both constructed *and* probed with one
    `get_gpu_metrics()` call before being accepted.

    Raises `MonitorUnavailableError` (with every candidate's own
    failure reason) if none of them can talk to real hardware.
    """
    candidates = candidates if candidates is not None else DEFAULT_CANDIDATES
    failures = []
    for candidate in candidates:
        try:
            monitor = candidate()
            monitor.get_gpu_metrics()
        except MonitorUnavailableError as exc:
            failures.append(f"{candidate.__name__}: {exc}")
        else:
            return monitor

    joined = "; ".join(failures) if failures else "no candidates were given"
    raise MonitorUnavailableError(f"no real GPU monitor is available - {joined}")
