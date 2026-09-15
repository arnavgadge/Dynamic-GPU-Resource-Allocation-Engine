"""SimulatorGPUMonitor: the simulation's own GPU state, exposed as a
`GPUMonitor` - the same interface a real deployment's monitor uses.

This does not generate any data. A `Simulator`'s scenario actions
already write utilization/memory directly onto the real `GPU` model
objects (through `Scheduler.record_utilization`, Phase 6); this class
only *reads* what is already there and reshapes it as `GPUMetrics`,
so any code written against `GPUMonitor` (a poller, a test, a future
CLI) works identically whether the source is this simulator or real
hardware.
"""

from typing import List

from engine.hardware.metrics import GPUMetrics
from engine.hardware.monitor import GPUMonitor
from engine.simulation.simulator import Simulator


class SimulatorGPUMonitor(GPUMonitor):
    """Reports a `Simulator`'s current GPU state as `GPUMetrics`."""

    def __init__(self, simulator: Simulator) -> None:
        self._simulator = simulator

    def get_gpu_metrics(self) -> List[GPUMetrics]:
        state = self._simulator.snapshot()
        now = self._simulator.clock.now()
        return [
            GPUMetrics(
                gpu_id=gpu.gpu_id,
                utilization_percent=gpu.utilization_percent,
                memory_used_mb=gpu.memory_used_mb,
                memory_total_mb=gpu.total_memory_mb,
                timestamp=now,
            )
            for gpu in state.gpus.values()
        ]
