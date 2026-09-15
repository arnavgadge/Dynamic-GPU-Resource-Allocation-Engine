"""The bridge from any `GPUMonitor` into the real `Scheduler`.

This is the one place a `GPUMetrics` reading turns into a call to
`Scheduler.record_utilization` - the exact same entry point a
scenario's `UtilizationAction` already uses (Phase 6). Whether the
metrics came from `SimulatorGPUMonitor`, `NvidiaSMIMonitor`, or
`NVMLMonitor` is invisible past this point: the scheduler receives
the same shape of input either way and reacts with the same
Allocation/Reclamation/Balancing logic regardless of source - the
guarantee Phase 8 Part A exists to make true.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from engine.hardware.metrics import GPUMetrics
from engine.hardware.monitor import GPUMonitor, MonitorUnavailableError
from engine.scheduler import Scheduler


def feed_metrics(scheduler: Scheduler, metrics: Iterable[GPUMetrics]) -> Dict[str, str]:
    """Push a batch of `GPUMetrics` into ``scheduler`` through
    `Scheduler.record_utilization` - one call per GPU the scheduler
    actually knows about. A metric for a GPU outside the scheduler's
    pool is skipped rather than raising: a monitor may legitimately
    see devices this scheduler's pool doesn't (yet) include.

    100-scenario validation, Phase 16: each GPU's own reading is
    isolated in its own try/except - a single corrupt/out-of-range
    value (`UtilizationObservation`'s own validation rejecting it,
    the exact case Test H12 exercises) or any other per-GPU error no
    longer aborts the rest of the batch; every *other* GPU's reading
    from the same poll still gets applied. Returns ``{gpu_id: error}``
    for whichever GPUs failed, for the caller's own health reporting -
    never raised.
    """
    errors: Dict[str, str] = {}
    for metric in metrics:
        if scheduler.state.get_gpu(metric.gpu_id) is None:
            continue
        try:
            scheduler.record_utilization(
                metric.gpu_id, metric.utilization_percent, metric.timestamp, metric.memory_used_mb
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: one bad GPU must never stop the batch
            errors[metric.gpu_id] = str(exc)
    return errors


@dataclass
class PollerHealth:
    """The poller's own observable state (Phase 16) - never raised,
    always read. A dashboard/admin view can show this directly."""

    consecutive_failures: int = 0
    last_error: Optional[str] = None
    last_success_at: Optional[datetime] = None
    last_attempt_at: Optional[datetime] = None
    per_gpu_errors: Dict[str, str] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.consecutive_failures == 0


class MonitorPoller:
    """Pulls current metrics from one `GPUMonitor` and drives one
    `Scheduler` with them: feed the readings in, then let the
    scheduler react (a freshly-freed GPU gets allocated, a
    now-overdue prompt auto-reclaims) - the same
    observe -> decide -> update sequence every phase of this project
    has used, now running on whichever `GPUMonitor` it was built with.

    100-scenario validation, Phase 16/H06: a *whole-poll* failure
    (`monitor.get_gpu_metrics()` itself raising - e.g. `nvidia-smi`
    genuinely not responding) is caught here, never left to propagate
    into whatever's driving this poller (a background loop that has
    no idea how to recover from an arbitrary exception). The scheduler
    keeps its last-known state and keeps running; `health` records the
    failure for observability instead.
    """

    def __init__(self, scheduler: Scheduler, monitor: GPUMonitor) -> None:
        self.scheduler = scheduler
        self.monitor = monitor
        self.health = PollerHealth()

    def poll_once(self, now: Optional[datetime] = None) -> List[GPUMetrics]:
        """One full cycle: read metrics, feed them in, let the
        scheduler react. Returns the metrics that were read (empty on
        a whole-poll failure - never raised); `self.health` always
        reflects the outcome, and the scheduler always keeps
        operating either way, per-GPU errors included.
        """
        now = now or datetime.now(timezone.utc)
        self.health.last_attempt_at = now

        try:
            metrics = self.monitor.get_gpu_metrics()
        except MonitorUnavailableError as exc:
            self.health.consecutive_failures += 1
            self.health.last_error = str(exc)
            # Preserve last known state: no metrics fed in, but the
            # scheduler still gets to react to time passing (a
            # confirmation prompt's grace period can still expire, for
            # instance) - a hardware hiccup never freezes the rest of
            # the engine.
            self.scheduler.check_reclamation_timeouts(now)
            self.scheduler.try_allocate_all(now=now)
            return []

        errors = feed_metrics(self.scheduler, metrics)
        self.health.per_gpu_errors = errors
        self.health.consecutive_failures = 0
        self.health.last_success_at = now
        if errors:
            self.health.last_error = f"{len(errors)} GPU(s) reported bad readings this poll: {errors}"

        self.scheduler.check_reclamation_timeouts(now)
        self.scheduler.try_allocate_all(now=now)
        return metrics
