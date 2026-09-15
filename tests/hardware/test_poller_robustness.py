"""Phase 16 of the 100-scenario fix set: the hardware poller must
isolate per-GPU failures and survive a whole-poll failure without
crashing the scheduler or the caller - verified by actually injecting
both kinds of failure, not by reading the try/except and assuming it
works.
"""

from datetime import datetime, timezone

from engine.hardware.metrics import GPUMetrics
from engine.hardware.monitor import GPUMonitor, MonitorUnavailableError
from engine.hardware.poller import MonitorPoller
from engine.models.enums import GPUStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.scheduler import Scheduler

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


class _PartlyBadMonitor(GPUMonitor):
    """One GPU reports a corrupt (out-of-range) reading; the other
    reports a perfectly normal one, in the same poll."""

    def get_gpu_metrics(self):
        return [
            GPUMetrics(gpu_id="GPU-1", utilization_percent=150.0, memory_used_mb=None, memory_total_mb=None, timestamp=NOW),
            GPUMetrics(gpu_id="GPU-2", utilization_percent=42.0, memory_used_mb=None, memory_total_mb=None, timestamp=NOW),
        ]


class _AlwaysFailingMonitor(GPUMonitor):
    def get_gpu_metrics(self):
        raise MonitorUnavailableError("nvidia-smi: device not found (simulated)")


def _scheduler_with_two_gpus():
    s = Scheduler()
    s.add_gpu(GPU(gpu_id="GPU-1", total_memory_mb=1000, status=GPUStatus.IDLE))
    s.add_gpu(GPU(gpu_id="GPU-2", total_memory_mb=1000, status=GPUStatus.IDLE))
    s.add_user(User(user_id="A", name="A", priority=Priority.MEDIUM))
    return s


def test_a_corrupt_reading_on_one_gpu_never_blocks_the_other():
    s = _scheduler_with_two_gpus()
    poller = MonitorPoller(s, _PartlyBadMonitor())

    metrics = poller.poll_once(NOW)

    assert len(metrics) == 2  # the poll itself succeeded
    assert "GPU-1" in poller.health.per_gpu_errors  # the bad reading was recorded...
    assert s.state.get_gpu("GPU-1").utilization_percent == 0.0  # ...and never applied
    assert s.state.get_gpu("GPU-2").utilization_percent == 42.0  # the good one still went through
    assert poller.health.consecutive_failures == 0  # a per-GPU error is not a whole-poll failure


def test_a_whole_poll_failure_never_raises_and_is_recorded_as_health():
    s = _scheduler_with_two_gpus()
    poller = MonitorPoller(s, _AlwaysFailingMonitor())

    metrics = poller.poll_once(NOW)  # must not raise

    assert metrics == []
    assert poller.health.consecutive_failures == 1
    assert poller.health.healthy is False
    assert "nvidia-smi" in poller.health.last_error


def test_the_scheduler_keeps_operating_across_repeated_poll_failures():
    """The actual guarantee Phase 6/H06 cares about: the scheduler
    itself (allocation, reclamation timeouts) keeps running normally
    even while every single poll is failing."""
    s = _scheduler_with_two_gpus()
    poller = MonitorPoller(s, _AlwaysFailingMonitor())

    job = Job(job_id="J1", user_id="A", name="j1", priority=Priority.MEDIUM, estimated_size_minutes=10, submitted_at=NOW)
    s.submit_job(job, now=NOW)

    for _ in range(10):
        poller.poll_once(NOW)  # repeatedly failing hardware - must never raise, never wedge the scheduler

    s.try_allocate_all(now=NOW)
    assert job.status.value == "RUNNING"  # the scheduler itself was never affected by the hardware outage
    assert poller.health.consecutive_failures == 10


def test_recovery_after_failures_resets_the_health_counter():
    calls = {"n": 0}

    class _RecoveringMonitor(GPUMonitor):
        def get_gpu_metrics(self):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise MonitorUnavailableError("still down")
            return [GPUMetrics(gpu_id="GPU-1", utilization_percent=10.0, memory_used_mb=None, memory_total_mb=None, timestamp=NOW)]

    s = _scheduler_with_two_gpus()
    poller = MonitorPoller(s, _RecoveringMonitor())
    poller.poll_once(NOW)
    poller.poll_once(NOW)
    assert poller.health.consecutive_failures == 2

    poller.poll_once(NOW)
    assert poller.health.consecutive_failures == 0
    assert poller.health.last_success_at == NOW
