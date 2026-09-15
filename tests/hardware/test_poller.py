from datetime import datetime, timedelta, timezone

from engine.hardware.metrics import GPUMetrics
from engine.hardware.monitor import GPUMonitor
from engine.hardware.poller import MonitorPoller, feed_metrics
from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.scheduler import Scheduler

START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


class ScriptedMonitor(GPUMonitor):
    """A GPUMonitor whose readings are pre-scripted per call - stands
    in for "whatever real hardware happens to report right now" in a
    fully deterministic test."""

    def __init__(self, readings_sequence):
        self._sequence = list(readings_sequence)
        self._index = 0

    def get_gpu_metrics(self):
        readings = self._sequence[min(self._index, len(self._sequence) - 1)]
        self._index += 1
        return readings


def build_scheduler_with_one_busy_gpu():
    scheduler = Scheduler()
    scheduler.add_user(User(user_id="u1", name="Alice", priority=Priority.MEDIUM))
    gpu = GPU(gpu_id="GPU-1", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
               assigned_user_id="u1", assigned_job_id="J1")
    job = Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
               estimated_size_minutes=600, status=JobStatus.RUNNING, started_at=START,
               submitted_at=START, assigned_gpu_ids=["GPU-1"])
    scheduler.state.add_job(job)
    scheduler.allocation_engine.add_gpu(gpu)
    scheduler.state.get_user("u1").assigned_gpu_ids.append("GPU-1")
    scheduler.state.get_user("u1").running_job_ids.append("J1")
    return scheduler, gpu, job


def test_feed_metrics_reaches_the_real_gpu_model():
    scheduler, gpu, _ = build_scheduler_with_one_busy_gpu()
    metrics = [GPUMetrics(gpu_id="GPU-1", utilization_percent=55.0, memory_used_mb=2000.0,
                            memory_total_mb=24_576, timestamp=START)]

    feed_metrics(scheduler, metrics)

    assert gpu.utilization_percent == 55.0
    assert gpu.memory_used_mb == 2000.0


def test_feed_metrics_ignores_a_gpu_the_scheduler_does_not_know_about():
    scheduler, gpu, _ = build_scheduler_with_one_busy_gpu()
    metrics = [GPUMetrics(gpu_id="GPU-UNKNOWN", utilization_percent=1.0, memory_used_mb=None,
                            memory_total_mb=None, timestamp=START)]

    feed_metrics(scheduler, metrics)  # must not raise

    assert gpu.utilization_percent == 0.0  # untouched


def test_poller_drives_a_full_sustained_reclaim_cycle_from_a_non_simulator_monitor():
    scheduler, gpu, job = build_scheduler_with_one_busy_gpu()

    # Six readings, five minutes apart, sustained below 2% - the exact
    # same Tier 1 shape Phase 4/6 already validated, now arriving
    # through a "hardware-shaped" monitor instead of a scenario action.
    readings = [
        [GPUMetrics(gpu_id="GPU-1", utilization_percent=1.0, memory_used_mb=None,
                     memory_total_mb=None, timestamp=START + timedelta(minutes=m))]
        for m in (0, 5, 10, 15, 20, 25)
    ]
    monitor = ScriptedMonitor(readings)
    poller = MonitorPoller(scheduler, monitor)

    for m in (0, 5, 10, 15, 20, 25):
        poller.poll_once(now=START + timedelta(minutes=m))

    assert gpu.status == GPUStatus.IDLE_WARNING
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1") is True
