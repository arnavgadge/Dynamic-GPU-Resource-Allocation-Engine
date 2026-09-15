"""The central guarantee of Phase 8 Part A: the scheduler produces
identical decisions regardless of which GPUMonitor supplied the
readings. Two independent Schedulers, seeded identically, are driven
by (a) a SimulatorGPUMonitor wrapping a real Simulator and (b) a
"hardware-shaped" monitor scripted with the exact same numbers - and
must end up in exactly the same state.
"""

from datetime import datetime, timedelta, timezone

from engine.hardware.metrics import GPUMetrics
from engine.hardware.monitor import GPUMonitor
from engine.hardware.poller import MonitorPoller
from engine.hardware.simulator_monitor import SimulatorGPUMonitor
from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.policy import ConfirmationResponse
from engine.scheduler import Scheduler
from engine.simulation.scenario import Scenario
from engine.simulation.simulator import Simulator

START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
READINGS_MINUTES = (0, 5, 10, 15, 20, 25)


class ScriptedHardwareMonitor(GPUMonitor):
    """A GPUMonitor shaped like a real one (`nvidia-smi`/NVML) would
    be - readings supplied per poll - carrying the exact same numbers
    a scenario's UtilizationAction would report."""

    def __init__(self, readings_by_call):
        self._readings = list(readings_by_call)
        self._index = 0

    def get_gpu_metrics(self):
        readings = self._readings[min(self._index, len(self._readings) - 1)]
        self._index += 1
        return readings


def build_plain_scheduler():
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
    return scheduler, gpu


def run_via_simulator():
    """Drives the same scenario purely through `SimulatorGPUMonitor` +
    `MonitorPoller` - no scenario `UtilizationAction`s at all. The raw
    utilization reading is poked directly onto the `GPU` object before
    each poll, standing in for "whatever a real sensor would read right
    now" exactly the way `ScriptedHardwareMonitor` stands in for a real
    monitor below - so both paths are driven the same way: an external
    reading appears, a `MonitorPoller` pulls it, and the same
    `Scheduler.record_utilization` call is what actually happens next.
    """
    scenario = Scenario(
        scenario_id="agnostic-test", name="test", description="test", start_time=START,
        gpus=[GPU(gpu_id="GPU-1", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
                    assigned_user_id="u1", assigned_job_id="J1")],
        users=[User(user_id="u1", name="Alice", priority=Priority.MEDIUM)],
        initial_jobs=[Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
                            estimated_size_minutes=600, status=JobStatus.RUNNING, started_at=START,
                            submitted_at=START, assigned_gpu_ids=["GPU-1"])],
    )
    simulator = Simulator(scenario)
    monitor = SimulatorGPUMonitor(simulator)
    poller = MonitorPoller(simulator.scheduler, monitor)

    for minute in READINGS_MINUTES:
        simulator.clock.set(START + timedelta(minutes=minute))
        simulator.scheduler.state.get_gpu("GPU-1").utilization_percent = 1.0
        poller.poll_once(now=simulator.clock.now())

    return simulator.scheduler


def run_via_scripted_hardware_monitor():
    scheduler, _ = build_plain_scheduler()
    readings = [
        [GPUMetrics(gpu_id="GPU-1", utilization_percent=1.0, memory_used_mb=None,
                     memory_total_mb=None, timestamp=START + timedelta(minutes=m))]
        for m in READINGS_MINUTES
    ]
    monitor = ScriptedHardwareMonitor(readings)
    poller = MonitorPoller(scheduler, monitor)

    for minute in READINGS_MINUTES:
        poller.poll_once(now=START + timedelta(minutes=minute))

    return scheduler


def test_sustained_breach_reaches_the_same_outcome_regardless_of_monitor_source():
    scheduler_via_simulator = run_via_simulator()
    scheduler_via_hardware = run_via_scripted_hardware_monitor()

    gpu_a = scheduler_via_simulator.state.get_gpu("GPU-1")
    gpu_b = scheduler_via_hardware.state.get_gpu("GPU-1")

    assert gpu_a.status == gpu_b.status == GPUStatus.IDLE_WARNING
    assert scheduler_via_simulator.reclamation_engine.has_pending_prompt("GPU-1") is True
    assert scheduler_via_hardware.reclamation_engine.has_pending_prompt("GPU-1") is True


def test_the_no_response_and_reclaim_outcome_is_also_identical_regardless_of_source():
    scheduler_via_simulator = run_via_simulator()
    scheduler_via_hardware = run_via_scripted_hardware_monitor()

    now = START + timedelta(minutes=30)
    scheduler_via_simulator.reclamation_engine.respond("GPU-1", ConfirmationResponse.NO, now=now)
    scheduler_via_hardware.reclamation_engine.respond("GPU-1", ConfirmationResponse.NO, now=now)

    gpu_a = scheduler_via_simulator.state.get_gpu("GPU-1")
    gpu_b = scheduler_via_hardware.state.get_gpu("GPU-1")

    assert gpu_a.status == gpu_b.status == GPUStatus.IDLE
    assert gpu_a.assigned_user_id is None
    assert gpu_b.assigned_user_id is None
    assert scheduler_via_simulator.state.get_job("J1").status == JobStatus.RECLAIMED
    assert scheduler_via_hardware.state.get_job("J1").status == JobStatus.RECLAIMED
