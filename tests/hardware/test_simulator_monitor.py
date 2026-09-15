from datetime import datetime, timedelta, timezone

from engine.hardware.simulator_monitor import SimulatorGPUMonitor
from engine.models.enums import GPUStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.simulation.scenario import Scenario
from engine.simulation.simulator import Simulator

START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def build_simulator():
    scenario = Scenario(
        scenario_id="test", name="test", description="test", start_time=START,
        gpus=[GPU(gpu_id="GPU-1", total_memory_mb=24_576, utilization_percent=42.0,
                    memory_used_mb=1024.0, status=GPUStatus.IDLE)],
        users=[User(user_id="u1", name="U1", priority=Priority.MEDIUM)],
    )
    return Simulator(scenario)


def test_reports_exactly_the_gpu_state_already_in_the_simulator():
    simulator = build_simulator()
    monitor = SimulatorGPUMonitor(simulator)

    metrics = monitor.get_gpu_metrics()

    assert len(metrics) == 1
    assert metrics[0].gpu_id == "GPU-1"
    assert metrics[0].utilization_percent == 42.0
    assert metrics[0].memory_used_mb == 1024.0
    assert metrics[0].memory_total_mb == 24_576


def test_does_not_generate_data_it_only_reflects_current_state():
    simulator = build_simulator()
    monitor = SimulatorGPUMonitor(simulator)

    gpu = simulator.snapshot().get_gpu("GPU-1")
    gpu.utilization_percent = 7.5  # simulate an action having updated the real GPU object

    metrics = monitor.get_gpu_metrics()
    assert metrics[0].utilization_percent == 7.5


def test_timestamp_comes_from_the_simulated_clock_not_the_wall_clock():
    simulator = build_simulator()
    simulator.advance(timedelta(hours=2))
    monitor = SimulatorGPUMonitor(simulator)

    metrics = monitor.get_gpu_metrics()
    assert metrics[0].timestamp == START + timedelta(hours=2)


def test_base_class_lookups_work_through_the_simulator_monitor():
    simulator = build_simulator()
    monitor = SimulatorGPUMonitor(simulator)

    assert monitor.get_utilization("GPU-1") == 42.0
    assert monitor.get_memory("GPU-1") == (1024.0, 24_576)
