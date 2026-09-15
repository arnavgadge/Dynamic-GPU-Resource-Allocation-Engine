import pytest

from engine.simulation import load_default_registry
from engine.simulation.simulator import Simulator

ALL_SCENARIO_IDS = ["gta5_excel", "ml_video", "multiple_ml", "idle_user", "imbalance", "full_lifecycle",
                     "interactive_demo"]


def _run(scenario_id: str):
    registry = load_default_registry()
    scenario = registry.load(scenario_id)
    sim = Simulator(scenario)
    sim.run_to_completion()
    return sim.snapshot()


def _job_fingerprint(state):
    return {job.job_id: (job.status.value, job.assigned_gpu_id) for job in state.jobs.values()}


def _gpu_fingerprint(state):
    return {
        gpu.gpu_id: (gpu.status.value, gpu.assigned_user_id, gpu.assigned_job_id, gpu.utilization_percent)
        for gpu in state.gpus.values()
    }


def _event_fingerprint(state):
    # Timestamps are included deliberately - they come from the
    # simulated clock, not the wall clock, so they must also be
    # identical across two independent runs.
    return [(e.timestamp, e.event_type.value, e.gpu_id, e.user_id, e.job_id, e.message, e.reason) for e in state.events]


# ------------------------------------------------------------------
# Test 18 - running a scenario twice produces equivalent results
# ------------------------------------------------------------------

@pytest.mark.parametrize("scenario_id", ALL_SCENARIO_IDS)
def test_running_a_scenario_twice_is_deterministic(scenario_id):
    first = _run(scenario_id)
    second = _run(scenario_id)

    assert _job_fingerprint(first) == _job_fingerprint(second)
    assert _gpu_fingerprint(first) == _gpu_fingerprint(second)
    assert _event_fingerprint(first) == _event_fingerprint(second)


@pytest.mark.parametrize("scenario_id", ALL_SCENARIO_IDS)
def test_resetting_and_rerunning_the_same_simulator_is_deterministic(scenario_id):
    registry = load_default_registry()
    scenario = registry.load(scenario_id)
    sim = Simulator(scenario)
    sim.run_to_completion()
    first_jobs = _job_fingerprint(sim.snapshot())
    first_events = _event_fingerprint(sim.snapshot())

    sim.reset()
    sim.run_to_completion()
    second_jobs = _job_fingerprint(sim.snapshot())
    second_events = _event_fingerprint(sim.snapshot())

    assert first_jobs == second_jobs
    assert first_events == second_events
