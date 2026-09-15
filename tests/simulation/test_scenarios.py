"""One behavioral assertion per built-in scenario, beyond the generic
simulator-mechanics tests - confirming each scenario actually
demonstrates what its docstring/description claims, purely by reading
the real `Scheduler`'s resulting state (never by asserting on the
scenario definition itself).
"""

from engine.models.enums import JobStatus
from engine.simulation import load_default_registry
from engine.simulation.simulator import Simulator


def run(scenario_id: str):
    registry = load_default_registry()
    sim = Simulator(registry.load(scenario_id))
    sim.run_to_completion()
    return sim.snapshot()


def test_gta5_excel_both_users_get_a_gpu_each():
    state = run("gta5_excel")
    assert state.get_job("J-GTA5").status == JobStatus.RUNNING
    assert state.get_job("J-EXCEL").status == JobStatus.RUNNING
    assert state.get_job("J-GTA5").assigned_gpu_id != state.get_job("J-EXCEL").assigned_gpu_id


def test_ml_video_smaller_job_wins_the_score_based_competition():
    state = run("ml_video")
    # The weighted formula favors the much smaller Video job over the
    # larger, higher-priority ML job on the pool's only GPU.
    assert state.get_job("J-VIDEO").status == JobStatus.RUNNING
    assert state.get_job("J-ML").status == JobStatus.WAITING


def test_multiple_ml_exactly_one_job_is_left_waiting_with_four_gpus():
    state = run("multiple_ml")
    running = [j for j in state.jobs.values() if j.status == JobStatus.RUNNING]
    waiting = [j for j in state.jobs.values() if j.status == JobStatus.WAITING]
    assert len(running) == 4
    assert len(waiting) == 1
    # All four GPUs are actually in use, none doubly-assigned.
    assigned_gpus = {j.assigned_gpu_id for j in running}
    assert len(assigned_gpus) == 4


def test_idle_user_reclaims_and_reallocates():
    state = run("idle_user")
    assert state.get_job("J-LONGTAIL").status == JobStatus.RECLAIMED
    assert state.get_job("J-NEWWORK").status == JobStatus.RUNNING
    assert state.get_job("J-NEWWORK").assigned_gpu_id == "GPU-1"


def test_imbalance_never_selects_the_underutilized_assigned_gpu():
    state = run("imbalance")
    # GPU-3 (4%, assigned to User C) must remain exactly as it started.
    gpu3 = state.get_gpu("GPU-3")
    assert gpu3.assigned_user_id == "user-c"
    assert gpu3.utilization_percent == 4.0
    # The two new jobs went to the genuinely available GPUs instead.
    assert state.get_job("J-D").assigned_gpu_id == "GPU-4"
    assert state.get_job("J-E").assigned_gpu_id == "GPU-2"
