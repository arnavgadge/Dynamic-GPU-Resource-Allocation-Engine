"""Direct unit tests for `api.session.SimulationSession`, focused on
the Phase 9 additions (interactive GPU requests, estimated-completion
prompting, GPU-ownership lookup) that were not covered by the
existing FastAPI route tests.
"""

from datetime import timedelta

import pytest

from engine.models.enums import GPUStatus, JobStatus
from engine.simulation import load_default_registry
from api.session import SimulationSession


@pytest.fixture
def session():
    registry = load_default_registry()
    return SimulationSession(registry, "interactive_demo")


def test_submit_user_request_reaches_the_real_scheduler_and_allocates():
    registry = load_default_registry()
    session = SimulationSession(registry, "interactive_demo")

    job = session.submit_user_request("user_a", "User A", "ML Training", 1, 20, "HIGH")

    assert job.status == JobStatus.RUNNING
    assert job.assigned_gpu_id is not None
    state = session.simulator.scheduler.state
    assert state.get_user("user_a").name == "User A"


def test_submit_user_request_auto_registers_a_user_not_already_in_the_scenario():
    registry = load_default_registry()
    session = SimulationSession(registry, "gta5_excel")  # doesn't define "user_a"

    assert session.simulator.scheduler.state.get_user("user_a") is None
    job = session.submit_user_request("user_a", "User A", "job", 1, 10, "MEDIUM")
    assert session.simulator.scheduler.state.get_user("user_a") is not None
    assert job.user_id == "user_a"


def test_submit_user_request_accepts_a_multi_gpu_request(session):
    # interactive_demo has 10 free GPUs - a 2-GPU request is fully
    # satisfiable and should come back RUNNING with both GPUs held.
    job = session.submit_user_request("user_a", "User A", "job", 2, 10, "HIGH")
    assert job.status == JobStatus.RUNNING
    assert len(job.assigned_gpu_ids) == 2


def test_submit_user_request_rejects_the_full_pool_and_zero(session):
    with pytest.raises(ValueError, match="gpu_count"):
        session.submit_user_request("user_a", "User A", "job", 10, 10, "HIGH")
    with pytest.raises(ValueError, match="gpu_count"):
        session.submit_user_request("user_a", "User A", "job", 0, 10, "HIGH")


def test_submit_user_request_rejects_an_unknown_priority(session):
    with pytest.raises(ValueError, match="priority"):
        session.submit_user_request("user_a", "User A", "job", 1, 10, "URGENT")


def test_request_waits_when_pool_is_full(session):
    # interactive_demo has 10 free GPUs.
    jobs = [
        session.submit_user_request(f"user-{i}", f"User {i}", "job", 1, 30, "MEDIUM")
        for i in range(10)
    ]
    assert all(j.status == JobStatus.RUNNING for j in jobs)

    eleventh = session.submit_user_request("user_a", "User A", "second job", 1, 10, "MEDIUM")
    assert eleventh.status == JobStatus.WAITING


def test_gpu_owner_reflects_the_real_assignment(session):
    job = session.submit_user_request("user_a", "User A", "job", 1, 10, "HIGH")
    assert session.gpu_owner(job.assigned_gpu_id) == "user_a"
    assert session.gpu_owner("GPU-999") is None


def test_estimated_completion_prompts_without_deleting_the_job(session):
    job = session.submit_user_request("user_a", "User A", "job", 1, 20, "HIGH")
    gpu_id = job.assigned_gpu_id

    session.step(minutes=20)

    gpu = session.simulator.scheduler.state.get_gpu(gpu_id)
    assert gpu.status == GPUStatus.IDLE_WARNING
    assert session.simulator.scheduler.reclamation_engine.has_pending_prompt(gpu_id) is True
    assert job.status == JobStatus.RUNNING  # never silently marked complete


def test_background_tick_also_checks_estimated_completions(session):
    job = session.submit_user_request("user_a", "User A", "job", 1, 5, "HIGH")
    gpu_id = job.assigned_gpu_id
    session.start()

    # At 1x, a background tick now advances simulated time at the same
    # rate as real time (`BACKGROUND_TICK` == `TICK_INTERVAL_SECONDS`
    # - the reclamation timing fix), so reaching 5 simulated minutes
    # takes 5*60/TICK_INTERVAL_SECONDS = 600 ticks, not 5 - well within
    # this loop's margin (each tick is a plain function call, not a
    # real sleep, so 700 iterations still runs in milliseconds).
    for _ in range(700):
        if not session.background_tick():
            break
        if session.simulator.scheduler.reclamation_engine.has_pending_prompt(gpu_id):
            break

    assert session.simulator.scheduler.reclamation_engine.has_pending_prompt(gpu_id) is True
