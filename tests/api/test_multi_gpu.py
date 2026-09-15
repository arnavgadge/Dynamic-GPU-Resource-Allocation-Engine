"""Phase 10: the 10-GPU company pool, multi-GPU (1-9) user requests,
manual admin assignment, and the partial-availability resource-request
flow between two users - exercised end to end through
`SimulationSession`, the same object `api/app.py`'s routes call.

Numbered to match the phase brief's own test list.
"""

from engine.models.enums import GPUStatus, JobStatus
from engine.reclamation.policy import ConfirmationResponse
from engine.simulation import load_default_registry
from api.session import SimulationSession

import pytest


def _session() -> SimulationSession:
    return SimulationSession(load_default_registry(), "interactive_demo")


# -- Test 1: 10-GPU pool, all free -------------------------------------

def test_1_company_starts_with_ten_free_gpus():
    session = _session()
    gpus = list(session.simulator.scheduler.state.gpus.values())
    assert len(gpus) == 10
    assert all(gpu.status == GPUStatus.IDLE and not gpu.is_assigned for gpu in gpus)


# -- Tests 2-5: request bounds -----------------------------------------

def test_2_user_requests_one_gpu_is_valid():
    session = _session()
    job = session.submit_user_request("user_a", "User A", "job", 1, 10, "MEDIUM")
    assert job.status == JobStatus.RUNNING
    assert len(job.assigned_gpu_ids) == 1


def test_3_user_requests_nine_gpus_is_valid():
    session = _session()
    job = session.submit_user_request("user_a", "User A", "job", 9, 10, "MEDIUM")
    assert job.status == JobStatus.RUNNING
    assert len(job.assigned_gpu_ids) == 9


def test_4_user_requests_ten_gpus_is_rejected():
    session = _session()
    with pytest.raises(ValueError, match="gpu_count"):
        session.submit_user_request("user_a", "User A", "job", 10, 10, "MEDIUM")


def test_5_user_requests_zero_gpus_is_rejected():
    session = _session()
    with pytest.raises(ValueError, match="gpu_count"):
        session.submit_user_request("user_a", "User A", "job", 0, 10, "MEDIUM")


# -- Test 6: fully satisfiable multi-GPU request -----------------------

def test_6_two_gpu_request_with_two_free_gpus_allocates_both():
    session = _session()
    job = session.submit_user_request("user_a", "User A", "job", 2, 10, "MEDIUM")
    assert job.status == JobStatus.RUNNING
    assert len(job.assigned_gpu_ids) == 2
    for gpu_id in job.assigned_gpu_ids:
        gpu = session.simulator.scheduler.state.get_gpu(gpu_id)
        assert gpu.status == GPUStatus.ACTIVE
        assert gpu.assigned_user_id == "user_a"


# -- Test 7: partial availability - resource request raised -----------

def _fill_all_but_one(session: SimulationSession, holder: str = "user_c") -> None:
    """Occupy 9 of the 10 GPUs under ``holder``, leaving exactly one free."""
    session.submit_user_request(holder, "User C", "existing work", 9, 500, "MEDIUM")


def test_7_partial_availability_identifies_free_and_candidate_and_asks_holder():
    session = _session()
    _fill_all_but_one(session, holder="user_c")  # GPU-1..9 -> user_c, GPU-10 free

    job = session.submit_user_request("user_b", "User B", "job", 2, 10, "HIGH")

    assert job.status == JobStatus.WAITING
    assert len(job.assigned_gpu_ids) == 1  # the one genuinely free GPU
    assert job.gpus_still_needed == 1

    engine = session.simulator.scheduler.reclamation_engine
    state = session.simulator.scheduler.state
    # Exactly one of user_c's GPUs must now have a resource-request
    # prompt pending, tagged with User B's own request.
    prompted = [
        gpu for gpu in state.gpus.values()
        if gpu.assigned_user_id == "user_c" and engine.has_pending_prompt(gpu.gpu_id)
    ]
    assert len(prompted) == 1
    context = engine.pending_request_context(prompted[0].gpu_id)
    assert context is not None
    assert context.requesting_user_id == "user_b"
    assert context.requesting_job_id == job.job_id

    # Observable in the event log, per the brief's own narrative.
    messages = [e.message for e in state.events]
    assert any("requesting release" in m for m in messages)


# -- Test 8: User C accepts (releases) ---------------------------------

def test_8_holder_releases_and_requester_receives_the_gpu():
    session = _session()
    _fill_all_but_one(session, holder="user_c")
    job = session.submit_user_request("user_b", "User B", "job", 2, 10, "HIGH")

    engine = session.simulator.scheduler.reclamation_engine
    state = session.simulator.scheduler.state
    requested_gpu = next(
        gpu for gpu in state.gpus.values()
        if gpu.assigned_user_id == "user_c" and engine.has_pending_prompt(gpu.gpu_id)
    )

    now = session.simulator.clock.now()
    session.simulator.scheduler.respond_to_prompt(requested_gpu.gpu_id, ConfirmationResponse.NO, now=now)
    session.simulator.scheduler.try_allocate_all(now=now)

    assert requested_gpu.assigned_user_id != "user_c"
    assert job.status == JobStatus.RUNNING
    assert len(job.assigned_gpu_ids) == 2
    assert job.gpus_still_needed == 0
    # user_c really lost it - not just the job pointer.
    assert requested_gpu.gpu_id not in state.get_user("user_c").assigned_gpu_ids


# -- Test 9: User C rejects (keeps) -------------------------------------

def test_9_holder_keeps_gpu_and_requester_stays_partially_waiting():
    session = _session()
    _fill_all_but_one(session, holder="user_c")
    job = session.submit_user_request("user_b", "User B", "job", 2, 10, "HIGH")

    engine = session.simulator.scheduler.reclamation_engine
    state = session.simulator.scheduler.state
    requested_gpu = next(
        gpu for gpu in state.gpus.values()
        if gpu.assigned_user_id == "user_c" and engine.has_pending_prompt(gpu.gpu_id)
    )

    now = session.simulator.clock.now()
    session.simulator.scheduler.respond_to_prompt(requested_gpu.gpu_id, ConfirmationResponse.YES, now=now)
    session.simulator.scheduler.try_allocate_all(now=now)

    # GPU-C keeps their GPU.
    assert requested_gpu.assigned_user_id == "user_c"
    assert requested_gpu.gpu_id in state.get_user("user_c").assigned_gpu_ids

    # User B's job remains consistently short by exactly one GPU -
    # never silently completed, never double-counted.
    assert job.status == JobStatus.WAITING
    assert len(job.assigned_gpu_ids) == 1
    assert job.gpus_still_needed == 1

    # And the same candidate is not immediately re-asked for the same
    # job (a declined request stays declined for that job).
    assert engine.has_declined_for(requested_gpu.gpu_id, job.job_id) is True
    session.simulator.scheduler.try_allocate_all(now=now)
    assert engine.has_pending_prompt(requested_gpu.gpu_id) is False


# -- manual admin assignment (initial test state) ----------------------

def test_manual_assignment_establishes_initial_state_through_the_real_scheduler():
    session = _session()
    job_a = session.manual_assign_gpu("GPU-1", "user_a", "User A")
    job_a2 = session.manual_assign_gpu("GPU-2", "user_a", "User A")
    job_b = session.manual_assign_gpu("GPU-3", "user_b", "User B")

    state = session.simulator.scheduler.state
    assert state.get_gpu("GPU-1").assigned_user_id == "user_a"
    assert state.get_gpu("GPU-2").assigned_user_id == "user_a"
    assert state.get_gpu("GPU-3").assigned_user_id == "user_b"
    assert sorted(state.get_user("user_a").assigned_gpu_ids) == ["GPU-1", "GPU-2"]
    assert job_a.status == JobStatus.RUNNING and job_b.status == JobStatus.RUNNING
    # Remaining GPUs are still genuinely free and allocatable.
    assert state.get_gpu("GPU-4").status == GPUStatus.IDLE


def test_manual_assignment_rejects_an_already_assigned_gpu():
    session = _session()
    session.manual_assign_gpu("GPU-1", "user_a", "User A")
    with pytest.raises(ValueError):
        session.manual_assign_gpu("GPU-1", "user_b", "User B")
