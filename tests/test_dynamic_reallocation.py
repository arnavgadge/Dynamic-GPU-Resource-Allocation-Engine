"""Issue 7: dynamic partial GPU reallocation from an underutilized
holder to a waiting multi-GPU request - generalized for an arbitrary
number of users/GPUs, reusing the existing allocation/reclamation
policy rather than any user/GPU-specific branching.

Built directly on `Scheduler` (the same pattern
`tests/balancing/test_integration.py` already uses) for tight control
over exactly which GPUs are "actively used" vs "underutilized" -
`Scheduler.record_utilization` is the one real entry point that
reports utilization, exactly like a scenario's `UtilizationAction` or
the hardware poller would.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.policy import ConfirmationResponse
from engine.scheduler import Scheduler

START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def _make_pool(gpu_count: int) -> Scheduler:
    scheduler = Scheduler()
    for i in range(1, gpu_count + 1):
        scheduler.add_gpu(GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE))
    return scheduler


def _request(scheduler: Scheduler, user_id: str, name: str, gpu_count: int, now: datetime, priority=Priority.MEDIUM) -> Job:
    if scheduler.state.get_user(user_id) is None:
        scheduler.add_user(User(user_id=user_id, name=user_id, priority=priority))
    job = Job(
        job_id=f"{user_id}-{name}", user_id=user_id, name=name, priority=priority,
        estimated_size_minutes=120, gpu_count=gpu_count, submitted_at=now,
    )
    scheduler.submit_job(job, now=now)
    scheduler.try_allocate_all(now=now)
    return job


# -- Test 4: partial reallocation identified ----------------------------

def test_4_underutilized_gpus_identified_as_reallocation_candidates_rest_pending():
    scheduler = _make_pool(5)
    now = START

    job_a = _request(scheduler, "user_a", "A-work", 4, now)
    assert job_a.status == JobStatus.RUNNING
    assert len(job_a.assigned_gpu_ids) == 4  # 4 of 5 GPUs, 1 free left

    # A actively uses only 2 of its 4 GPUs - report real utilization
    # for those two; the other two stay at their default 0% (never
    # reported => genuinely not being used).
    active, idle = job_a.assigned_gpu_ids[:2], job_a.assigned_gpu_ids[2:]
    for gpu_id in active:
        scheduler.record_utilization(gpu_id, 85.0, now)

    job_b = _request(scheduler, "user_b", "B-work", 3, now)

    # 1 GPU was genuinely free -> allocated immediately. 2 more are
    # needed; A's two underutilized GPUs are exactly the candidates -
    # both identified and prompted, never A's two actively-used ones.
    assert job_b.status == JobStatus.WAITING
    assert len(job_b.assigned_gpu_ids) == 1  # the one free GPU
    assert job_b.gpus_still_needed == 2

    engine = scheduler.reclamation_engine
    prompted = [gid for gid in idle if engine.has_pending_prompt(gid)]
    assert len(prompted) == 2  # both of A's underutilized GPUs
    for gpu_id in active:
        assert engine.has_pending_prompt(gpu_id) is False  # never touches the actively-used ones

    for gpu_id in prompted:
        context = engine.pending_request_context(gpu_id)
        assert context.requesting_user_id == "user_b"


# -- Test 5: completion triggers reallocation ----------------------------

def test_5_a_releasing_remaining_gpus_completes_bs_request_one_at_a_time():
    scheduler = _make_pool(5)
    now = START

    job_a = _request(scheduler, "user_a", "A-work", 4, now)
    active, idle = job_a.assigned_gpu_ids[:2], job_a.assigned_gpu_ids[2:]
    for gpu_id in active:
        scheduler.record_utilization(gpu_id, 85.0, now)

    job_b = _request(scheduler, "user_b", "B-work", 3, now)
    assert len(job_b.assigned_gpu_ids) == 1  # only the genuinely free GPU so far

    engine = scheduler.reclamation_engine
    first, second = idle[0], idle[1]
    assert engine.has_pending_prompt(first) and engine.has_pending_prompt(second)

    # A's work on the first underutilized GPU finishes -> released ->
    # B's still-waiting request is reevaluated and receives it (2/3).
    scheduler.respond_to_prompt(first, ConfirmationResponse.NO, now=now)
    scheduler.try_allocate_all(now=now)
    assert job_b.status == JobStatus.WAITING
    assert len(job_b.assigned_gpu_ids) == 2
    assert first in job_b.assigned_gpu_ids
    assert first not in scheduler.state.get_user("user_a").assigned_gpu_ids

    # The second one finishes too -> B reaches its full request (3/3).
    scheduler.respond_to_prompt(second, ConfirmationResponse.NO, now=now)
    scheduler.try_allocate_all(now=now)
    assert job_b.status == JobStatus.RUNNING
    assert len(job_b.assigned_gpu_ids) == 3
    assert job_b.gpus_still_needed == 0
    assert second in job_b.assigned_gpu_ids


# -- Test 6: state synchronization (single SchedulerState) ---------------

def test_6_admin_a_and_b_all_read_the_one_consistent_state():
    scheduler = _make_pool(5)
    now = START
    job_a = _request(scheduler, "user_a", "A-work", 4, now)
    for gpu_id in job_a.assigned_gpu_ids[:2]:
        scheduler.record_utilization(gpu_id, 85.0, now)
    job_b = _request(scheduler, "user_b", "B-work", 3, now)

    # "Admin", "user A" and "user B" are just three different reads of
    # the exact same object - never separate copies.
    state = scheduler.state
    a_gpus = sorted(state.get_gpus_for_user("user_a"), key=lambda g: g.gpu_id)
    b_gpus = sorted(state.get_gpus_for_user("user_b"), key=lambda g: g.gpu_id)
    assert {g.gpu_id for g in a_gpus} == set(job_a.assigned_gpu_ids)
    assert {g.gpu_id for g in b_gpus} == set(job_b.assigned_gpu_ids)
    assert set(g.gpu_id for g in a_gpus) & set(g.gpu_id for g in b_gpus) == set()


# -- Test 7: arbitrary number of users, no hardcoded ids -----------------

def test_7_works_for_arbitrary_users_not_hardcoded_to_a_b_c_d():
    scheduler = _make_pool(5)
    now = START

    job_alpha = _request(scheduler, "alpha", "alpha-work", 3, now)
    for gpu_id in job_alpha.assigned_gpu_ids[:1]:
        scheduler.record_utilization(gpu_id, 90.0, now)
    _request(scheduler, "beta", "beta-work", 1, now)
    job_gamma = _request(scheduler, "gamma", "gamma-work", 1, now)  # 5th GPU
    assert job_gamma.status == JobStatus.RUNNING

    job_delta = _request(scheduler, "delta", "delta-work", 2, now)
    # Nothing free at all now; alpha holds 2 underutilized GPUs (never
    # reported utilization) among its 3.
    assert job_delta.status == JobStatus.WAITING
    engine = scheduler.reclamation_engine
    prompted_gpus = [g for g in scheduler.state.gpus.values() if engine.has_pending_prompt(g.gpu_id)]
    assert len(prompted_gpus) == 2
    for gpu in prompted_gpus:
        assert gpu.assigned_user_id == "alpha"
        context = engine.pending_request_context(gpu.gpu_id)
        assert context.requesting_user_id == "delta"


# -- Test 8: different allocation counts within a 5-GPU pool -------------

@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_8_various_request_sizes_allocate_correctly(count):
    scheduler = _make_pool(5)
    now = START
    job = _request(scheduler, "solo", "job", count, now)
    assert job.status == JobStatus.RUNNING
    assert len(job.assigned_gpu_ids) == count


def test_orphaned_resource_request_is_withdrawn_once_the_job_is_fully_satisfied():
    """Regression: caught via a live end-to-end run against the real
    FastAPI app. With three candidates for a two-GPU deficit, two get
    prompted first; releasing one completes 2/3 and immediately raises
    a *fresh* request for the last slot against a third candidate,
    leaving the still-pending original second request now unneeded.
    Left alone, that second GPU would sit in IDLE_WARNING asking for
    capacity nobody needs anymore.
    """
    scheduler = _make_pool(6)
    now = START

    job_a = _request(scheduler, "user_a", "A-work", 4, now, priority=Priority.HIGH)
    for gpu_id in job_a.assigned_gpu_ids[:2]:
        scheduler.record_utilization(gpu_id, 85.0, now)
    idle_a = job_a.assigned_gpu_ids[2:]  # 2 underutilized GPUs, HIGH-priority holder

    ghost_job = _request(scheduler, "ghost", "ghost-work", 1, now, priority=Priority.LOW)
    # ghost's GPU is also underutilized (never reported) and LOW
    # priority - the lowest-priority candidate, asked first.
    ghost_gpu = ghost_job.assigned_gpu_ids[0]

    job_b = _request(scheduler, "user_b", "B-work", 3, now)  # 1 free GPU left in the 6-pool
    assert len(job_b.assigned_gpu_ids) == 1
    assert job_b.gpus_still_needed == 2

    engine = scheduler.reclamation_engine
    all_candidates = [ghost_gpu] + idle_a
    prompted = [gid for gid in all_candidates if engine.has_pending_prompt(gid)]
    assert len(prompted) == 2
    assert ghost_gpu in prompted  # lowest priority -> asked first

    first = ghost_gpu
    still_pending_original = [gid for gid in prompted if gid != first][0]

    # Releasing the lowest-priority holder's GPU completes 2/3 and
    # immediately raises a *new* request for the last slot against
    # the next candidate (the still-pending original wasn't re-picked
    # - it's already accounted for as outstanding).
    scheduler.respond_to_prompt(first, ConfirmationResponse.NO, now=now)
    scheduler.try_allocate_all(now=now)
    assert len(job_b.assigned_gpu_ids) == 2
    assert job_b.gpus_still_needed == 1

    newly_prompted = [
        gid for gid in idle_a if gid != still_pending_original and engine.has_pending_prompt(gid)
    ]
    assert len(newly_prompted) == 1
    second_new = newly_prompted[0]
    assert engine.has_pending_prompt(still_pending_original) is True  # the original ask is still outstanding too

    # Whichever of the two outstanding requests resolves first
    # completes B's job - here, the newer one.
    scheduler.respond_to_prompt(second_new, ConfirmationResponse.NO, now=now)
    scheduler.try_allocate_all(now=now)

    assert job_b.is_fully_allocated
    # The original, now-unneeded request must have been withdrawn -
    # not left pending forever on a GPU nobody needs anymore.
    assert engine.has_pending_prompt(still_pending_original) is False
    gpu_orphan = scheduler.state.get_gpu(still_pending_original)
    assert gpu_orphan.status == GPUStatus.ACTIVE
    assert gpu_orphan.assigned_user_id == "user_a"  # untouched - never reclaimed for nothing


def test_8_partial_allocation_across_sizes_when_pool_is_short():
    scheduler = _make_pool(5)
    now = START
    job_a = _request(scheduler, "user_a", "A-work", 4, now)
    for gpu_id in job_a.assigned_gpu_ids[:3]:  # 3 actively used, 1 underutilized
        scheduler.record_utilization(gpu_id, 60.0, now)

    job_b = _request(scheduler, "user_b", "B-work", 5, now)  # wants all 5; only 1 free + 1 underutilized reachable
    assert job_b.status == JobStatus.WAITING
    assert len(job_b.assigned_gpu_ids) == 1  # the one genuinely free GPU
    assert job_b.gpus_still_needed == 4

    engine = scheduler.reclamation_engine
    prompted = [g for g in scheduler.state.gpus.values() if engine.has_pending_prompt(g.gpu_id)]
    assert len(prompted) == 1  # only A's single underutilized GPU qualifies
