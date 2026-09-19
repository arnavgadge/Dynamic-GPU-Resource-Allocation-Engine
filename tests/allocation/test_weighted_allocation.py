"""Day 5: the core allocation engine + weighted scoring, verified end to
end - score components, the score breakdown a decision exposes, the
Priority Queue (Max-Heap) actually picking the winner, deterministic
tie-breaking, GPU selection (available GPUs only, least-utilized
first), multi-GPU/partial allocation, queue synchronization
(cancel/requeue/priority change), and the edge cases (no jobs, no
GPUs, maintenance/unavailable GPUs, a large waiting queue).

Existing coverage this file does not repeat: `test_scoring.py` (the
normalizations and formula bounds), `test_similarity.py` (the 20%
rule), and `test_engine.py` (the seven scenario-level allocation
tests). Everything here uses deterministic, hand-picked data and goes
through the real `Scheduler`/`AllocationEngine` - nothing mocked.

Two small findings while writing these tests (both fixed, neither a
policy change): `AllocationEngine.requeue_job` enqueued a second copy
of a job that was already in the waiting queue (a partially-allocated
multi-GPU job that lost its last GPU appeared twice), and
`CandidateInfo` only exposed `base_score`, not the priority/size
components behind it - now it exposes both.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.allocation.config import AGING_MAX_CONTRIBUTION, AGING_RATE_PER_MINUTE, PRIORITY_WEIGHT, SIZE_WEIGHT
from engine.allocation.decision import AllocationPolicy
from engine.allocation.scoring import (
    calculate_aging_component,
    calculate_allocation_score,
    calculate_priority_component,
    calculate_size_component,
)
from engine.dsa.priority_queue import PriorityQueue
from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.models.utilization import UtilizationObservation
from engine.scheduler import Scheduler

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def make_scheduler(n_gpus: int, utilizations=None) -> Scheduler:
    scheduler = Scheduler()
    for i in range(1, n_gpus + 1):
        gpu = GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE)
        if utilizations is not None:
            gpu.record_observation(UtilizationObservation(utilization_percent=utilizations[i - 1], timestamp=NOW))
        scheduler.add_gpu(gpu)
    return scheduler


def make_job(job_id, user_id="U1", priority=Priority.MEDIUM, size=30, gpu_count=1, offset_seconds=0) -> Job:
    return Job(job_id=job_id, user_id=user_id, name=job_id, priority=priority,
               estimated_size_minutes=size, gpu_count=gpu_count,
               submitted_at=NOW + timedelta(seconds=offset_seconds))


def ensure_user(scheduler: Scheduler, user_id: str) -> None:
    if scheduler.state.get_user(user_id) is None:
        scheduler.add_user(User(user_id=user_id, name=user_id, priority=Priority.MEDIUM))


def submit(scheduler: Scheduler, job: Job, allocate: bool = True) -> Job:
    ensure_user(scheduler, job.user_id)
    scheduler.submit_job(job, now=job.submitted_at)
    if allocate:
        scheduler.try_allocate_all(now=NOW + timedelta(seconds=1000))
    return job


# ---- 5-8. The four named scoring functions --------------------------------

def test_priority_component():
    assert calculate_priority_component(Priority.LOW) == 0.0
    assert calculate_priority_component(Priority.CRITICAL) == 1.0
    assert calculate_priority_component(Priority.MEDIUM) == pytest.approx(1 / 3)
    assert calculate_priority_component(Priority.HIGH) == pytest.approx(2 / 3)


def test_size_component():
    assert calculate_size_component(10, 10, 100) == 1.0   # smallest candidate
    assert calculate_size_component(100, 10, 100) == 0.0  # largest candidate
    assert calculate_size_component(55, 10, 100) == pytest.approx(0.5)


def test_aging_component_is_linear_then_capped():
    assert calculate_aging_component(0) == 0.0
    assert calculate_aging_component(-5) == 0.0
    assert calculate_aging_component(10) == pytest.approx(10 * AGING_RATE_PER_MINUTE)
    assert calculate_aging_component(10_000) == AGING_MAX_CONTRIBUTION


def test_final_score_is_weighted_priority_plus_weighted_size_plus_aging():
    job = make_job("J1", priority=Priority.HIGH, size=20)
    now = NOW + timedelta(minutes=30)
    breakdown = calculate_allocation_score(job, 20, 200, now)

    expected_base = PRIORITY_WEIGHT * (2 / 3) + SIZE_WEIGHT * 1.0
    assert breakdown.priority_component == pytest.approx(2 / 3)
    assert breakdown.size_component == 1.0
    assert breakdown.base_score == pytest.approx(expected_base)
    assert breakdown.aging_component == pytest.approx(30 * AGING_RATE_PER_MINUTE)
    assert breakdown.final_score == pytest.approx(expected_base + 30 * AGING_RATE_PER_MINUTE)
    assert breakdown.waiting_minutes == pytest.approx(30.0)


# ---- 1/2. Single and multiple waiting jobs --------------------------------

def test_single_waiting_job_is_allocated():
    scheduler = make_scheduler(1)
    job = submit(scheduler, make_job("J1"))
    assert job.status == JobStatus.RUNNING
    assert job.assigned_gpu_ids == ["GPU-1"]


def test_multiple_waiting_jobs_are_all_allocated_when_capacity_allows():
    scheduler = make_scheduler(4)
    jobs = [submit(scheduler, make_job(f"J{i}", user_id=f"U{i}", offset_seconds=i)) for i in range(4)]
    assert all(job.status == JobStatus.RUNNING for job in jobs)
    assert len({job.assigned_gpu_ids[0] for job in jobs}) == 4


# ---- 3. Higher-score job selected -----------------------------------------

def test_higher_score_job_wins_the_only_gpu():
    scheduler = make_scheduler(1)
    # Occupy the GPU first so both contenders are genuinely waiting together.
    holder = submit(scheduler, make_job("HOLD", user_id="UH"))
    low = submit(scheduler, make_job("LOW", user_id="UL", priority=Priority.LOW, size=200, offset_seconds=1))
    high = submit(scheduler, make_job("HIGH", user_id="UX", priority=Priority.HIGH, size=20, offset_seconds=2))

    scheduler.complete_job(holder.job_id, now=NOW + timedelta(seconds=1000))
    decisions = scheduler.try_allocate_all(now=NOW + timedelta(seconds=1000))

    assert decisions[0].job_id == "HIGH"
    assert decisions[0].policy == AllocationPolicy.SCORE_BASED
    assert high.status == JobStatus.RUNNING
    assert low.status == JobStatus.WAITING


# ---- 4. Equal-score deterministic tie --------------------------------------

def test_equal_scores_tie_break_deterministically_by_earliest_submission():
    scheduler = make_scheduler(1)
    holder = submit(scheduler, make_job("HOLD", user_id="UH"))
    # A and B have identical priority+size (so an identical score);
    # C's much larger size forces the score-based branch. B was
    # *submitted* earlier than A even though A was enqueued first.
    a = submit(scheduler, make_job("A", user_id="UA", size=10, offset_seconds=50), allocate=False)
    b = submit(scheduler, make_job("B", user_id="UB", size=10, offset_seconds=5), allocate=False)
    c = submit(scheduler, make_job("C", user_id="UC", size=1000, offset_seconds=60), allocate=False)

    scheduler.complete_job(holder.job_id, now=NOW + timedelta(seconds=70))
    decisions = scheduler.try_allocate_all(now=NOW + timedelta(seconds=70))

    assert decisions[0].policy == AllocationPolicy.SCORE_BASED
    assert decisions[0].job_id == "B"  # earliest submitted_at among the tied top scorers


# ---- 9. Score breakdown on a decision ---------------------------------------

def test_decision_exposes_a_full_score_breakdown_per_candidate():
    scheduler = make_scheduler(1)
    holder = submit(scheduler, make_job("HOLD", user_id="UH"))
    submit(scheduler, make_job("A", user_id="UA", priority=Priority.HIGH, size=20, offset_seconds=1), allocate=False)
    submit(scheduler, make_job("B", user_id="UB", priority=Priority.LOW, size=200, offset_seconds=2), allocate=False)

    scheduler.complete_job(holder.job_id, now=NOW + timedelta(seconds=100))
    decision = scheduler.try_allocate_all(now=NOW + timedelta(seconds=100))[0]

    assert {c.job_id for c in decision.candidates} == {"A", "B"}
    for candidate in decision.candidates:
        assert candidate.user_id in {"UA", "UB"}
        assert candidate.priority in {Priority.HIGH, Priority.LOW}
        assert candidate.size_minutes in {20, 200}
        assert candidate.waiting_time is not None
        assert candidate.priority_component is not None
        assert candidate.size_component is not None
        assert candidate.aging_component is not None
        assert candidate.base_score == pytest.approx(
            PRIORITY_WEIGHT * candidate.priority_component + SIZE_WEIGHT * candidate.size_component
        )
        assert candidate.score == pytest.approx(candidate.base_score + candidate.aging_component)

    winner = next(c for c in decision.candidates if c.job_id == decision.job_id)
    assert all(winner.score >= c.score for c in decision.candidates)


# ---- 10. Priority Queue ordering ---------------------------------------------

def test_priority_queue_orders_jobs_by_allocation_score_descending():
    jobs = [
        make_job("LOW-BIG", priority=Priority.LOW, size=500),
        make_job("HIGH-SMALL", priority=Priority.HIGH, size=10),
        make_job("MED-MID", priority=Priority.MEDIUM, size=100),
    ]
    sizes = [j.estimated_size_minutes for j in jobs]
    lo, hi = min(sizes), max(sizes)
    queue = PriorityQueue(key_fn=lambda j: calculate_allocation_score(j, lo, hi).final_score)
    for job in jobs:
        queue.insert(job)

    assert queue.peek_best().job_id == "HIGH-SMALL"  # O(1) peek
    order = [queue.pop_best().job_id for _ in range(3)]
    assert order == ["HIGH-SMALL", "MED-MID", "LOW-BIG"]


# ---- 11. GPU selection --------------------------------------------------------

def test_least_utilized_available_gpu_is_selected():
    scheduler = make_scheduler(3, utilizations=[80.0, 5.0, 40.0])
    job = submit(scheduler, make_job("J1"))
    assert job.assigned_gpu_ids == ["GPU-2"]


# ---- 12/13. Multi-GPU and partial allocation ---------------------------------

def test_multi_gpu_job_gets_all_requested_gpus_when_available():
    scheduler = make_scheduler(5)
    job = submit(scheduler, make_job("J1", gpu_count=3))
    assert len(job.assigned_gpu_ids) == 3
    assert job.gpus_still_needed == 0
    assert job.status == JobStatus.RUNNING


def test_partial_allocation_progresses_2_3_4_as_gpus_arrive():
    """requested=4: 2 allocated / 2 remaining -> 3 / 1 -> 4 / 0 RUNNING.
    Nothing hardcoded to these numbers - each step just adds a GPU."""
    scheduler = make_scheduler(2)
    job = submit(scheduler, make_job("J1", gpu_count=4))
    assert (job.gpu_count, len(job.assigned_gpu_ids), job.gpus_still_needed) == (4, 2, 2)
    assert job.status == JobStatus.WAITING
    assert scheduler.allocation_engine.waiting_job_ids() == ["J1"]  # one entry, not one per missing GPU

    scheduler.add_gpu(GPU(gpu_id="GPU-3", total_memory_mb=1000, status=GPUStatus.IDLE))
    scheduler.try_allocate_all(now=NOW + timedelta(seconds=2000))
    assert (len(job.assigned_gpu_ids), job.gpus_still_needed) == (3, 1)
    assert job.status == JobStatus.WAITING

    scheduler.add_gpu(GPU(gpu_id="GPU-4", total_memory_mb=1000, status=GPUStatus.IDLE))
    scheduler.try_allocate_all(now=NOW + timedelta(seconds=3000))
    assert (len(job.assigned_gpu_ids), job.gpus_still_needed) == (4, 0)
    assert job.status == JobStatus.RUNNING
    assert scheduler.allocation_engine.waiting_job_ids() == []


# ---- 14-16. No GPU / maintenance / unavailable ---------------------------------

def test_no_gpu_available_leaves_the_job_waiting_without_crashing():
    scheduler = make_scheduler(0)
    job = submit(scheduler, make_job("J1"))
    assert job.status == JobStatus.WAITING
    assert scheduler.try_allocate_all(now=NOW) == []


def test_no_waiting_jobs_is_a_clean_no_op():
    scheduler = make_scheduler(3)
    assert scheduler.try_allocate_all(now=NOW) == []
    assert scheduler.allocation_engine.select_next_job() is None


def test_maintenance_gpu_is_excluded_from_allocation():
    scheduler = make_scheduler(2)
    scheduler.set_gpu_maintenance("GPU-1", now=NOW)
    job = submit(scheduler, make_job("J1", gpu_count=2))
    assert job.assigned_gpu_ids == ["GPU-2"]  # GPU-1 never handed out
    assert job.status == JobStatus.WAITING


def test_unavailable_gpu_is_excluded_from_allocation():
    scheduler = make_scheduler(2)
    scheduler.handle_gpu_failure("GPU-2", now=NOW)
    job = submit(scheduler, make_job("J1", gpu_count=2))
    assert job.assigned_gpu_ids == ["GPU-1"]
    assert scheduler.state.get_gpu("GPU-2").status == GPUStatus.UNAVAILABLE


# ---- 17. Cancelled job excluded ------------------------------------------------

def test_cancelled_job_is_never_allocated():
    scheduler = make_scheduler(1)
    holder = submit(scheduler, make_job("HOLD", user_id="UH"))
    cancelled = submit(scheduler, make_job("CANCEL", user_id="UC", offset_seconds=1))
    kept = submit(scheduler, make_job("KEEP", user_id="UK", offset_seconds=2))

    scheduler.cancel_job("CANCEL", now=NOW + timedelta(seconds=10))
    scheduler.complete_job(holder.job_id, now=NOW + timedelta(seconds=20))
    scheduler.try_allocate_all(now=NOW + timedelta(seconds=20))

    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.assigned_gpu_ids == []
    assert kept.status == JobStatus.RUNNING


def test_a_stale_cancelled_entry_left_in_the_queue_is_still_excluded():
    """Defense in depth (edge case 12): even if a job's status flips
    to CANCELLED without going through `cancel_job` (so it is still
    physically in the FIFO), selection filters on `JobStatus.WAITING`
    and never offers it."""
    scheduler = make_scheduler(1)
    holder = submit(scheduler, make_job("HOLD", user_id="UH"))
    ghost = submit(scheduler, make_job("GHOST", user_id="UG", offset_seconds=1))
    ghost.status = JobStatus.CANCELLED  # bypasses cancel_job on purpose

    assert "GHOST" not in scheduler.allocation_engine.waiting_job_ids()
    scheduler.complete_job(holder.job_id, now=NOW + timedelta(seconds=20))
    assert scheduler.try_allocate_all(now=NOW + timedelta(seconds=20)) == []
    assert ghost.assigned_gpu_ids == []


# ---- 18. Requeue / priority change keep the queue synchronized ------------------

def test_requeued_job_re_enters_the_queue_exactly_once():
    scheduler = make_scheduler(1)
    job = submit(scheduler, make_job("J1"))
    assert scheduler.allocation_engine.waiting_job_ids() == []

    scheduler.handle_gpu_failure("GPU-1", now=NOW + timedelta(seconds=5))  # loses its only GPU
    assert scheduler.allocation_engine.waiting_job_ids() == ["J1"]
    assert job.status == JobStatus.WAITING


def test_requeue_of_an_already_queued_partial_job_does_not_duplicate_it():
    """Regression for the duplicate-entry bug: a partially-allocated
    multi-GPU job is WAITING and already queued; when it loses the
    one GPU it held, requeueing must not add a second copy."""
    scheduler = make_scheduler(1)
    job = submit(scheduler, make_job("J1", gpu_count=3))
    assert job.assigned_gpu_ids == ["GPU-1"]
    assert scheduler.allocation_engine.waiting_job_ids() == ["J1"]

    scheduler.handle_gpu_failure("GPU-1", now=NOW + timedelta(seconds=5))

    assert scheduler.allocation_engine.waiting_job_ids() == ["J1"]  # exactly once, not ['J1', 'J1']
    assert job.assigned_gpu_ids == []
    assert job.gpus_still_needed == 3


def test_priority_change_is_reflected_on_the_very_next_decision():
    scheduler = make_scheduler(1)
    holder = submit(scheduler, make_job("HOLD", user_id="UH"))
    submit(scheduler, make_job("A", user_id="UA", priority=Priority.HIGH, size=20, offset_seconds=1), allocate=False)
    b = submit(scheduler, make_job("B", user_id="UB", priority=Priority.LOW, size=200, offset_seconds=2), allocate=False)

    scheduler.change_job_priority("B", Priority.CRITICAL, now=NOW + timedelta(seconds=10))
    scheduler.complete_job(holder.job_id, now=NOW + timedelta(seconds=20))
    scheduler.try_allocate_all(now=NOW + timedelta(seconds=20))

    assert b.status == JobStatus.RUNNING  # CRITICAL now outranks A


# ---- 19/20/22/23. State consistency, mappings, no duplicates, no negatives ------

def test_state_and_mappings_are_consistent_after_allocation():
    scheduler = make_scheduler(4)
    jobs = [
        submit(scheduler, make_job("J1", user_id="UA", gpu_count=2)),
        submit(scheduler, make_job("J2", user_id="UB", offset_seconds=1)),
    ]
    state = scheduler.state
    for job in jobs:
        user = state.get_user(job.user_id)
        assert sorted(user.assigned_gpu_ids) == sorted(job.assigned_gpu_ids)
        assert job.job_id in user.running_job_ids
        assert sorted(scheduler.allocation_engine.get_gpus_for_user(job.user_id)) == sorted(job.assigned_gpu_ids)
        for gpu_id in job.assigned_gpu_ids:
            gpu = state.get_gpu(gpu_id)
            assert gpu.assigned_job_id == job.job_id
            assert gpu.assigned_user_id == job.user_id
            assert gpu.status == GPUStatus.ACTIVE
            assert state.get_job_on_gpu(gpu_id) is job


def test_no_gpu_is_ever_allocated_to_two_jobs_and_no_count_goes_negative():
    scheduler = make_scheduler(6)
    jobs = [
        submit(scheduler, make_job(f"J{i}", user_id=f"U{i}", gpu_count=1 + (i % 3), offset_seconds=i))
        for i in range(6)
    ]
    owners = {}
    for job in jobs:
        assert job.gpus_still_needed >= 0
        assert len(job.assigned_gpu_ids) == len(set(job.assigned_gpu_ids))
        for gpu_id in job.assigned_gpu_ids:
            assert gpu_id not in owners, f"{gpu_id} given to both {owners.get(gpu_id)} and {job.job_id}"
            owners[gpu_id] = job.job_id
    # Repeated passes never double-allocate either.
    for _ in range(3):
        scheduler.try_allocate_all(now=NOW + timedelta(seconds=5000))
    for job in jobs:
        assert job.gpus_still_needed >= 0
        assert len(job.assigned_gpu_ids) <= job.gpu_count


# ---- 21. Large waiting queue ------------------------------------------------------

def test_large_waiting_queue_allocates_exactly_the_free_capacity():
    scheduler = make_scheduler(5)
    jobs = [
        submit(scheduler, make_job(f"J{i}", user_id=f"U{i % 25}", priority=Priority((i % 4) + 1),
                                    size=10 + (i % 7) * 40, offset_seconds=i), allocate=False)
        for i in range(200)
    ]
    scheduler.try_allocate_all(now=NOW + timedelta(seconds=500))

    running = [j for j in jobs if j.status == JobStatus.RUNNING]
    waiting_ids = scheduler.allocation_engine.waiting_job_ids()
    assert len(running) == 5
    assert len(waiting_ids) == 195
    assert len(waiting_ids) == len(set(waiting_ids))  # no duplicates
    assert len({j.assigned_gpu_ids[0] for j in running}) == 5
    assert all(j.priority == Priority.CRITICAL for j in running)  # the top scorers actually won
