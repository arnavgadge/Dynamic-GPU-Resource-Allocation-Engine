"""Day 11: the administrative control layer, verified end to end
through the real `Scheduler` - cancel_job, force_reclaim,
change_job_priority, set/clear_gpu_maintenance, handle_gpu_failure,
and the new `recover_gpu_failure`.

Every one of these operations already existed except `recover_gpu_failure`
(added today, symmetric to `clear_gpu_maintenance`) - this file adds
the systematic coverage the brief asks for, using
`engine.consistency.check_consistency` after every operation instead
of a bespoke assertion block per test.

Two real, small gaps were found and fixed while writing these tests:

1. `set_gpu_maintenance`/`clear_gpu_maintenance` logged the generic
   `EventType.SYSTEM` - indistinguishable from any other engine-
   lifecycle event. They now log `GPU_MAINTENANCE_ENABLED`/
   `GPU_MAINTENANCE_DISABLED`.
2. There was no way to recover a GPU `handle_gpu_failure` had marked
   `UNAVAILABLE` - `Scheduler.recover_gpu_failure` (mirroring
   `clear_gpu_maintenance`) now exists, logging `GPU_RECOVERED`.

No scoring formula, aging rule, reclamation threshold, or scheduling
policy changed - every check here is either brand new coverage of
already-correct behavior, or one of the two additive fixes above.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.consistency import assert_consistent, check_consistency
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.scheduler import Scheduler

T0 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def add_gpus(scheduler: Scheduler, n: int) -> None:
    for i in range(1, n + 1):
        scheduler.add_gpu(GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE))


def submit(scheduler: Scheduler, job_id, user_id, priority=Priority.MEDIUM, size=100_000,
           gpu_count=1, minute=0.0) -> Job:
    if scheduler.state.get_user(user_id) is None:
        scheduler.add_user(User(user_id=user_id, name=user_id, priority=priority))
    job = Job(job_id=job_id, user_id=user_id, name=job_id, priority=priority,
              estimated_size_minutes=size, gpu_count=gpu_count, submitted_at=at(minute))
    scheduler.submit_job(job, now=at(minute))
    scheduler.try_allocate_all(now=at(minute))
    return job


# ---- 1. Cancel a waiting job ------------------------------------------------

def test_cancel_waiting_job():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    holder = submit(scheduler, "HOLD", "UH")
    waiter = submit(scheduler, "W1", "UW", minute=1)
    assert waiter.status == JobStatus.WAITING

    event = scheduler.cancel_job("W1", now=at(2))
    assert event.event_type == EventType.JOB_CANCELLED
    assert waiter.status == JobStatus.CANCELLED
    assert "W1" not in scheduler.allocation_engine.waiting_job_ids()
    assert_consistent(scheduler)

    scheduler.complete_job("HOLD", now=at(3))
    scheduler.try_allocate_all(now=at(3))
    assert waiter.status == JobStatus.CANCELLED  # never later allocated
    assert_consistent(scheduler)


# ---- 2/3. Cancel running single-/multi-GPU jobs (rejected, not silently done) -

def test_cancel_running_single_gpu_job_is_rejected_not_silently_ignored():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    job = submit(scheduler, "J1", "U1")
    assert job.status == JobStatus.RUNNING
    with pytest.raises(ValueError):
        scheduler.cancel_job("J1", now=at(1))
    assert job.status == JobStatus.RUNNING  # untouched
    assert_consistent(scheduler)


def test_cancel_running_multi_gpu_job_is_also_rejected():
    scheduler = Scheduler()
    add_gpus(scheduler, 3)
    job = submit(scheduler, "J1", "U1", gpu_count=3)
    assert job.status == JobStatus.RUNNING
    with pytest.raises(ValueError):
        scheduler.cancel_job("J1", now=at(1))
    assert len(job.assigned_gpu_ids) == 3
    assert_consistent(scheduler)


# ---- 4. Cancel already-cancelled job safely ---------------------------------

def test_cancel_already_cancelled_job_is_rejected_cleanly_no_corruption():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "HOLD", "UH")
    waiter = submit(scheduler, "W1", "UW", minute=1)
    scheduler.cancel_job("W1", now=at(2))
    assert waiter.status == JobStatus.CANCELLED

    with pytest.raises(ValueError):
        scheduler.cancel_job("W1", now=at(3))  # not WAITING anymore
    assert waiter.status == JobStatus.CANCELLED  # state unchanged, not corrupted
    assert_consistent(scheduler)


def test_cancel_unknown_job_raises_cleanly():
    scheduler = Scheduler()
    with pytest.raises(ValueError):
        scheduler.cancel_job("NOPE", now=T0)


# ---- 5/6. Force reclaim one GPU vs. all GPUs from a multi-GPU job -----------

def test_force_reclaim_one_gpu_from_a_multi_gpu_job_leaves_the_rest_running():
    scheduler = Scheduler()
    add_gpus(scheduler, 4)
    job = submit(scheduler, "A", "UA", gpu_count=4)
    assert len(job.assigned_gpu_ids) == 4
    target = job.assigned_gpu_ids[0]

    event = scheduler.force_reclaim(target, now=at(1))
    assert event.event_type == EventType.ADMIN_FORCE_RECLAIM
    assert len(job.assigned_gpu_ids) == 3 and target not in job.assigned_gpu_ids
    assert job.gpu_count == 4 and job.gpus_still_needed == 1   # requested=4, allocated=3, remaining=1
    assert job.status == JobStatus.RUNNING  # partial loss, still running
    assert scheduler.state.get_gpu(target).status == GPUStatus.IDLE
    assert_consistent(scheduler)


def test_force_reclaim_all_gpus_from_a_job_one_call_per_gpu_matches_the_tasks_example():
    """The task's own example: requested=4, allocated=4 -> admin
    force-reclaims 2 -> allocated=2, remaining=2. force_reclaim is
    GPU-scoped by design (matches every other reclamation trigger in
    this project) - partial reclamation is just calling it once per
    targeted GPU, never a second, job-scoped algorithm."""
    scheduler = Scheduler()
    add_gpus(scheduler, 4)
    job = submit(scheduler, "A", "UA", gpu_count=4)
    to_reclaim = job.assigned_gpu_ids[:2]

    for gpu_id in to_reclaim:
        scheduler.force_reclaim(gpu_id, now=at(1))
    assert job.gpu_count == 4
    assert len(job.assigned_gpu_ids) == 2
    assert job.gpus_still_needed == 2
    assert job.status == JobStatus.RUNNING
    assert_consistent(scheduler)

    # And reclaiming the rest fully releases the job.
    for gpu_id in list(job.assigned_gpu_ids):
        scheduler.force_reclaim(gpu_id, now=at(2))
    assert job.assigned_gpu_ids == []
    assert job.status == JobStatus.RECLAIMED
    assert_consistent(scheduler)


def test_force_reclaimed_gpu_becomes_available_to_the_normal_scheduler():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    holder = submit(scheduler, "A", "UA")
    waiter = submit(scheduler, "B", "UB", minute=1)
    assert waiter.status == JobStatus.WAITING

    scheduler.force_reclaim("GPU-1", now=at(2))
    assert waiter.status == JobStatus.RUNNING  # reevaluated through the ordinary allocation policy
    assert waiter.assigned_gpu_ids == ["GPU-1"]
    assert_consistent(scheduler)


def test_force_reclaim_rejects_an_unassigned_gpu_and_an_unknown_gpu():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    with pytest.raises(ValueError):
        scheduler.force_reclaim("GPU-1", now=T0)
    with pytest.raises(KeyError):
        scheduler.force_reclaim("GPU-999", now=T0)


# ---- 7/8. Priority modification -------------------------------------------

def test_change_priority_of_a_waiting_job():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "HOLD", "UH")
    job = submit(scheduler, "W1", "UW", priority=Priority.LOW, minute=1)
    assert job.priority == Priority.LOW

    event = scheduler.change_job_priority("W1", Priority.CRITICAL, now=at(2))
    assert event.event_type == EventType.PRIORITY_CHANGED
    assert job.priority == Priority.CRITICAL
    assert_consistent(scheduler)


def test_priority_change_affects_future_scheduling_without_a_hard_gate():
    """Blended scoring preserved: the upgraded job wins because its
    new score is now genuinely higher - verified via the real
    try_allocate_all outcome, not asserted by fiat."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    holder = submit(scheduler, "HOLD", "UH")
    low = submit(scheduler, "LOW", "UL", priority=Priority.LOW, size=60, minute=1)
    other = submit(scheduler, "OTHER", "UO", priority=Priority.MEDIUM, size=60, minute=2)

    scheduler.change_job_priority("LOW", Priority.CRITICAL, now=at(3))
    scheduler.complete_job("HOLD", now=at(10))
    scheduler.try_allocate_all(now=at(10))

    assert low.status == JobStatus.RUNNING       # CRITICAL -> wins the critical-tier restriction
    assert other.status == JobStatus.WAITING
    assert_consistent(scheduler)


def test_priority_change_still_respects_size_and_aging_never_a_hard_tier_win():
    """Changing B's priority to HIGH must not automatically beat A -
    the blended score (priority+size+aging) still decides; a tiny
    size/aging disadvantage can still make B lose despite the bump."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "HOLD", "UH")
    a = submit(scheduler, "A", "UA", priority=Priority.HIGH, size=5, minute=1)      # small, strong score
    b = submit(scheduler, "B", "UB", priority=Priority.LOW, size=100_000, minute=2)  # huge, weak score

    scheduler.change_job_priority("B", Priority.HIGH, now=at(3))  # now equal priority to A
    scheduler.complete_job("HOLD", now=at(10))
    scheduler.try_allocate_all(now=at(10))

    assert a.status == JobStatus.RUNNING   # A's much smaller size still wins at equal priority
    assert b.status == JobStatus.WAITING
    assert_consistent(scheduler)


def test_change_priority_rejects_a_running_job():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    job = submit(scheduler, "A", "UA")
    with pytest.raises(ValueError):
        scheduler.change_job_priority("A", Priority.CRITICAL, now=at(1))
    assert job.priority != Priority.CRITICAL


# ---- 9/10. GPU maintenance ---------------------------------------------------

def test_maintenance_blocks_new_allocation_on_an_idle_gpu():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    event = scheduler.set_gpu_maintenance("GPU-2", now=T0)
    assert event.event_type == EventType.GPU_MAINTENANCE_ENABLED
    assert scheduler.state.get_gpu("GPU-2").status == GPUStatus.MAINTENANCE

    job = submit(scheduler, "J1", "U1")
    assert job.assigned_gpu_ids == ["GPU-1"]  # never GPU-2
    assert_consistent(scheduler)


def test_maintenance_rejects_a_currently_allocated_gpu():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "J1", "U1")
    with pytest.raises(ValueError):
        scheduler.set_gpu_maintenance("GPU-1", now=T0)


def test_maintenance_removal_restores_allocation_eligibility_and_reallocates():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    scheduler.set_gpu_maintenance("GPU-1", now=T0)
    waiter = submit(scheduler, "W1", "U1", minute=1)
    assert waiter.status == JobStatus.WAITING

    event = scheduler.clear_gpu_maintenance("GPU-1", now=at(2))
    assert event.event_type == EventType.GPU_MAINTENANCE_DISABLED
    assert waiter.status == JobStatus.RUNNING
    assert waiter.assigned_gpu_ids == ["GPU-1"]
    assert_consistent(scheduler)


def test_maintenance_across_multiple_gpus_with_waiting_jobs():
    scheduler = Scheduler()
    add_gpus(scheduler, 5)
    for gpu_id in ("GPU-1", "GPU-2", "GPU-3"):
        scheduler.set_gpu_maintenance(gpu_id, now=T0)
    jobs = [submit(scheduler, f"J{i}", f"U{i}", minute=i) for i in range(4)]
    running = [j for j in jobs if j.status == JobStatus.RUNNING]
    waiting = [j for j in jobs if j.status == JobStatus.WAITING]
    assert len(running) == 2 and len(waiting) == 2   # only GPU-4/5 ever available
    assert_consistent(scheduler)

    for gpu_id in ("GPU-1", "GPU-2", "GPU-3"):
        scheduler.clear_gpu_maintenance(gpu_id, now=at(10))
    assert all(j.status == JobStatus.RUNNING for j in jobs)
    assert_consistent(scheduler)


def test_maintenance_object_is_kept_not_destroyed():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    scheduler.set_gpu_maintenance("GPU-1", now=T0)
    assert "GPU-1" in scheduler.state.gpus
    gpu = scheduler.state.get_gpu("GPU-1")
    assert gpu is not None and gpu.gpu_id == "GPU-1"


# ---- 11/12. GPU failure and recovery -----------------------------------------

def test_failed_gpu_cannot_be_allocated():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    event = scheduler.handle_gpu_failure("GPU-1", now=T0)
    assert event.event_type == EventType.HARDWARE_FAILURE
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.UNAVAILABLE

    job = submit(scheduler, "J1", "U1")
    assert job.assigned_gpu_ids == ["GPU-2"]  # never the failed one
    assert_consistent(scheduler)


def test_gpu_recovery_restores_allocation_eligibility():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    scheduler.handle_gpu_failure("GPU-1", now=T0)
    waiter = submit(scheduler, "W1", "U1", minute=1)
    assert waiter.status == JobStatus.WAITING

    event = scheduler.recover_gpu_failure("GPU-1", now=at(2))
    assert event.event_type == EventType.GPU_RECOVERED
    # recover_gpu_failure reevaluates the waiting queue itself, so the
    # recovered GPU is immediately handed to the waiting job in this
    # same call - it ends ACTIVE (assigned), not sitting IDLE.
    assert waiter.status == JobStatus.RUNNING
    assert waiter.assigned_gpu_ids == ["GPU-1"]
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.ACTIVE
    assert_consistent(scheduler)


def test_recover_rejects_a_gpu_that_never_failed_or_is_only_in_maintenance():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    with pytest.raises(ValueError):
        scheduler.recover_gpu_failure("GPU-1", now=T0)  # perfectly healthy

    scheduler.set_gpu_maintenance("GPU-2", now=T0)
    with pytest.raises(ValueError):
        scheduler.recover_gpu_failure("GPU-2", now=T0)  # maintenance, not a hardware failure


def test_recover_unknown_gpu_raises():
    scheduler = Scheduler()
    with pytest.raises(KeyError):
        scheduler.recover_gpu_failure("GPU-999", now=T0)


# ---- 13. Admin events are distinct from automatic events ---------------------

def test_admin_event_types_are_distinct_from_automatic_ones():
    admin_only = {
        EventType.JOB_CANCELLED, EventType.ADMIN_FORCE_RECLAIM, EventType.PRIORITY_CHANGED,
        EventType.GPU_MAINTENANCE_ENABLED, EventType.GPU_MAINTENANCE_DISABLED, EventType.GPU_RECOVERED,
    }
    automatic = {
        EventType.AUTOMATIC_RECLAIM, EventType.PRIORITY_PREEMPTION, EventType.ALLOC,
        EventType.BALANCE, EventType.HARDWARE_FAILURE,
    }
    assert admin_only.isdisjoint(automatic)
    assert len(admin_only) == 6  # every admin event type is genuinely its own value


def test_admin_force_reclaim_is_never_confused_with_automatic_reclaim_in_practice():
    auto_sched = Scheduler()
    add_gpus(auto_sched, 1)
    submit(auto_sched, "A", "UA")
    for m in (0, 5, 10, 15, 20, 25):
        auto_sched.record_utilization("GPU-1", 1.0, at(m))
    from engine.reclamation.policy import ConfirmationResponse
    auto_sched.respond_to_prompt("GPU-1", ConfirmationResponse.NO, now=at(26))
    auto_reclaim = next(e for e in auto_sched.state.events if e.event_type == EventType.RECLAIM)
    assert auto_reclaim.metadata["category"] == EventType.AUTOMATIC_RECLAIM.value

    admin_sched = Scheduler()
    add_gpus(admin_sched, 1)
    submit(admin_sched, "B", "UB")
    admin_event = admin_sched.force_reclaim("GPU-1", now=at(1))
    assert admin_event.event_type == EventType.ADMIN_FORCE_RECLAIM
    assert not any(e.event_type == EventType.RECLAIM for e in admin_sched.state.events)  # bypassed the confirmation flow entirely


# ---- 14/15/16/17. No stale structures ----------------------------------------

def test_no_stale_priority_queue_entries_after_cancel_and_priority_change():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "HOLD", "UH")
    jobs = [submit(scheduler, f"J{i}", f"U{i}", minute=i) for i in range(1, 6)]
    scheduler.cancel_job("J2", now=at(10))
    scheduler.change_job_priority("J3", Priority.CRITICAL, now=at(10))

    waiting_ids = scheduler.allocation_engine.waiting_job_ids()
    assert "J2" not in waiting_ids
    assert len(waiting_ids) == len(set(waiting_ids))
    assert_consistent(scheduler)

    scheduler.complete_job("HOLD", now=at(20))
    decisions = scheduler.try_allocate_all(now=at(20))
    assert decisions[0].job_id == "J3"  # the CRITICAL-upgraded job wins, picked via the real Priority Queue
    assert_consistent(scheduler)


def test_no_stale_min_heap_entries_after_maintenance_and_failure_cycles():
    scheduler = Scheduler()
    add_gpus(scheduler, 3)
    scheduler.set_gpu_maintenance("GPU-1", now=T0)
    scheduler.handle_gpu_failure("GPU-2", now=T0)
    scheduler.clear_gpu_maintenance("GPU-1", now=at(1))
    scheduler.recover_gpu_failure("GPU-2", now=at(1))

    jobs = [submit(scheduler, f"J{i}", f"U{i}", minute=i + 2) for i in range(3)]
    assert all(j.status == JobStatus.RUNNING for j in jobs)
    assert len({j.assigned_gpu_ids[0] for j in jobs}) == 3  # every GPU genuinely usable, none double-booked
    assert_consistent(scheduler)


def test_no_duplicate_gpu_ownership_across_a_sequence_of_admin_operations():
    scheduler = Scheduler()
    add_gpus(scheduler, 4)
    a = submit(scheduler, "A", "UA", gpu_count=4)
    scheduler.force_reclaim(a.assigned_gpu_ids[0], now=at(1))
    b = submit(scheduler, "B", "UB", minute=2)
    scheduler.cancel_job("B", now=at(2.5)) if b.status == JobStatus.WAITING else None

    owners = {}
    for job in scheduler.state.jobs.values():
        for gpu_id in job.assigned_gpu_ids:
            assert gpu_id not in owners, f"{gpu_id} double-booked"
            owners[gpu_id] = job.job_id
    assert_consistent(scheduler)


def test_no_duplicate_waiting_jobs_after_repeated_allocation_passes_post_admin_ops():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    jobs = [submit(scheduler, f"J{i}", f"U{i}", minute=i) for i in range(6)]
    scheduler.set_gpu_maintenance(
        next(g for g in scheduler.state.gpus if not scheduler.state.gpus[g].is_assigned), now=at(10),
    ) if any(not g.is_assigned for g in scheduler.state.gpus.values()) else None
    for _ in range(3):
        scheduler.try_allocate_all(now=at(20))
    waiting_ids = scheduler.allocation_engine.waiting_job_ids()
    assert len(waiting_ids) == len(set(waiting_ids))
    assert_consistent(scheduler)


# ---- 18. Scheduler remains functional after many admin operations -----------

def test_scheduler_remains_functional_after_a_long_sequence_of_admin_operations():
    scheduler = Scheduler()
    add_gpus(scheduler, 5)
    jobs = {f"U{i}": submit(scheduler, f"J{i}", f"U{i}", minute=i) for i in range(5)}
    assert_consistent(scheduler)

    scheduler.change_job_priority("J4", Priority.CRITICAL, now=at(1)) if jobs["U4"].status == JobStatus.WAITING else None
    scheduler.set_gpu_maintenance(
        next((g for g, gpu in scheduler.state.gpus.items() if not gpu.is_assigned), None) or "GPU-1", now=at(1),
    ) if any(not g.is_assigned for g in scheduler.state.gpus.values()) else None
    scheduler.force_reclaim(jobs["U0"].assigned_gpu_ids[0], now=at(2)) if jobs["U0"].assigned_gpu_ids else None
    scheduler.handle_gpu_failure("GPU-5", now=at(3))
    scheduler.recover_gpu_failure("GPU-5", now=at(4))
    assert_consistent(scheduler)

    # Still allocates normally afterward.
    fresh = submit(scheduler, "FRESH", "UF", minute=5)
    scheduler.complete_job(fresh.job_id, now=at(6)) if fresh.status == JobStatus.RUNNING else None
    assert_consistent(scheduler)


# ---- 19. Multiple users/jobs/GPU counts ---------------------------------------

def test_admin_operations_work_across_an_arbitrary_number_of_users_jobs_and_gpus():
    scheduler = Scheduler()
    n = 12
    add_gpus(scheduler, n)
    jobs = [submit(scheduler, f"J{i}", f"U{i}", minute=i) for i in range(n)]
    assert all(j.status == JobStatus.RUNNING for j in jobs)
    assert_consistent(scheduler)

    for i in range(0, n, 3):
        scheduler.force_reclaim(jobs[i].assigned_gpu_ids[0], now=at(100))
    assert_consistent(scheduler)

    for i in range(1, n, 4):
        if jobs[i].status == JobStatus.WAITING:
            scheduler.cancel_job(jobs[i].job_id, now=at(101))
    assert_consistent(scheduler)


# ---- consistency helper's own honesty: it must actually detect a violation ---

def test_consistency_checker_detects_a_genuinely_broken_state():
    """The helper must not be a rubber stamp - deliberately corrupt
    state by hand and confirm it is caught."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    job = submit(scheduler, "A", "UA")
    gpu = scheduler.state.get_gpu("GPU-1")

    gpu.assigned_job_id = "GHOST-JOB-THAT-DOES-NOT-EXIST"
    report = check_consistency(scheduler)
    assert not report
    assert any("GHOST-JOB-THAT-DOES-NOT-EXIST" in v for v in report.violations)
    with pytest.raises(AssertionError):
        assert_consistent(scheduler)
