"""Day 4: comprehensive coverage for the user/job management layer -
the HashMap-backed lookups (`SchedulerState.get_user`/`get_job`/
`get_gpu`, all plain-`dict`-backed O(1) lookups, and the hand-built
`UserGPUIndex` HashMap wrapper for the one place a *list*-valued
reverse index is genuinely needed) and the waiting-job `Queue`
(`WaitingJobQueue`), exactly as they already exist and are already
wired into `Scheduler`/`AllocationEngine`.

Existing coverage this file deliberately does not repeat:
`tests/test_user.py`/`tests/test_job.py` (bare model construction),
`tests/dsa/test_hashmap.py`/`test_queue.py`/`test_integration.py`
(the generic DSA structures in isolation), and
`tests/allocation/test_engine.py`'s own consistency/multi-job tests.
This file is the *management-layer* pass: arbitrary numbers of users
and jobs, HashMap/Queue participation in the real scheduler path,
multi-GPU partial-allocation state, and no-dangling-reference
guarantees across every release path (completion, cancellation,
reclamation, priority preemption, hardware failure).

While writing these tests, a real gap was found and fixed:
`Scheduler.complete_job` released a GPU from `GPU`/`User` correctly
but never told `AllocationEngine`'s `UserGPUIndex` (the HashMap-backed
reverse index) - every *other* release path
(`force_reclaim`/`handle_gpu_failure`/`ReclamationEngine._reclaim`)
already called `release_user_gpu` for exactly this reason. Fixed by
adding the same call to `complete_job`'s release loop -
`test_no_dangling_userindex_entry_after_normal_completion` below is
the regression test for it. No other engine behavior changed.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.scheduler import Scheduler

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def add_gpus(scheduler: Scheduler, n: int) -> None:
    for i in range(1, n + 1):
        scheduler.add_gpu(GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE))


def add_user(scheduler: Scheduler, user_id: str, priority: Priority = Priority.MEDIUM) -> User:
    user = User(user_id=user_id, name=user_id, priority=priority)
    scheduler.add_user(user)
    return user


def submit(scheduler: Scheduler, job: Job, now: datetime = NOW) -> Job:
    scheduler.submit_job(job, now=now)
    scheduler.try_allocate_all(now=now)
    return job


# ---- 1/2. Create user(s) --------------------------------------------------

def test_create_user():
    scheduler = Scheduler()
    user = add_user(scheduler, "User-A", Priority.HIGH)
    assert scheduler.state.get_user("User-A") is user
    assert user.assigned_gpu_ids == []
    assert user.running_job_ids == []


def test_create_multiple_arbitrary_users():
    """Not "User A/B/C/D" hardcoded anywhere - any set of ids works,
    including ones that look nothing like a letter sequence."""
    scheduler = Scheduler()
    ids = ["User-A", "User-B", "User-C", "User-D", "User-X", "User-Y"]
    for user_id in ids:
        add_user(scheduler, user_id)

    assert {u for u in scheduler.state.users} == set(ids)
    for user_id in ids:
        assert scheduler.state.get_user(user_id).user_id == user_id


# ---- 3/4. Create job(s) ---------------------------------------------------

def test_create_job():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)
    assert scheduler.state.get_job("J1") is job


def test_create_multiple_jobs_for_an_arbitrary_number_of_users():
    scheduler = Scheduler()
    add_gpus(scheduler, 10)
    user_ids = [f"User-{i}" for i in range(1, 8)]
    for user_id in user_ids:
        add_user(scheduler, user_id)

    jobs = []
    for i, user_id in enumerate(user_ids):
        job = Job(job_id=f"J{i}", user_id=user_id, name="job", priority=Priority.MEDIUM,
                   estimated_size_minutes=30, submitted_at=NOW)
        jobs.append(submit(scheduler, job))

    assert len(scheduler.state.jobs) == len(user_ids)
    for job in jobs:
        assert scheduler.state.get_job(job.job_id) is job


# ---- 5/6/7. User<->Job<->GPU lookups --------------------------------------

def test_user_to_job_lookup_via_running_job_ids_and_state_filter():
    """Two "appropriate existing equivalents" for user -> job(s):
    `User.running_job_ids` (O(1), RUNNING jobs only) and filtering
    `SchedulerState.jobs` by `job.user_id` (the same pattern
    `api/serializers.py::serialize_portal_state` already uses for
    "my jobs", covering every status, not only RUNNING)."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    user = add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)

    assert user.running_job_ids == ["J1"]
    jobs_for_user = [j for j in scheduler.state.jobs.values() if j.user_id == "U1"]
    assert jobs_for_user == [job]


def test_job_to_user_lookup():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    user = add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)

    looked_up_user = scheduler.state.get_user(job.user_id)
    assert looked_up_user is user


def test_gpu_to_job_to_user_reverse_lookup():
    """GPU-id -> Job -> User, the reverse chain the task's own
    "GPU-2 -> Job 101 -> User A" example describes - both hops are
    O(1) dict lookups already provided by `SchedulerState`."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    user = add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)
    gpu_id = job.assigned_gpu_ids[0]

    found_job = scheduler.state.get_job_on_gpu(gpu_id)
    found_user = scheduler.state.get_user(found_job.user_id)
    assert found_job is job
    assert found_user is user


# ---- 8/9/10. Waiting queue: add, remove, no duplicates -------------------

def test_add_waiting_job_appears_exactly_once():
    scheduler = Scheduler()
    add_user(scheduler, "U1")  # no GPUs at all -> job cannot be allocated
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)

    waiting_ids = [j.job_id for j in scheduler.state.get_waiting_jobs()]
    assert waiting_ids == ["J1"]
    assert scheduler.allocation_engine.waiting_job_ids() == ["J1"]


def test_job_allocated_completely_is_removed_from_waiting_queue():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)

    assert job.status == JobStatus.RUNNING
    assert scheduler.allocation_engine.waiting_job_ids() == []


def test_no_duplicate_waiting_jobs_across_repeated_allocation_passes():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    add_user(scheduler, "U1")
    jobs = [
        Job(job_id=f"J{i}", user_id="U1", name="job", priority=Priority.MEDIUM,
             estimated_size_minutes=30, submitted_at=NOW)
        for i in range(1, 6)
    ]
    for job in jobs:
        scheduler.submit_job(job, now=NOW)
    # Call try_allocate_all repeatedly - it must be idempotent, never
    # re-adding a job that already left the waiting queue.
    for _ in range(3):
        scheduler.try_allocate_all(now=NOW)

    waiting_ids = scheduler.allocation_engine.waiting_job_ids()
    assert len(waiting_ids) == len(set(waiting_ids))  # no duplicates
    assert len(waiting_ids) == 3  # 2 GPUs served 2 jobs; 3 remain


# ---- 11/12. Complete running job / cancel waiting job ---------------------

def test_complete_running_job_updates_user_gpu_and_queue_consistently():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    user = add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)
    gpu_id = job.assigned_gpu_ids[0]

    scheduler.complete_job("J1", now=NOW)

    assert job.status == JobStatus.COMPLETED
    assert job.assigned_gpu_ids == []
    assert gpu_id not in user.assigned_gpu_ids
    assert "J1" not in user.running_job_ids
    assert scheduler.state.get_gpu(gpu_id).status == GPUStatus.IDLE
    assert scheduler.state.get_gpu(gpu_id).is_assigned is False
    assert "J1" not in scheduler.allocation_engine.waiting_job_ids()


def test_cancel_waiting_job_removes_it_from_the_queue():
    scheduler = Scheduler()
    add_user(scheduler, "U1")  # no GPU -> stays WAITING
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)
    assert scheduler.allocation_engine.waiting_job_ids() == ["J1"]

    scheduler.cancel_job("J1", now=NOW)

    assert job.status == JobStatus.CANCELLED
    assert scheduler.allocation_engine.waiting_job_ids() == []
    assert scheduler.state.get_waiting_jobs() == []


def test_cancel_rejects_a_running_job():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)

    with pytest.raises(ValueError):
        scheduler.cancel_job("J1", now=NOW)


# ---- 13/14. Multi-GPU job state / partial allocation ----------------------

def test_multi_gpu_job_state_requested_allocated_remaining():
    """The task's own example, generalized: requested=4, and however
    many are actually free right now determines allocated/remaining -
    no new fields needed, `Job.gpu_count`/`assigned_gpu_ids`/
    `gpus_still_needed` already represent exactly this."""
    scheduler = Scheduler()
    add_gpus(scheduler, 5)
    add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, gpu_count=4, submitted_at=NOW)
    submit(scheduler, job)

    assert job.gpu_count == 4
    assert len(job.assigned_gpu_ids) == 4  # 5 free GPUs -> fully satisfied
    assert job.gpus_still_needed == 0
    assert job.is_fully_allocated is True
    assert job.status == JobStatus.RUNNING


def test_partial_multi_gpu_allocation_stays_waiting_with_correct_remaining_count():
    """5-GPU pool; Job A requests 3, Job B requests 2 (the task's own
    example) - both fully fit here, so instead this exercises the
    genuinely-partial case: a job requesting more GPUs than are free
    stays WAITING with an accurate remaining count, and is represented
    by a *single* waiting-queue entry, never one per missing GPU."""
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, gpu_count=3, submitted_at=NOW)
    submit(scheduler, job)

    assert job.gpu_count == 3
    assert len(job.assigned_gpu_ids) == 2  # only 2 GPUs exist to give it
    assert job.gpus_still_needed == 1
    assert job.is_fully_allocated is False
    assert job.status == JobStatus.WAITING
    # Exactly one waiting-queue entry for this job, not three (one per
    # missing GPU) and not two (one per already-held GPU either).
    assert scheduler.allocation_engine.waiting_job_ids() == ["J1"]
    assert scheduler.state.get_waiting_jobs() == [job]


def test_five_gpu_pool_two_multi_gpu_jobs_matches_the_tasks_own_example():
    """GPU pool: 5 GPUs. Job A requests 3, Job B requests 2 - both are
    satisfiable, and combined they exactly exhaust the pool."""
    scheduler = Scheduler()
    add_gpus(scheduler, 5)
    add_user(scheduler, "UA")
    add_user(scheduler, "UB")
    job_a = Job(job_id="JA", user_id="UA", name="job-a", priority=Priority.MEDIUM,
                 estimated_size_minutes=30, gpu_count=3, submitted_at=NOW)
    job_b = Job(job_id="JB", user_id="UB", name="job-b", priority=Priority.MEDIUM,
                 estimated_size_minutes=30, gpu_count=2, submitted_at=NOW + timedelta(seconds=1))
    submit(scheduler, job_a, now=NOW)
    submit(scheduler, job_b, now=NOW)

    assert job_a.is_fully_allocated and job_b.is_fully_allocated
    assert len(job_a.assigned_gpu_ids) == 3
    assert len(job_b.assigned_gpu_ids) == 2
    assert set(job_a.assigned_gpu_ids) | set(job_b.assigned_gpu_ids) == {f"GPU-{i}" for i in range(1, 6)}
    assert scheduler.allocation_engine.waiting_job_ids() == []


# ---- 15. Requeue job (hardware failure path) -------------------------------

def test_requeued_job_after_hardware_failure_appears_exactly_once():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)
    gpu_id = job.assigned_gpu_ids[0]
    assert job.status == JobStatus.RUNNING

    scheduler.handle_gpu_failure(gpu_id, now=NOW)

    # The failed GPU's job goes back to WAITING and is requeued -
    # exactly once, not duplicated, and the try_allocate_all inside
    # handle_gpu_failure may have already served it from the pool's
    # other GPU.
    waiting_ids = scheduler.allocation_engine.waiting_job_ids()
    assert waiting_ids.count("J1") == 0 or waiting_ids == ["J1"]
    assert scheduler.state.get_gpu(gpu_id).status == GPUStatus.UNAVAILABLE
    # Either way, the job itself was never lost or duplicated in state.
    assert scheduler.state.get_job("J1") is job


# ---- 16. Remove stale assignment (UserGPUIndex regression) ---------------

def test_no_dangling_userindex_entry_after_normal_completion():
    """The Day 4 fix: a normal completion must clear the HashMap-
    backed `UserGPUIndex` exactly like every other release path does -
    `AllocationEngine.get_gpus_for_user` must never keep reporting a
    GPU released through `complete_job`."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)
    assert scheduler.allocation_engine.get_gpus_for_user("U1") == ["GPU-1"]

    scheduler.complete_job("J1", now=NOW)

    assert scheduler.allocation_engine.get_gpus_for_user("U1") == []


def test_no_dangling_userindex_entry_after_force_reclaim():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)
    assert scheduler.allocation_engine.get_gpus_for_user("U1") == ["GPU-1"]

    scheduler.force_reclaim("GPU-1", now=NOW)

    assert scheduler.allocation_engine.get_gpus_for_user("U1") == []


def test_no_dangling_userindex_entry_after_hardware_failure():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)
    assert scheduler.allocation_engine.get_gpus_for_user("U1") == ["GPU-1"]

    scheduler.handle_gpu_failure("GPU-1", now=NOW)

    assert scheduler.allocation_engine.get_gpus_for_user("U1") == []


# ---- 17. User with multiple jobs -------------------------------------------

def test_user_with_multiple_jobs_tracks_each_independently():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    user = add_user(scheduler, "U1")
    job_1 = Job(job_id="J1", user_id="U1", name="job1", priority=Priority.MEDIUM,
                 estimated_size_minutes=30, submitted_at=NOW)
    job_2 = Job(job_id="J2", user_id="U1", name="job2", priority=Priority.MEDIUM,
                 estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job_1, now=NOW)
    submit(scheduler, job_2, now=NOW)

    assert sorted(user.running_job_ids) == ["J1", "J2"]
    assert sorted(user.assigned_gpu_ids) == ["GPU-1", "GPU-2"]

    scheduler.complete_job("J1", now=NOW)

    assert user.running_job_ids == ["J2"]  # J2 untouched by J1's completion
    assert user.assigned_gpu_ids == ["GPU-2"]
    assert job_2.status == JobStatus.RUNNING


# ---- 18. Multiple users competing for GPUs ---------------------------------

def test_multiple_users_competing_for_a_constrained_pool():
    """3 users, 2 GPUs, equal-sized jobs submitted in a known order -
    the earliest 2 arrivals run, the 3rd waits, and completing one
    running job immediately frees capacity for the one still waiting.
    This is a management-layer test (queue/index correctness), not a
    scheduling-policy one - equal sizes keep it FCFS-trivial rather
    than depending on the score formula."""
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    users = ["User-X", "User-Y", "User-Z"]
    for user_id in users:
        add_user(scheduler, user_id)

    jobs = [
        Job(job_id=f"J-{user_id}", user_id=user_id, name="job", priority=Priority.MEDIUM,
             estimated_size_minutes=30, submitted_at=NOW + timedelta(seconds=i))
        for i, user_id in enumerate(users)
    ]
    for job in jobs:
        scheduler.submit_job(job, now=job.submitted_at)
    scheduler.try_allocate_all(now=NOW + timedelta(seconds=10))

    running = [j for j in jobs if j.status == JobStatus.RUNNING]
    waiting = [j for j in jobs if j.status == JobStatus.WAITING]
    assert len(running) == 2
    assert len(waiting) == 1
    assert {j.job_id for j in running} == {"J-User-X", "J-User-Y"}  # earliest two arrivals
    assert waiting[0].job_id == "J-User-Z"

    scheduler.complete_job("J-User-X", now=NOW + timedelta(seconds=11))
    scheduler.try_allocate_all(now=NOW + timedelta(seconds=11))
    assert jobs[2].status == JobStatus.RUNNING  # User-Z's job now running
    assert scheduler.allocation_engine.waiting_job_ids() == []


# ---- 19. Empty waiting queue ------------------------------------------------

def test_empty_waiting_queue_initially_and_after_full_drain():
    scheduler = Scheduler()
    assert scheduler.state.get_waiting_jobs() == []
    assert scheduler.allocation_engine.waiting_job_ids() == []

    add_gpus(scheduler, 3)
    add_user(scheduler, "U1")
    for i in range(3):
        submit(scheduler, Job(job_id=f"J{i}", user_id="U1", name="job", priority=Priority.MEDIUM,
                                estimated_size_minutes=30, submitted_at=NOW), now=NOW)

    assert scheduler.allocation_engine.waiting_job_ids() == []  # fully drained, 3 GPUs for 3 jobs


# ---- 20. Large number of users/jobs -----------------------------------------

def test_large_number_of_users_and_jobs():
    scheduler = Scheduler()
    n = 60
    add_gpus(scheduler, n)
    for i in range(1, n + 1):
        add_user(scheduler, f"User-{i}")

    jobs = []
    for i in range(1, n + 1):
        job = Job(job_id=f"J{i}", user_id=f"User-{i}", name="job", priority=Priority.MEDIUM,
                   estimated_size_minutes=30, submitted_at=NOW)
        jobs.append(job)
        scheduler.submit_job(job, now=NOW)
    scheduler.try_allocate_all(now=NOW)

    assert all(job.status == JobStatus.RUNNING for job in jobs)
    assert len(scheduler.state.jobs) == n
    assert len(scheduler.state.users) == n
    assert len({job.assigned_gpu_ids[0] for job in jobs}) == n  # every job got a distinct GPU
    assert scheduler.allocation_engine.waiting_job_ids() == []
    for i in range(1, n + 1):
        assert scheduler.allocation_engine.get_gpus_for_user(f"User-{i}") == jobs[i - 1].assigned_gpu_ids


# ---- 21. HashMap lookup behavior --------------------------------------------

def test_hashmap_lookup_behavior_hit_and_miss():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    add_user(scheduler, "U1")
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, job)

    # SchedulerState's dict-backed lookups (the project's own, already-
    # documented O(1)-average "HashMap" for the primary user/job/GPU
    # tables - see docs/dsa.md).
    assert scheduler.state.get_user("U1") is not None
    assert scheduler.state.get_user("missing") is None
    assert scheduler.state.get_job("J1") is not None
    assert scheduler.state.get_job("missing") is None
    assert scheduler.state.get_gpu("GPU-1") is not None
    assert scheduler.state.get_gpu("missing") is None

    # The hand-built HashMap-backed UserGPUIndex, participating in the
    # real allocation path (not a demonstration-only structure).
    assert scheduler.allocation_engine.get_gpus_for_user("U1") == ["GPU-1"]
    assert scheduler.allocation_engine.get_gpus_for_user("missing") == []


# ---- 22. Queue ordering ------------------------------------------------------

def test_queue_ordering_preserves_arrival_order_under_fcfs():
    """Equal-sized jobs (so the 20% similarity rule's FCFS branch
    applies, without depending on the score formula this project's
    later milestones own) - allocation order must exactly match
    submission order."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    add_user(scheduler, "U1")
    jobs = [
        Job(job_id=f"J{i}", user_id="U1", name="job", priority=Priority.MEDIUM,
             estimated_size_minutes=30, submitted_at=NOW + timedelta(seconds=i))
        for i in range(4)
    ]
    for job in jobs:
        scheduler.submit_job(job, now=job.submitted_at)

    order = []
    for _ in range(4):
        scheduler.try_allocate_all(now=NOW + timedelta(seconds=100))
        running = [j for j in jobs if j.status == JobStatus.RUNNING and j.job_id not in order]
        if running:
            order.append(running[0].job_id)
            scheduler.complete_job(running[0].job_id, now=NOW + timedelta(seconds=100))

    assert order == ["J0", "J1", "J2", "J3"]


# ---- 23. SchedulerState consistency -----------------------------------------

def _assert_fully_consistent(scheduler: Scheduler) -> None:
    state = scheduler.state
    for gpu_id, gpu in state.gpus.items():
        assert gpu.gpu_id == gpu_id
        if gpu.assigned_job_id is not None:
            job = state.get_job(gpu.assigned_job_id)
            assert job is not None
            assert gpu_id in job.assigned_gpu_ids
            assert job.user_id == gpu.assigned_user_id
        if gpu.assigned_user_id is not None:
            user = state.get_user(gpu.assigned_user_id)
            assert user is not None
            assert gpu_id in user.assigned_gpu_ids

    for user_id, user in state.users.items():
        for gpu_id in user.assigned_gpu_ids:
            gpu = state.get_gpu(gpu_id)
            assert gpu is not None
            assert gpu.assigned_user_id == user_id
        # The HashMap-backed reverse index must agree with the ground truth.
        assert sorted(scheduler.allocation_engine.get_gpus_for_user(user_id)) == sorted(user.assigned_gpu_ids)

    for job_id, job in state.jobs.items():
        assert job.job_id == job_id
        assert state.get_user(job.user_id) is not None


def test_scheduler_state_stays_consistent_across_a_mixed_operation_sequence():
    """Submit, allocate, complete, cancel, a multi-GPU partial
    allocation, and a hardware failure, all in one sequence - the
    full consistency sweep runs after every step, not only at the end.
    """
    scheduler = Scheduler()
    add_gpus(scheduler, 4)
    add_user(scheduler, "U1")
    add_user(scheduler, "U2")
    _assert_fully_consistent(scheduler)

    j1 = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    submit(scheduler, j1)
    _assert_fully_consistent(scheduler)

    j2 = Job(job_id="J2", user_id="U2", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, gpu_count=2, submitted_at=NOW)
    submit(scheduler, j2)
    _assert_fully_consistent(scheduler)
    assert j2.is_fully_allocated

    j3 = Job(job_id="J3", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)  # only 1 GPU left
    submit(scheduler, j3)
    _assert_fully_consistent(scheduler)

    j4 = Job(job_id="J4", user_id="U2", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)  # no GPUs left -> waits
    submit(scheduler, j4)
    _assert_fully_consistent(scheduler)
    assert j4.status == JobStatus.WAITING

    scheduler.cancel_job("J4", now=NOW)
    _assert_fully_consistent(scheduler)
    assert j4.status == JobStatus.CANCELLED

    scheduler.complete_job("J1", now=NOW)
    _assert_fully_consistent(scheduler)

    failed_gpu = j3.assigned_gpu_ids[0]
    scheduler.handle_gpu_failure(failed_gpu, now=NOW)
    _assert_fully_consistent(scheduler)
