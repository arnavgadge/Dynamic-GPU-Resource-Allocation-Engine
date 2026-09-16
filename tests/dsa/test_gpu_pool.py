"""Day 2 (100-scenario-style foundation pass): comprehensive coverage
for the dynamic GPU pool - `engine.dsa.gpu_pool.GPUPool`, the
project's real Linked-List-backed GPU inventory.

`tests/dsa/test_linked_list.py` already covers the pool's basic
add/get/remove path (4 tests) as part of Phase 2's original DSA
suite. This file adds the coverage that was still missing: first/
last/middle removal, allocation/release through the pool's actual
GPU objects, counting available vs. allocated GPUs (via the
project's *existing* `is_gpu_available` predicate - never a second,
competing definition of "available"), MAINTENANCE/UNAVAILABLE
exclusion, no-dangling-reference guarantees after removal, dynamic
pool sizes from 1 to 50+, and consistency between `GPUPool` and
`SchedulerState` through the real `AllocationEngine`/`Scheduler`
paths that keep them in sync.

Nothing here changes engine behavior - every assertion is against
the existing `GPUPool`, `AllocationEngine`, and `Scheduler` exactly
as they already work.
"""

from datetime import datetime, timezone

import pytest

from engine.allocation.engine import AllocationEngine
from engine.balancing.availability import is_gpu_available
from engine.dsa.gpu_pool import GPUPool
from engine.models.enums import GPUStatus
from engine.models.gpu import GPU
from engine.models.scheduler_state import SchedulerState
from engine.scheduler import Scheduler

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def make_gpu(gpu_id: str, status: GPUStatus = GPUStatus.IDLE) -> GPU:
    return GPU(gpu_id=gpu_id, total_memory_mb=24_576, status=status)


def build_pool(gpu_count: int) -> GPUPool:
    """A pool of ``gpu_count`` GPUs, named GPU-1..GPU-N - the exact
    same generic `range(1, n + 1)` construction already used by
    `engine/simulation/scenarios.py` and several existing tests
    (`test_dynamic_reallocation.py`, `test_scheduler_admin_actions.py`).
    Nothing about the pool's own code knows or cares about ``n``.
    """
    pool = GPUPool()
    for i in range(1, gpu_count + 1):
        pool.add_gpu(make_gpu(f"GPU-{i}"))
    return pool


# ---- 1. Empty GPU pool -------------------------------------------------

def test_empty_gpu_pool():
    pool = GPUPool()
    assert pool.is_empty() is True
    assert pool.size() == 0
    assert len(pool) == 0
    assert pool.all_gpus() == []
    assert list(pool) == []
    assert pool.get_gpu("GPU-1") is None
    assert pool.remove_gpu("GPU-1") is False


# ---- 2. Add one GPU -----------------------------------------------------

def test_add_one_gpu():
    pool = GPUPool()
    gpu = make_gpu("GPU-1")
    pool.add_gpu(gpu)
    assert pool.size() == 1
    assert pool.is_empty() is False
    assert pool.get_gpu("GPU-1") is gpu
    assert pool.all_gpus() == [gpu]


# ---- 3. Add multiple GPUs -----------------------------------------------

def test_add_multiple_gpus_preserves_insertion_order():
    pool = build_pool(5)
    assert pool.size() == 5
    assert [g.gpu_id for g in pool.all_gpus()] == [f"GPU-{i}" for i in range(1, 6)]


# ---- 4. Find GPU ---------------------------------------------------------

def test_find_gpu_by_id():
    pool = build_pool(5)
    found = pool.get_gpu("GPU-3")
    assert found is not None
    assert found.gpu_id == "GPU-3"
    assert pool.get_gpu("GPU-999") is None


# ---- 5/6/7/8. Remove GPU - first, last, middle, missing -----------------

def test_remove_gpu_found():
    pool = build_pool(3)
    assert pool.remove_gpu("GPU-2") is True
    assert pool.get_gpu("GPU-2") is None
    assert pool.size() == 2


def test_remove_first_gpu():
    pool = build_pool(5)
    assert pool.remove_gpu("GPU-1") is True
    assert [g.gpu_id for g in pool.all_gpus()] == ["GPU-2", "GPU-3", "GPU-4", "GPU-5"]
    assert pool.size() == 4


def test_remove_last_gpu():
    pool = build_pool(5)
    assert pool.remove_gpu("GPU-5") is True
    assert [g.gpu_id for g in pool.all_gpus()] == ["GPU-1", "GPU-2", "GPU-3", "GPU-4"]
    assert pool.size() == 4


def test_remove_middle_gpu():
    pool = build_pool(5)
    assert pool.remove_gpu("GPU-3") is True
    assert [g.gpu_id for g in pool.all_gpus()] == ["GPU-1", "GPU-2", "GPU-4", "GPU-5"]
    assert pool.size() == 4


def test_remove_unknown_gpu_returns_false_and_changes_nothing():
    pool = build_pool(3)
    assert pool.remove_gpu("GPU-999") is False
    assert pool.size() == 3


# ---- 9/10. Allocate / release a GPU (through the real engine) -----------

def test_allocate_gpu_marks_it_unavailable():
    """"Allocate" here means what the rest of the project means by it:
    `AllocationEngine`/`Scheduler` assigning a job to a GPU, not a
    pool-level operation of its own - the pool has no allocation
    policy, by design (see `docs/dsa.md`)."""
    state = SchedulerState()
    engine = AllocationEngine(state)
    gpu = make_gpu("GPU-1")
    engine.add_gpu(gpu)
    assert is_gpu_available(gpu) is True

    gpu.status = GPUStatus.ACTIVE
    gpu.assigned_user_id = "U1"
    gpu.assigned_job_id = "J1"
    assert is_gpu_available(gpu) is False
    # The pool still knows about the GPU - it is unavailable, not gone.
    assert engine.gpu_pool.get_gpu("GPU-1") is gpu


def test_release_gpu_makes_it_available_again():
    state = SchedulerState()
    engine = AllocationEngine(state)
    gpu = make_gpu("GPU-1", status=GPUStatus.ACTIVE)
    gpu.assigned_user_id, gpu.assigned_job_id = "U1", "J1"
    engine.add_gpu(gpu)  # already assigned -> not inserted into the available heap
    assert is_gpu_available(gpu) is False

    gpu.assigned_user_id = None
    gpu.assigned_job_id = None
    gpu.status = GPUStatus.IDLE
    assert engine.mark_gpu_available("GPU-1") is True
    assert is_gpu_available(gpu) is True


# ---- 11. Count GPUs -------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 5, 10])
def test_count_gpus(n):
    pool = build_pool(n)
    assert pool.size() == n
    assert len(pool) == n


# ---- 12. Count available GPUs --------------------------------------------

def test_count_available_gpus_uses_the_projects_one_availability_definition():
    """"Available" is deliberately not a pool-level concept the pool
    invents itself - it's the project's one existing predicate,
    `is_gpu_available` (`engine/balancing/availability.py`), applied
    to whatever the pool's traversal yields. This is the "appropriate
    existing equivalent" for "determine available GPUs"."""
    pool = build_pool(5)
    busy = pool.get_gpu("GPU-2")
    busy.status, busy.assigned_user_id, busy.assigned_job_id = GPUStatus.ACTIVE, "U1", "J1"

    available = [g for g in pool if is_gpu_available(g)]
    assert {g.gpu_id for g in available} == {"GPU-1", "GPU-3", "GPU-4", "GPU-5"}
    assert len(available) == 4


def test_count_allocated_gpus():
    """The mirror image of the above, via the GPU's own `is_assigned`
    (a plain model-level fact, not a policy) - the "appropriate
    existing equivalent" for "determine allocated GPUs"."""
    pool = build_pool(5)
    for gpu_id in ("GPU-1", "GPU-2"):
        gpu = pool.get_gpu(gpu_id)
        gpu.status, gpu.assigned_user_id, gpu.assigned_job_id = GPUStatus.ACTIVE, "U1", "J1"

    allocated = [g for g in pool if g.is_assigned]
    assert {g.gpu_id for g in allocated} == {"GPU-1", "GPU-2"}
    assert len(allocated) == 2


# ---- 13/14. MAINTENANCE / UNAVAILABLE GPUs cannot be allocated ----------

def test_maintenance_gpu_is_never_available():
    pool = build_pool(3)
    gpu = pool.get_gpu("GPU-2")
    gpu.status = GPUStatus.MAINTENANCE
    assert is_gpu_available(gpu) is False
    assert [g.gpu_id for g in pool if is_gpu_available(g)] == ["GPU-1", "GPU-3"]


def test_unavailable_gpu_is_never_available():
    pool = build_pool(3)
    gpu = pool.get_gpu("GPU-3")
    gpu.status = GPUStatus.UNAVAILABLE
    assert is_gpu_available(gpu) is False
    assert [g.gpu_id for g in pool if is_gpu_available(g)] == ["GPU-1", "GPU-2"]


def test_maintenance_gpu_is_skipped_by_allocate_next():
    """End-to-end: a GPU an admin parked in MAINTENANCE is never
    handed to a waiting job, through the real `Scheduler`/
    `AllocationEngine` path - not just the raw predicate above."""
    scheduler = Scheduler()
    scheduler.add_gpu(make_gpu("GPU-1"))
    good_gpu = make_gpu("GPU-2")
    scheduler.add_gpu(good_gpu)
    scheduler.set_gpu_maintenance("GPU-1", now=NOW)

    from engine.models.enums import Priority
    from engine.models.job import Job
    from engine.models.user import User

    scheduler.add_user(User(user_id="U1", name="U1", priority=Priority.MEDIUM))
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    scheduler.submit_job(job, now=NOW)
    scheduler.try_allocate_all(now=NOW)

    assert job.assigned_gpu_ids == ["GPU-2"]
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.MAINTENANCE
    assert scheduler.state.get_gpu("GPU-1").is_assigned is False


# ---- 15-18. Dynamic pool sizes -------------------------------------------

@pytest.mark.parametrize("n", [1, 5, 10, 50, 100])
def test_dynamic_pool_of_arbitrary_size(n):
    """The engine hardcodes no GPU count anywhere - the exact same
    `GPUPool`/`AllocationEngine` code path supports 1 GPU or 100
    identically. This is the core Day 2 guarantee."""
    pool = build_pool(n)
    assert pool.size() == n
    assert len(pool.all_gpus()) == n
    assert {g.gpu_id for g in pool} == {f"GPU-{i}" for i in range(1, n + 1)}


def test_dynamic_pool_of_five_gpus_matches_the_current_demo_configuration():
    """The current demonstration configuration: 5 logical GPUs,
    GPU-1..GPU-5. Nothing below this pool object needs to know that
    the demo happens to pick 5 - see `test_dynamic_pool_of_arbitrary_size`
    for the same code path at 1/10/50/100."""
    pool = build_pool(5)
    assert [g.gpu_id for g in pool.all_gpus()] == ["GPU-1", "GPU-2", "GPU-3", "GPU-4", "GPU-5"]
    assert all(is_gpu_available(g) for g in pool)


def test_scheduler_scales_to_a_large_dynamic_pool_without_engine_changes():
    """Not just the bare pool - the real `Scheduler` allocates
    correctly across a 50-GPU pool with no special-casing."""
    from engine.models.enums import Priority
    from engine.models.job import Job
    from engine.models.user import User

    scheduler = Scheduler()
    for i in range(1, 51):
        scheduler.add_gpu(make_gpu(f"GPU-{i}"))
    scheduler.add_user(User(user_id="U1", name="U1", priority=Priority.MEDIUM))

    jobs = [
        Job(job_id=f"J{i}", user_id="U1", name=f"job{i}", priority=Priority.MEDIUM,
            estimated_size_minutes=30, submitted_at=NOW)
        for i in range(1, 51)
    ]
    for job in jobs:
        scheduler.submit_job(job, now=NOW)
    scheduler.try_allocate_all(now=NOW)

    assert all(job.assigned_gpu_ids for job in jobs)
    assert len({job.assigned_gpu_ids[0] for job in jobs}) == 50  # every job got a distinct GPU


# ---- 19. No dangling GPU references after removal ------------------------

def test_no_dangling_reference_after_removal():
    pool = build_pool(3)
    pool.remove_gpu("GPU-2")

    assert pool.get_gpu("GPU-2") is None
    assert "GPU-2" not in [g.gpu_id for g in pool.all_gpus()]
    assert "GPU-2" not in [g.gpu_id for g in pool]
    assert pool.size() == 2
    # Removing again is a clean no-op, not a crash on a stale node.
    assert pool.remove_gpu("GPU-2") is False


def test_no_dangling_reference_after_removal_through_allocation_engine():
    """`AllocationEngine.remove_gpu` keeps `GPUPool` and
    `SchedulerState` in lock-step - removing a GPU there must not
    leave it reachable from either."""
    state = SchedulerState()
    engine = AllocationEngine(state)
    for i in range(1, 4):
        engine.add_gpu(make_gpu(f"GPU-{i}"))

    assert engine.remove_gpu("GPU-2") is True
    assert engine.gpu_pool.get_gpu("GPU-2") is None
    assert state.get_gpu("GPU-2") is None
    assert "GPU-2" not in state.gpus
    assert [g.gpu_id for g in engine.gpu_pool] == ["GPU-1", "GPU-3"]


# ---- 20. SchedulerState and GPU pool remain consistent -------------------

def test_scheduler_state_and_gpu_pool_agree_after_add_remove_allocate_release():
    """The end-to-end consistency guarantee: whatever the pool
    (`AllocationEngine.gpu_pool`) says exists is exactly what
    `SchedulerState.gpus` says exists, at every step - add, remove,
    allocate, release - never two independently-drifting truths."""
    from engine.models.enums import Priority
    from engine.models.job import Job
    from engine.models.user import User

    scheduler = Scheduler()
    for i in range(1, 6):
        scheduler.add_gpu(make_gpu(f"GPU-{i}"))
    scheduler.add_user(User(user_id="U1", name="U1", priority=Priority.MEDIUM))

    def pool_ids():
        return {g.gpu_id for g in scheduler.allocation_engine.gpu_pool}

    def state_ids():
        return set(scheduler.state.gpus.keys())

    assert pool_ids() == state_ids() == {f"GPU-{i}" for i in range(1, 6)}

    # Allocate.
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    scheduler.submit_job(job, now=NOW)
    scheduler.try_allocate_all(now=NOW)
    assert pool_ids() == state_ids()  # membership unaffected by an assignment

    gpu_id = job.assigned_gpu_ids[0]
    assert scheduler.allocation_engine.gpu_pool.get_gpu(gpu_id) is scheduler.state.get_gpu(gpu_id)

    # Release (job completes normally).
    scheduler.complete_job("J1", now=NOW)
    assert pool_ids() == state_ids()
    assert is_gpu_available(scheduler.state.get_gpu(gpu_id)) is True

    # Remove a GPU from the pool entirely.
    scheduler.allocation_engine.remove_gpu("GPU-5")
    assert pool_ids() == state_ids() == {f"GPU-{i}" for i in range(1, 5)}
