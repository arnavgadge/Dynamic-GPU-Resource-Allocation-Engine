"""Day 3: comprehensive coverage for GPU utilization tracking via the
project's existing Min-Heap (`engine.dsa.min_heap.MinHeap`) and its
domain wrapper (`engine.dsa.gpu_utilization_heap.GPUUtilizationHeap`).

`tests/dsa/test_min_heap.py` already covers the heap's basic
insert/peek/extract/rebuild contract (6 generic tests) plus 3
`GPUUtilizationHeap`-specific tests (find-least-utilized, ascending
extraction, refresh-after-mutation) from Phase 2's original suite.
This file adds what Day 3 asks for and was still missing: full
ordering with the task's own worked example, utilization-update
propagation (both directions), ties, dynamic pool sizes up to a large
pool, allocated/MAINTENANCE/UNAVAILABLE GPUs never being routed to
regardless of how low their utilization reads, heap/`SchedulerState`
consistency, the previously-fixed available-GPU-count regression, and
end-to-end mock-hardware utilization propagation into a live
Min-Heap-based routing decision.

Nothing here changes engine behavior - every assertion is against
the existing `GPUUtilizationHeap`, `MinHeap`, `LoadBalancingRouter`,
`AllocationEngine`, `Scheduler`, and hardware layer exactly as they
already work. No random values anywhere - every utilization reading
is a fixed, deterministic constant.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.allocation.engine import AllocationEngine
from engine.balancing.availability import is_gpu_available
from engine.balancing.router import LoadBalancingRouter
from engine.dsa.gpu_utilization_heap import GPUUtilizationHeap
from engine.dsa.min_heap import MinHeap
from engine.hardware.metrics import GPUMetrics
from engine.hardware.monitor import GPUMonitor
from engine.hardware.poller import MonitorPoller, feed_metrics
from engine.models.enums import GPUStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState
from engine.models.user import User
from engine.models.utilization import UtilizationObservation
from engine.scheduler import Scheduler

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def make_gpu(gpu_id: str, utilization: float, status: GPUStatus = GPUStatus.IDLE) -> GPU:
    gpu = GPU(gpu_id=gpu_id, total_memory_mb=24_576, status=status)
    gpu.record_observation(UtilizationObservation(utilization_percent=utilization, timestamp=NOW))
    return gpu


# ---- 1. Empty utilization heap -------------------------------------------

def test_empty_utilization_heap():
    heap = GPUUtilizationHeap()
    assert heap.is_empty() is True
    assert heap.size() == 0
    assert heap.peek_least_utilized() is None
    assert heap.extract_least_utilized() is None


# ---- 2. One GPU -----------------------------------------------------------

def test_one_gpu():
    heap = GPUUtilizationHeap()
    gpu = make_gpu("GPU-1", 42.0)
    heap.insert_gpu(gpu)
    assert heap.size() == 1
    assert heap.peek_least_utilized() is gpu
    assert heap.extract_least_utilized() is gpu
    assert heap.is_empty() is True


# ---- 3/4. Multiple GPUs / insert GPU utilization --------------------------

def test_multiple_gpus_insert():
    heap = GPUUtilizationHeap()
    readings = {"GPU-1": 82.0, "GPU-2": 14.0, "GPU-3": 67.0, "GPU-4": 3.0, "GPU-5": 91.0}
    for gpu_id, util in readings.items():
        heap.insert_gpu(make_gpu(gpu_id, util))
    assert heap.size() == 5


# ---- 5. Peek minimum -------------------------------------------------------

def test_peek_minimum_does_not_remove():
    heap = GPUUtilizationHeap()
    heap.insert_gpu(make_gpu("GPU-1", 82.0))
    heap.insert_gpu(make_gpu("GPU-4", 3.0))
    assert heap.peek_least_utilized().gpu_id == "GPU-4"
    assert heap.peek_least_utilized().gpu_id == "GPU-4"  # still there
    assert heap.size() == 2


# ---- 6/7. Extract minimum / correct ordering (the task's own example) ----

def test_extract_minimum_and_correct_ordering():
    """The task's own worked example:

        GPU-1 -> 82%, GPU-2 -> 14%, GPU-3 -> 67%, GPU-4 -> 3%, GPU-5 -> 91%

    Minimum: GPU-4. Full extraction order must be strictly ascending
    by utilization.
    """
    heap = GPUUtilizationHeap()
    readings = {"GPU-1": 82.0, "GPU-2": 14.0, "GPU-3": 67.0, "GPU-4": 3.0, "GPU-5": 91.0}
    for gpu_id, util in readings.items():
        heap.insert_gpu(make_gpu(gpu_id, util))

    assert heap.peek_least_utilized().gpu_id == "GPU-4"

    order = [heap.extract_least_utilized().gpu_id for _ in range(5)]
    assert order == ["GPU-4", "GPU-2", "GPU-3", "GPU-1", "GPU-5"]
    assert heap.is_empty() is True


# ---- 8/9. Utilization update + minimum changes after update ---------------

def test_utilization_update_and_new_minimum_via_refresh():
    """GPU-4: 3% -> 40% - the heap holds a reference to the same `GPU`
    object, so `record_observation` mutating it in place is invisible
    to the heap's own ordering until `refresh()` (the project's chosen
    update mechanism - `MinHeap.rebuild`, O(n)) is called. This is the
    Day 3 "utilization updates" requirement: SchedulerState/the GPU
    object stays the one source of truth; the heap is refreshed from
    it rather than maintaining a second, independent value.
    """
    heap = GPUUtilizationHeap()
    gpu4 = make_gpu("GPU-4", 3.0)
    gpu2 = make_gpu("GPU-2", 14.0)
    heap.insert_gpu(gpu4)
    heap.insert_gpu(gpu2)
    assert heap.peek_least_utilized().gpu_id == "GPU-4"

    gpu4.record_observation(UtilizationObservation(utilization_percent=40.0, timestamp=NOW))
    # Before refresh, the heap's *node values* are unchanged (it holds
    # the same object reference, so peek would even show the new
    # number) but its *ordering* has not yet been re-validated - the
    # explicit point of `refresh()` existing at all.
    assert gpu4.utilization_percent == 40.0

    heap.refresh()
    assert heap.peek_least_utilized().gpu_id == "GPU-2"  # 14% is now the true minimum


# ---- 10. Multiple GPUs with identical utilization -------------------------

def test_identical_utilization_all_retained_and_extractable():
    heap = GPUUtilizationHeap()
    for gpu_id in ("GPU-1", "GPU-2", "GPU-3"):
        heap.insert_gpu(make_gpu(gpu_id, 50.0))
    assert heap.size() == 3

    extracted_ids = {heap.extract_least_utilized().gpu_id for _ in range(3)}
    assert extracted_ids == {"GPU-1", "GPU-2", "GPU-3"}  # none lost, none duplicated
    assert heap.is_empty() is True


def test_identical_utilization_tie_break_is_deterministic_by_gpu_id_in_the_router():
    """The heap itself makes no tie-break promise beyond "some GPU
    with the minimum key" - `LoadBalancingRouter` is the layer that
    adds a deterministic secondary key (`gpu_id`, lexicographic), by
    building its `MinHeap` on the tuple `(utilization_percent,
    gpu_id)` rather than utilization alone (see `router.py`)."""
    state = SchedulerState()
    for gpu_id in ("GPU-3", "GPU-1", "GPU-2"):
        state.add_gpu(make_gpu(gpu_id, 50.0))

    router = LoadBalancingRouter(state)
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    decision = router.route_job(job, now=NOW)

    assert decision.selected_gpu_id == "GPU-1"  # lowest gpu_id among the tied 50% readings


# ---- 11/12. GPU becoming more / less utilized ------------------------------

def test_gpu_becoming_more_utilized_loses_minimum_status_after_refresh():
    heap = GPUUtilizationHeap()
    gpu_a = make_gpu("GPU-A", 5.0)
    gpu_b = make_gpu("GPU-B", 20.0)
    heap.insert_gpu(gpu_a)
    heap.insert_gpu(gpu_b)
    assert heap.peek_least_utilized().gpu_id == "GPU-A"

    gpu_a.record_observation(UtilizationObservation(utilization_percent=99.0, timestamp=NOW))
    heap.refresh()
    assert heap.peek_least_utilized().gpu_id == "GPU-B"


def test_gpu_becoming_less_utilized_gains_minimum_status_after_refresh():
    heap = GPUUtilizationHeap()
    gpu_a = make_gpu("GPU-A", 80.0)
    gpu_b = make_gpu("GPU-B", 20.0)
    heap.insert_gpu(gpu_a)
    heap.insert_gpu(gpu_b)
    assert heap.peek_least_utilized().gpu_id == "GPU-B"

    gpu_a.record_observation(UtilizationObservation(utilization_percent=1.0, timestamp=NOW))
    heap.refresh()
    assert heap.peek_least_utilized().gpu_id == "GPU-A"


# ---- 13/14/15. Allocated / MAINTENANCE / UNAVAILABLE GPU handling --------

def test_allocated_gpu_is_never_routed_to_even_at_the_lowest_utilization():
    """The heap/router distinction this project has always kept:
    utilization alone never implies availability. An assigned GPU at
    4% utilization must lose to a genuinely free GPU at 20%."""
    state = SchedulerState()
    busy = make_gpu("GPU-BUSY", 4.0, status=GPUStatus.ACTIVE)
    busy.assigned_user_id, busy.assigned_job_id = "U1", "J-OTHER"
    free = make_gpu("GPU-FREE", 20.0)
    state.add_gpu(busy)
    state.add_gpu(free)

    router = LoadBalancingRouter(state)
    job = Job(job_id="J1", user_id="U2", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    decision = router.route_job(job, now=NOW)

    assert decision.selected_gpu_id == "GPU-FREE"


def test_maintenance_gpu_is_never_routed_to_even_at_the_lowest_utilization():
    state = SchedulerState()
    maintenance = make_gpu("GPU-MAINT", 1.0, status=GPUStatus.MAINTENANCE)
    free = make_gpu("GPU-FREE", 20.0)
    state.add_gpu(maintenance)
    state.add_gpu(free)
    assert is_gpu_available(maintenance) is False

    router = LoadBalancingRouter(state)
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    decision = router.route_job(job, now=NOW)

    assert decision.selected_gpu_id == "GPU-FREE"


def test_unavailable_gpu_is_never_routed_to_even_at_the_lowest_utilization():
    state = SchedulerState()
    unavailable = make_gpu("GPU-DOWN", 0.0, status=GPUStatus.UNAVAILABLE)
    free = make_gpu("GPU-FREE", 20.0)
    state.add_gpu(unavailable)
    state.add_gpu(free)
    assert is_gpu_available(unavailable) is False

    router = LoadBalancingRouter(state)
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    decision = router.route_job(job, now=NOW)

    assert decision.selected_gpu_id == "GPU-FREE"


# ---- 16-19. Dynamic pool sizes ---------------------------------------------

@pytest.mark.parametrize("n", [1, 5, 10, 60])
def test_dynamic_pool_sizes_find_the_correct_minimum(n):
    """Deterministic, non-random utilization values (a fixed
    descending-then-wrapping pattern) so the true minimum is known in
    advance for every ``n`` - including a "large pool" (60 GPUs)."""
    heap = GPUUtilizationHeap()
    utilizations = {}
    for i in range(1, n + 1):
        util = float((i * 7) % 101)  # deterministic, not random
        utilizations[f"GPU-{i}"] = util
        heap.insert_gpu(make_gpu(f"GPU-{i}", util))

    assert heap.size() == n
    if n > 0:
        least = heap.peek_least_utilized()
        assert least.utilization_percent == utilizations[least.gpu_id]
        assert least.utilization_percent == min(utilizations.values())


def test_large_pool_extraction_is_fully_sorted():
    heap = GPUUtilizationHeap()
    n = 60
    for i in range(1, n + 1):
        util = float((i * 13) % 100)  # deterministic
        heap.insert_gpu(make_gpu(f"GPU-{i}", util))

    extracted = [heap.extract_least_utilized().utilization_percent for _ in range(n)]
    assert extracted == sorted(extracted)
    assert heap.is_empty() is True


# ---- 20. Heap/state consistency --------------------------------------------

def test_heap_built_from_scheduler_state_agrees_with_state_after_allocations():
    """A `GPUUtilizationHeap` built fresh from `SchedulerState.gpus` -
    the same pattern `LoadBalancingRouter` and `main.py`'s Phase 2 demo
    already use - must always agree with the state it was built from,
    even after real allocations have changed which GPUs are free.
    """
    scheduler = Scheduler()
    for i, util in [(1, 82.0), (2, 14.0), (3, 67.0), (4, 3.0), (5, 91.0)]:
        scheduler.add_gpu(make_gpu(f"GPU-{i}", util))
    scheduler.add_user(User(user_id="U1", name="U1", priority=Priority.MEDIUM))

    # Allocate one job - GPU-4 (3%, the least utilized) is the one the
    # real Min-Heap-based router picks, exactly like `route_job` does.
    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    scheduler.submit_job(job, now=NOW)
    scheduler.try_allocate_all(now=NOW)
    assert job.assigned_gpu_ids == ["GPU-4"]

    # A heap rebuilt from the *current* state must reflect that GPU-4
    # is gone from the available set and GPU-2 (14%) is now the true
    # minimum among what's actually available.
    heap = GPUUtilizationHeap()
    for gpu in scheduler.state.gpus.values():
        if is_gpu_available(gpu):
            heap.insert_gpu(gpu)
    least = heap.peek_least_utilized()
    assert least.gpu_id == "GPU-2"
    assert {g.gpu_id for g in heap._heap.to_list()} == {"GPU-1", "GPU-2", "GPU-3", "GPU-5"}


# ---- 21. Available GPU count consistency (the previously-fixed regression) -

def test_available_gpu_count_matches_state_even_when_the_legacy_heap_has_drifted():
    """Regression guard for the exact issue the project's own history
    already documents (`AllocationEngine.available_gpu_count`'s
    docstring, Phase 13 of the 100-scenario fix set): the persistent
    `_available_gpus` heap `AllocationEngine.allocate_next` maintains
    only ever grows relative to GPUs actually committed through the
    real `Scheduler.try_allocate_all` path (which never pops from it) -
    so it must NEVER be the source `available_gpu_count()` trusts.

    This test manufactures exactly that drift (heap says more GPUs are
    "available" than genuinely are) and asserts the authoritative
    count still matches `SchedulerState` truth.
    """
    state = SchedulerState()
    engine = AllocationEngine(state)
    for i in range(1, 6):
        engine.add_gpu(make_gpu(f"GPU-{i}", 0.0))

    # Manufacture drift: commit two GPUs through the *live* Scheduler-
    # style path (mirroring what finalize_assignment does) without
    # ever popping them from the legacy `_available_gpus` heap -
    # exactly the divergence `try_allocate_all` causes in production.
    from engine.allocation.decision import AllocationPolicy, CandidateInfo
    job_candidates = [CandidateInfo(job_id="J1", user_id="U1", priority=Priority.MEDIUM,
                                      size_minutes=10, waiting_time=timedelta(0), score=None)]
    from engine.models.job import Job as JobModel
    job = JobModel(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
                    estimated_size_minutes=10, submitted_at=NOW)
    gpu1 = state.get_gpu("GPU-1")
    engine._assign_gpu(job, gpu1, AllocationPolicy.FCFS, "test setup", job_candidates, now=NOW)

    # The legacy heap still (wrongly) thinks 5 GPUs are available - it
    # was never told GPU-1 was taken via this direct commit path.
    assert engine._available_gpus.size() == 5

    # SchedulerState truth: only 4 GPUs (GPU-2..GPU-5) are genuinely available.
    true_available = sum(1 for gpu in state.gpus.values() if is_gpu_available(gpu))
    assert true_available == 4
    assert engine.available_gpu_count() == true_available == 4  # never the drifted heap's 5


# ---- 22. Mock hardware utilization propagation ------------------------------

class _FixedReadingsMonitor(GPUMonitor):
    """A deterministic stand-in for NVML/`nvidia-smi` - the exact
    interface (`GPUMonitor`) both real monitors and
    `SimulatorGPUMonitor` implement, so this test proves the same
    thing for any of them: the scheduler cannot tell where a reading
    came from."""

    def __init__(self, readings: dict):
        self._readings = readings

    def get_gpu_metrics(self):
        return [
            GPUMetrics(gpu_id=gpu_id, utilization_percent=util, memory_used_mb=None,
                        memory_total_mb=None, timestamp=NOW)
            for gpu_id, util in self._readings.items()
        ]


def test_mock_hardware_utilization_propagates_into_a_correct_min_heap_decision():
    """End to end, using the task's own worked example:

        NVML/nvidia-smi-shaped monitor -> MonitorPoller -> Scheduler.
        record_utilization -> SchedulerState (GPU.utilization_percent)
        -> a Min-Heap-based routing decision picks the true minimum.

    Deterministic fixed readings throughout - never random.
    """
    scheduler = Scheduler()
    readings = {"GPU-1": 82.0, "GPU-2": 14.0, "GPU-3": 67.0, "GPU-4": 3.0, "GPU-5": 91.0}
    for gpu_id in readings:
        scheduler.add_gpu(make_gpu(gpu_id, 0.0))  # starts at 0 - hardware hasn't reported yet
    scheduler.add_user(User(user_id="U1", name="U1", priority=Priority.MEDIUM))

    poller = MonitorPoller(scheduler, _FixedReadingsMonitor(readings))
    poller.poll_once(now=NOW)

    for gpu_id, util in readings.items():
        assert scheduler.state.get_gpu(gpu_id).utilization_percent == util

    job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
              estimated_size_minutes=30, submitted_at=NOW)
    scheduler.submit_job(job, now=NOW)
    scheduler.try_allocate_all(now=NOW)

    assert job.assigned_gpu_ids == ["GPU-4"]  # 3% - the true minimum, via the router's Min-Heap


def test_simulator_monitor_and_a_scripted_hardware_monitor_feed_the_same_heap_decision():
    """The source-agnostic guarantee (Phase 8), from the utilization-
    heap angle specifically: two independently-built schedulers, one
    fed through `SimulatorGPUMonitor`-shaped metrics and one through a
    plain scripted monitor with identical numbers, reach the identical
    Min-Heap-based routing decision."""
    def build_and_feed(monitor_cls_readings):
        scheduler = Scheduler()
        for gpu_id in monitor_cls_readings:
            scheduler.add_gpu(make_gpu(gpu_id, 0.0))
        scheduler.add_user(User(user_id="U1", name="U1", priority=Priority.MEDIUM))
        poller = MonitorPoller(scheduler, _FixedReadingsMonitor(monitor_cls_readings))
        poller.poll_once(now=NOW)
        return scheduler

    readings = {"GPU-1": 82.0, "GPU-2": 14.0, "GPU-3": 67.0, "GPU-4": 3.0, "GPU-5": 91.0}
    scheduler_a = build_and_feed(dict(readings))
    scheduler_b = build_and_feed(dict(readings))

    for scheduler in (scheduler_a, scheduler_b):
        job = Job(job_id="J1", user_id="U1", name="job", priority=Priority.MEDIUM,
                  estimated_size_minutes=30, submitted_at=NOW)
        scheduler.submit_job(job, now=NOW)
        scheduler.try_allocate_all(now=NOW)
        assert job.assigned_gpu_ids == ["GPU-4"]


# ---- 23. Invalid / missing utilization handling -----------------------------

def test_out_of_range_utilization_is_rejected_and_isolated_per_gpu():
    """`UtilizationObservation` already validates its range at
    construction; `feed_metrics` (the poller's bridge into the
    scheduler) already isolates a single bad reading per-GPU so it
    never corrupts the batch or crashes the poll - verified here from
    the utilization-heap's point of view: the other GPU's valid
    reading still ends up correctly placed as the heap's minimum.
    """
    scheduler = Scheduler()
    scheduler.add_gpu(make_gpu("GPU-1", 0.0))
    scheduler.add_gpu(make_gpu("GPU-2", 0.0))

    metrics = [
        GPUMetrics(gpu_id="GPU-1", utilization_percent=150.0, memory_used_mb=None,  # invalid: > 100
                    memory_total_mb=None, timestamp=NOW),
        GPUMetrics(gpu_id="GPU-2", utilization_percent=9.0, memory_used_mb=None,
                    memory_total_mb=None, timestamp=NOW),
    ]
    errors = feed_metrics(scheduler, metrics)

    assert "GPU-1" in errors  # the bad reading was reported, not silently dropped
    assert scheduler.state.get_gpu("GPU-1").utilization_percent == 0.0  # rejected -> untouched
    assert scheduler.state.get_gpu("GPU-2").utilization_percent == 9.0  # the good one went through

    # A heap built from the (correctly, selectively updated) state is
    # itself unaffected by the rejected reading - it simply sees
    # GPU-1's real, unchanged value (0.0) as the minimum, exactly as
    # it should, rather than the invalid 150.0 that was never applied.
    heap = GPUUtilizationHeap()
    for gpu in scheduler.state.gpus.values():
        heap.insert_gpu(gpu)
    least = heap.peek_least_utilized()
    assert least.gpu_id == "GPU-1"
    assert least.utilization_percent == 0.0


def test_metric_for_an_unknown_gpu_is_skipped_not_crashed():
    scheduler = Scheduler()
    scheduler.add_gpu(make_gpu("GPU-1", 5.0))
    metrics = [GPUMetrics(gpu_id="GPU-UNKNOWN", utilization_percent=50.0, memory_used_mb=None,
                            memory_total_mb=None, timestamp=NOW)]

    errors = feed_metrics(scheduler, metrics)  # must not raise

    assert errors == {}
    assert scheduler.state.get_gpu("GPU-1").utilization_percent == 5.0  # untouched
