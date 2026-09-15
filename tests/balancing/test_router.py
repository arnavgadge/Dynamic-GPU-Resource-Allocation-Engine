from datetime import datetime, timezone

from engine.balancing.decision import RoutingOutcome
from engine.balancing.router import LoadBalancingRouter
from engine.models.enums import EventType, GPUStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_gpu(gpu_id, utilization=0.0, status=GPUStatus.IDLE, assigned_user=None, assigned_job=None):
    return GPU(
        gpu_id=gpu_id, total_memory_mb=24_576, utilization_percent=utilization, status=status,
        assigned_user_id=assigned_user, assigned_job_id=assigned_job,
    )


def make_job(job_id="J1", size=10.0):
    return Job(job_id=job_id, user_id="U1", name="job", priority=Priority.MEDIUM, estimated_size_minutes=size)


def build_router(gpus):
    state = SchedulerState()
    for gpu in gpus:
        state.add_gpu(gpu)
    return state, LoadBalancingRouter(state)


# ------------------------------------------------------------------
# Test 1 - multiple available GPUs -> least-utilized wins
# ------------------------------------------------------------------

def test_1_multiple_available_gpus_selects_least_utilized():
    state, router = build_router([
        make_gpu("GPU1", 10.0), make_gpu("GPU2", 40.0), make_gpu("GPU3", 20.0),
    ])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.outcome == RoutingOutcome.ROUTED
    assert decision.selected_gpu_id == "GPU1"


# ------------------------------------------------------------------
# Test 2 - least-utilized GPU is assigned -> skip it
# ------------------------------------------------------------------

def test_2_least_utilized_but_assigned_gpu_is_skipped():
    state, router = build_router([
        make_gpu("GPU3", 4.0, status=GPUStatus.ACTIVE, assigned_user="U_A", assigned_job="J_A"),
        make_gpu("GPU4", 50.0),
    ])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.selected_gpu_id == "GPU4"
    # GPU3 was considered but correctly recorded as unavailable.
    candidate = next(c for c in decision.candidates if c.gpu_id == "GPU3")
    assert candidate.available is False


# ------------------------------------------------------------------
# Test 3 - least-utilized GPU is ACTIVE -> skip it
# ------------------------------------------------------------------

def test_3_active_gpu_is_never_selected_regardless_of_utilization():
    state, router = build_router([
        make_gpu("GPU4", 2.0, status=GPUStatus.ACTIVE, assigned_user="U_D", assigned_job="J_D"),
        make_gpu("GPU1", 5.0),
        make_gpu("GPU2", 60.0),
        make_gpu("GPU3", 30.0),
    ])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.selected_gpu_id == "GPU1"


# ------------------------------------------------------------------
# Test 4 - equal utilization -> deterministic tie-break by GPU id
# ------------------------------------------------------------------

def test_4_equal_utilization_breaks_tie_by_gpu_id():
    state, router = build_router([make_gpu("GPU2", 20.0), make_gpu("GPU1", 20.0)])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.selected_gpu_id == "GPU1"  # lexicographically smaller id wins


def test_4b_tie_break_is_stable_regardless_of_insertion_order():
    state, router = build_router([make_gpu("GPU1", 20.0), make_gpu("GPU2", 20.0)])
    decision = router.route_job(make_job(), now=NOW)
    assert decision.selected_gpu_id == "GPU1"


# ------------------------------------------------------------------
# Test 5 - no available GPUs -> job remains WAITING (no fake allocation)
# ------------------------------------------------------------------

def test_5_no_available_gpus_leaves_job_unrouted():
    state, router = build_router([
        make_gpu("GPU1", 90.0, status=GPUStatus.ACTIVE, assigned_user="U_A", assigned_job="J_A"),
    ])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.outcome == RoutingOutcome.NO_GPU_AVAILABLE
    assert decision.selected_gpu_id is None


# ------------------------------------------------------------------
# Test 6 - exactly one available GPU -> select it
# ------------------------------------------------------------------

def test_6_single_available_gpu_is_selected():
    state, router = build_router([make_gpu("GPU1", 33.0)])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.selected_gpu_id == "GPU1"


# ------------------------------------------------------------------
# Test 7 - a reclaimed GPU (now IDLE, unassigned) becomes routable
# ------------------------------------------------------------------

def test_7_gpu_freed_by_reclamation_becomes_routable():
    gpu = make_gpu("GPU2", 90.0, status=GPUStatus.ACTIVE, assigned_user="U_B", assigned_job="J_B")
    state, router = build_router([gpu])

    decision_before = router.route_job(make_job("J_before"), now=NOW)
    assert decision_before.outcome == RoutingOutcome.NO_GPU_AVAILABLE

    # Phase 4 reclaims it (simulated directly - the actual reclaim
    # mechanics are Phase 4's own module/tests).
    gpu.assigned_user_id = None
    gpu.assigned_job_id = None
    gpu.utilization_percent = 0.0
    gpu.status = GPUStatus.IDLE

    decision_after = router.route_job(make_job("J_after"), now=NOW)
    assert decision_after.outcome == RoutingOutcome.ROUTED
    assert decision_after.selected_gpu_id == "GPU2"


# ------------------------------------------------------------------
# Test 8 - GPU in RECLAIMING is not selectable
# ------------------------------------------------------------------

def test_8_reclaiming_gpu_is_not_selectable():
    state, router = build_router([
        make_gpu("GPU1", 1.0, status=GPUStatus.RECLAIMING),
        make_gpu("GPU2", 50.0),
    ])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.selected_gpu_id == "GPU2"


# ------------------------------------------------------------------
# Test 9 - GPU in REALLOCATING is not double-allocated
# ------------------------------------------------------------------

def test_9_reallocating_gpu_is_not_selectable():
    state, router = build_router([
        make_gpu("GPU1", 1.0, status=GPUStatus.REALLOCATING),
        make_gpu("GPU2", 50.0),
    ])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.selected_gpu_id == "GPU2"


# ------------------------------------------------------------------
# Test 10 - invalid utilization reading is handled safely
# ------------------------------------------------------------------

def test_10_invalid_utilization_is_excluded_not_selected():
    state, router = build_router([
        make_gpu("GPU1", float("nan")),
        make_gpu("GPU2", 45.0),
    ])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.selected_gpu_id == "GPU2"


def test_10b_all_available_gpus_invalid_leaves_job_unrouted():
    state, router = build_router([make_gpu("GPU1", float("nan")), make_gpu("GPU2", float("inf"))])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.outcome == RoutingOutcome.NO_GPU_AVAILABLE
    assert decision.selected_gpu_id is None


# ------------------------------------------------------------------
# Test 11 - imbalance threshold (below and above)
# ------------------------------------------------------------------

def test_11_imbalance_below_threshold_is_not_flagged():
    state, router = build_router([make_gpu("GPU1", 20.0), make_gpu("GPU2", 25.0)])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.imbalance_detected is False


def test_11b_imbalance_above_threshold_is_flagged():
    state, router = build_router([make_gpu("GPU1", 5.0), make_gpu("GPU2", 40.0)])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.imbalance_detected is True
    assert "imbalance" in decision.reason.lower()


# ------------------------------------------------------------------
# Test 12 - multiple waiting jobs: router only picks the GPU, never
# the job order (that stays Phase 3's job)
# ------------------------------------------------------------------

def test_12_router_does_not_reorder_or_choose_between_jobs():
    state, router = build_router([make_gpu("GPU1", 10.0)])
    job_a = make_job("JOB_A")
    job_b = make_job("JOB_B")

    decision_a = router.route_job(job_a, now=NOW)
    assert decision_a.job_id == "JOB_A"
    assert decision_a.selected_gpu_id == "GPU1"
    # The router itself never decides which job comes first - that
    # ordering is entirely up to whatever calls it (Phase 3).


# ------------------------------------------------------------------
# Test 13 - running low-utilization GPU is never disturbed
# ------------------------------------------------------------------

def test_13_running_low_utilization_gpu_is_not_touched():
    running_low = make_gpu("GPU3", 4.0, status=GPUStatus.ACTIVE, assigned_user="U_A", assigned_job="J_A")
    state, router = build_router([running_low, make_gpu("GPU4", 50.0)])

    router.route_job(make_job(), now=NOW)

    assert running_low.assigned_user_id == "U_A"
    assert running_low.assigned_job_id == "J_A"
    assert running_low.status == GPUStatus.ACTIVE
    assert running_low.utilization_percent == 4.0


# ------------------------------------------------------------------
# Test 14 - repeated routing never double-allocates the same GPU
# ------------------------------------------------------------------

def test_14_repeated_routing_does_not_reuse_a_gpu_the_caller_already_committed():
    gpu1 = make_gpu("GPU1", 5.0)
    state, router = build_router([gpu1, make_gpu("GPU2", 50.0)])

    first = router.route_job(make_job("J1"), now=NOW)
    assert first.selected_gpu_id == "GPU1"

    # Simulate the caller committing that decision (as
    # AllocationEngine.finalize_assignment would) before routing the
    # next job - GPU1 is no longer available.
    gpu1.assigned_user_id = "U1"
    gpu1.assigned_job_id = "J1"
    gpu1.status = GPUStatus.ACTIVE

    second = router.route_job(make_job("J2"), now=NOW)
    assert second.selected_gpu_id == "GPU2"


def test_14b_routing_without_committing_returns_the_same_gpu_again():
    # The router itself is read-only/idempotent - calling it twice
    # without committing anything returns the same answer. Double
    # allocation is prevented by the caller committing the state
    # change (see test 14), not by the router mutating anything.
    state, router = build_router([make_gpu("GPU1", 5.0)])

    first = router.route_job(make_job("J1"), now=NOW)
    second = router.route_job(make_job("J2"), now=NOW)

    assert first.selected_gpu_id == "GPU1"
    assert second.selected_gpu_id == "GPU1"


# ------------------------------------------------------------------
# Empty pool
# ------------------------------------------------------------------

def test_empty_pool_is_handled_cleanly():
    state, router = build_router([])
    decision = router.route_job(make_job(), now=NOW)

    assert decision.outcome == RoutingOutcome.NO_GPU_AVAILABLE
    assert decision.candidates == []
    assert decision.selected_gpu_id is None


# ------------------------------------------------------------------
# Event generation
# ------------------------------------------------------------------

def test_routing_generates_balance_events():
    state, router = build_router([make_gpu("GPU1", 10.0)])
    decision = router.route_job(make_job(), now=NOW)

    balance_events = [e for e in state.events if e.event_type == EventType.BALANCE]
    assert len(balance_events) == 2  # "evaluated pool" + "selected"
    assert decision.event in state.events
    assert decision.event.event_type == EventType.BALANCE
