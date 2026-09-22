"""Day 8: utilization-aware GPU load balancing and hysteresis/cooldown,
verified end to end through the real `Scheduler`.

Two things are deliberately kept distinct, matching the existing
architecture (nothing here changes either):

- **Routing new work** (`LoadBalancingRouter.route_job`) - a job that
  needs a GPU is given the least-utilized *available* one, via the
  Min-Heap. "Available" (`engine.balancing.availability.is_gpu_available`)
  means genuinely free (`IDLE`, unassigned) - an actively-used GPU is
  never a candidate no matter how low its utilization reads, so this
  path can never "steal" a running job's GPU and needs no cooldown of
  its own (it never repeats an action against the same GPU - once
  routed, a GPU is assigned and no longer a candidate for anything).

- **Reallocation asks** (`Scheduler._request_additional_gpus_if_needed`,
  via `ReclamationEngine.request_gpu_for_reallocation`) - asking
  *another user* to release an underutilized (excess-capacity) or
  lower-priority (preemption) GPU. This is the path hysteresis
  protects: `BalancingPolicy.preemption_cooldown` blocks a repeated ask
  on the same GPU for a configurable duration after its last ask was
  resolved, so utilization oscillating near a threshold cannot cause
  a balancing request every tick. Cooldown deliberately never affects
  ordinary routing (see `test_cooldown_never_blocks_ordinary_routing_
  of_a_genuinely_free_gpu`).

All time is simulated and explicit; all utilization values are fixed
constants.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.balancing.config import BalancingPolicy
from engine.balancing.decision import ReallocationPath, RoutingOutcome
from engine.balancing.router import LoadBalancingRouter
from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState
from engine.models.user import User
from engine.models.utilization import UtilizationObservation
from engine.reclamation.policy import ConfirmationResponse
from engine.scheduler import Scheduler

T0 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
YES, NO = ConfirmationResponse.YES, ConfirmationResponse.NO


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def sched(cooldown_minutes: float = 5, imbalance: float = 30.0) -> Scheduler:
    return Scheduler(balancing_policy=BalancingPolicy(
        imbalance_threshold_percent=imbalance, preemption_cooldown=timedelta(minutes=cooldown_minutes),
    ))


def add_gpus(scheduler: Scheduler, n: int) -> None:
    for i in range(1, n + 1):
        scheduler.add_gpu(GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE))


def add_user(scheduler: Scheduler, user_id: str, priority: Priority = Priority.MEDIUM) -> User:
    user = User(user_id=user_id, name=user_id, priority=priority)
    scheduler.add_user(user)
    return user


def hold(scheduler: Scheduler, job_id: str, user_id: str, gpu_count: int = 1, minute: float = 0,
         priority: Priority = Priority.MEDIUM) -> Job:
    if scheduler.state.get_user(user_id) is None:
        add_user(scheduler, user_id, priority)
    job = Job(job_id=job_id, user_id=user_id, name=job_id, priority=priority,
              estimated_size_minutes=100_000, gpu_count=gpu_count, submitted_at=at(minute))
    scheduler.submit_job(job, now=at(minute))
    scheduler.try_allocate_all(now=at(minute))
    return job


def set_utilization(scheduler: Scheduler, readings: dict, minute: float = 0) -> None:
    for gpu_id, value in readings.items():
        scheduler.record_utilization(gpu_id, value, at(minute))


def pin(scheduler: Scheduler, gpu_id: str, user_id: str, utilization: float) -> Job:
    """Directly place a holder on a *specific* GPU id, bypassing the
    router's own placement choice - for tests that need to control
    exactly which GPU ends up busy, independent of routing order."""
    user = scheduler.state.get_user(user_id)
    if user is None:
        user = add_user(scheduler, user_id)
    job = Job(job_id=f"HOLD-{gpu_id}", user_id=user_id, name=f"HOLD-{gpu_id}", priority=Priority.MEDIUM,
              estimated_size_minutes=100_000, submitted_at=T0, status=JobStatus.RUNNING,
              started_at=T0, assigned_gpu_ids=[gpu_id])
    scheduler.state.add_job(job)
    gpu = scheduler.state.get_gpu(gpu_id)
    gpu.status = GPUStatus.ACTIVE
    gpu.assigned_user_id = user_id
    gpu.assigned_job_id = job.job_id
    gpu.record_observation(UtilizationObservation(utilization_percent=utilization, timestamp=T0))
    user.assigned_gpu_ids.append(gpu_id)
    user.running_job_ids.append(job.job_id)
    return job


# ---- 1/2/3/4/6. Routing: free GPUs, uneven load, protection of busy ones ---

def test_single_free_gpu_is_used():
    scheduler = sched()
    add_gpus(scheduler, 1)
    job = hold(scheduler, "J1", "U1")
    assert job.status == JobStatus.RUNNING and job.assigned_gpu_ids == ["GPU-1"]


def test_multiple_free_gpus_picks_the_least_utilized():
    scheduler = sched()
    add_gpus(scheduler, 3)
    set_utilization(scheduler, {"GPU-1": 50.0, "GPU-2": 5.0, "GPU-3": 20.0})
    job = hold(scheduler, "J1", "U1")
    assert job.assigned_gpu_ids == ["GPU-2"]


def test_the_tasks_own_uneven_utilization_example():
    """GPU-1..5 -> 90/80/4/65/70; a new job goes to GPU-3."""
    scheduler = sched()
    add_gpus(scheduler, 5)
    # Pin 1,2,4,5 busy at their stated utilization; GPU-3 stays free and low.
    for i, util in ((1, 90.0), (2, 80.0), (4, 65.0), (5, 70.0)):
        pin(scheduler, f"GPU-{i}", f"H{i}", util)
    set_utilization(scheduler, {"GPU-3": 4.0})

    job = hold(scheduler, "NEW", "N")
    assert job.assigned_gpu_ids == ["GPU-3"]


def test_high_utilization_gpus_are_never_interrupted_for_a_new_job():
    scheduler = sched()
    add_gpus(scheduler, 1)
    holder = hold(scheduler, "HOLD", "A")
    set_utilization(scheduler, {"GPU-1": 90.0})
    waiter = hold(scheduler, "NEW", "B", minute=1)

    assert holder.status == JobStatus.RUNNING and holder.assigned_gpu_ids == ["GPU-1"]
    assert waiter.status == JobStatus.WAITING and waiter.assigned_gpu_ids == []
    assert scheduler.state.get_gpu("GPU-1").assigned_user_id == "A"  # untouched


def test_new_job_prefers_the_available_gpu_over_heavily_loaded_ones_arbitrary_count():
    """Same idea, generalized to an arbitrary pool size/position - not
    hardcoded to "GPU-3"."""
    for n in (4, 9):
        for free_index in (1, n // 2, n):
            scheduler = sched()
            add_gpus(scheduler, n)
            for i in range(1, n + 1):
                if i != free_index:
                    pin(scheduler, f"GPU-{i}", f"H{i}", 75.0)
            job = hold(scheduler, "NEW", "N")
            assert job.assigned_gpu_ids == [f"GPU-{free_index}"]


# ---- 8/9/10. Maintenance / unavailable / assigned GPU never treated as free -

def test_maintenance_gpu_excluded_from_routing():
    scheduler = sched()
    add_gpus(scheduler, 2)
    scheduler.set_gpu_maintenance("GPU-1", now=T0)
    job = hold(scheduler, "J1", "U1")
    assert job.assigned_gpu_ids == ["GPU-2"]


def test_unavailable_gpu_excluded_from_routing():
    scheduler = sched()
    add_gpus(scheduler, 2)
    scheduler.handle_gpu_failure("GPU-1", now=T0)
    job = hold(scheduler, "J1", "U1")
    assert job.assigned_gpu_ids == ["GPU-2"]


def test_assigned_gpu_at_low_utilization_is_not_treated_as_globally_free():
    """User A owns GPU-1 at 3% - that must not make GPU-1 "available"
    for a new job; the new job must wait rather than share it."""
    scheduler = sched()
    add_gpus(scheduler, 1)
    holder = hold(scheduler, "HOLD", "A")
    set_utilization(scheduler, {"GPU-1": 3.0})

    router = LoadBalancingRouter(scheduler.state)
    job = Job(job_id="NEW", user_id="B", name="NEW", priority=Priority.MEDIUM,
              estimated_size_minutes=10, submitted_at=at(1))
    decision = router.route_job(job, now=at(1))

    assert decision.outcome == RoutingOutcome.NO_GPU_AVAILABLE
    assert decision.selected_gpu_id is None
    assert scheduler.state.get_gpu("GPU-1").assigned_user_id == "A"


# ---- 11/12/13/14. Cooldown lifecycle ----------------------------------------

def test_first_balancing_action_is_allowed():
    scheduler = sched(cooldown_minutes=5)
    add_gpus(scheduler, 2)
    holder = hold(scheduler, "HOLD", "A", gpu_count=2)
    set_utilization(scheduler, {"GPU-1": 80.0, "GPU-2": 1.0})
    waiter = hold(scheduler, "W1", "B", gpu_count=2, minute=1)
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-2")


def test_cooldown_starts_when_an_ask_is_resolved_and_blocks_a_repeat_ask():
    """GPU balanced (asked) at t, cooldown = 5 min; a second ask at
    t+2 is blocked."""
    scheduler = sched(cooldown_minutes=5)
    add_gpus(scheduler, 2)
    hold(scheduler, "HOLD", "A", gpu_count=2)
    set_utilization(scheduler, {"GPU-1": 80.0, "GPU-2": 1.0})
    hold(scheduler, "W1", "B", gpu_count=2, minute=1)
    scheduler.respond_to_prompt("GPU-2", YES, now=at(1.5))          # resolved (cooldown_start = 1.5)
    scheduler.try_allocate_all(now=at(1.5))

    hold(scheduler, "W2", "C", gpu_count=2, minute=2)               # cooldown_start + 0.5 min
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-2")
    assert scheduler.reclamation_engine.is_in_cooldown("GPU-2", at(2), timedelta(minutes=5)) is True


def test_action_allowed_again_once_the_cooldown_has_fully_elapsed():
    scheduler = sched(cooldown_minutes=5)
    add_gpus(scheduler, 2)
    hold(scheduler, "HOLD", "A", gpu_count=2)
    set_utilization(scheduler, {"GPU-1": 80.0, "GPU-2": 1.0})
    hold(scheduler, "W1", "B", gpu_count=2, minute=1)
    scheduler.respond_to_prompt("GPU-2", YES, now=at(1.5))           # cooldown_expiry = 6.5
    scheduler.try_allocate_all(now=at(1.5))

    hold(scheduler, "W2", "C", gpu_count=2, minute=6)                 # still inside cooldown
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-2")

    hold(scheduler, "W3", "D", gpu_count=2, minute=6.6)                # past cooldown_expiry
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-2")


def test_cooldown_is_time_based_not_call_count_based():
    """Many balancing-triggering calls inside the cooldown window must
    all be blocked, regardless of how many there were."""
    scheduler = sched(cooldown_minutes=5)
    add_gpus(scheduler, 2)
    hold(scheduler, "HOLD", "A", gpu_count=2)
    set_utilization(scheduler, {"GPU-1": 80.0, "GPU-2": 1.0})
    hold(scheduler, "W1", "B", gpu_count=2, minute=1)
    scheduler.respond_to_prompt("GPU-2", YES, now=at(1.5))
    scheduler.try_allocate_all(now=at(1.5))

    for i, minute in enumerate((2, 3, 4, 5, 6)):
        hold(scheduler, f"POLL-{i}", f"P{i}", gpu_count=2, minute=minute)
        assert not scheduler.reclamation_engine.has_pending_prompt("GPU-2")


# ---- 8 (spec section). Cooldown must not block *normal* allocation ---------

def test_cooldown_never_blocks_ordinary_routing_of_a_genuinely_free_gpu():
    """Once a GPU is genuinely released (NO / reclaimed), it must be
    immediately routable to a new job even while it is still "in
    cooldown" for reallocation-ask purposes - cooldown only prevents a
    repeated *ask*, never ordinary placement of new work onto a truly
    free GPU."""
    scheduler = sched(cooldown_minutes=5)
    add_gpus(scheduler, 1)
    hold(scheduler, "HOLD", "A", priority=Priority.LOW)
    waiter = hold(scheduler, "B-JOB", "B", priority=Priority.HIGH, minute=1)  # priority preemption ask
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1")

    scheduler.respond_to_prompt("GPU-1", NO, now=at(1.5))            # A releases -> cooldown starts, GPU genuinely free
    scheduler.try_allocate_all(now=at(1.5))

    assert scheduler.reclamation_engine.is_in_cooldown("GPU-1", at(1.6), timedelta(minutes=5)) is True
    assert waiter.status == JobStatus.RUNNING and waiter.assigned_gpu_ids == ["GPU-1"]  # not blocked


def test_cooldown_does_not_make_a_gpu_permanently_unavailable():
    """A GPU touched by a resolved reallocation ask must not stay
    effectively unusable for the rest of the cooldown window - once
    whoever ends up holding it finishes, the very next job can still
    get it immediately."""
    scheduler = sched(cooldown_minutes=5)
    add_gpus(scheduler, 1)
    hold(scheduler, "HOLD", "A", priority=Priority.LOW)
    waiter = hold(scheduler, "B-JOB", "B", priority=Priority.HIGH, minute=1)
    scheduler.respond_to_prompt("GPU-1", NO, now=at(1.5))          # A releases -> cooldown starts on GPU-1
    scheduler.try_allocate_all(now=at(1.5))
    assert waiter.assigned_gpu_ids == ["GPU-1"]

    scheduler.complete_job("B-JOB", now=at(2))                      # GPU-1 genuinely free again, still "in cooldown"
    assert scheduler.reclamation_engine.is_in_cooldown("GPU-1", at(2), timedelta(minutes=5)) is True

    fresh = hold(scheduler, "FRESH", "E", minute=2.1)
    assert fresh.assigned_gpu_ids == ["GPU-1"]


# ---- 9/15. Fluctuation does not thrash; a sustained flat reading still reclaims

def test_fluctuating_utilization_does_not_cause_repeated_balancing_requests():
    """4, 8, 5, 7, 4, 9 ... must not raise more than one ask total
    within the cooldown window - none of these dip low enough or long
    enough to matter for reclamation, and the oscillation itself must
    not spam reallocation asks."""
    scheduler = sched(cooldown_minutes=5)
    add_gpus(scheduler, 2)
    hold(scheduler, "HOLD", "A", gpu_count=2)
    values = [4.0, 8.0, 5.0, 7.0, 4.0, 9.0, 6.0, 5.0]
    prompts = 0
    for i, value in enumerate(values):
        set_utilization(scheduler, {"GPU-2": value}, minute=i * 0.5)
        hold(scheduler, f"POLL-{i}", f"P{i}", gpu_count=2, minute=i * 0.5 + 0.1)
        if scheduler.reclamation_engine.has_pending_prompt("GPU-2"):
            prompts += 1
            scheduler.respond_to_prompt("GPU-2", YES, now=at(i * 0.5 + 0.2))
            scheduler.try_allocate_all(now=at(i * 0.5 + 0.2))
    assert prompts <= 1


def test_sustained_flat_low_utilization_is_still_handled_by_reclamation_not_balancing():
    """A steady 5/5/5/5% reading is load balancing's business to
    notice (it's an available candidate at low utilization) but if it
    persists long enough to breach Tier 2, that is reclamation's
    separate, already-established mechanism - not a second one."""
    scheduler = sched()
    add_gpus(scheduler, 1)
    hold(scheduler, "HOLD", "A")
    for minute in range(0, 151, 10):
        decision = scheduler.record_utilization("GPU-1", 5.0, at(minute))
    assert decision is not None and decision.tier.value == "TIER_2"
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.IDLE_WARNING


# ---- 7/10. Multi-GPU load balancing -----------------------------------------

def test_multi_gpu_request_prefers_eligible_low_utilization_free_gpus():
    scheduler = sched()
    add_gpus(scheduler, 5)
    set_utilization(scheduler, {"GPU-1": 85.0, "GPU-2": 20.0, "GPU-3": 35.0, "GPU-4": 10.0, "GPU-5": 15.0})
    hold(scheduler, "BUSY", "H")                        # occupies GPU-1 (least-recently free is irrelevant; router picks by utilization among free ones)
    job = hold(scheduler, "MULTI", "M", gpu_count=3, minute=1)
    assert set(job.assigned_gpu_ids) <= {"GPU-2", "GPU-3", "GPU-4", "GPU-5"}
    assert len(job.assigned_gpu_ids) == 3
    assert job.status == JobStatus.RUNNING


def test_multi_gpu_request_never_grabs_the_three_numerically_lowest_while_ignoring_assignment_state():
    """GPU-1 is the lowest-utilization GPU in the pool but is actively
    assigned - the router must never pick it regardless of its number."""
    scheduler = sched()
    add_gpus(scheduler, 4)
    hold(scheduler, "HOLD", "A")
    set_utilization(scheduler, {"GPU-1": 1.0, "GPU-2": 40.0, "GPU-3": 45.0, "GPU-4": 50.0})
    job = hold(scheduler, "MULTI", "M", gpu_count=3, minute=1)
    assert "GPU-1" not in job.assigned_gpu_ids
    assert sorted(job.assigned_gpu_ids) == ["GPU-2", "GPU-3", "GPU-4"]


def test_partial_multi_gpu_placement_when_fewer_free_gpus_exist_than_requested():
    scheduler = sched()
    add_gpus(scheduler, 2)
    job = hold(scheduler, "MULTI", "M", gpu_count=5)
    assert len(job.assigned_gpu_ids) == 2
    assert job.gpus_still_needed == 3
    assert job.status == JobStatus.WAITING


# ---- 16/17. Multiple users / multiple jobs -----------------------------------

def test_multiple_users_each_land_on_their_own_least_utilized_free_gpu():
    scheduler = sched()
    add_gpus(scheduler, 4)
    set_utilization(scheduler, {"GPU-1": 60.0, "GPU-2": 10.0, "GPU-3": 70.0, "GPU-4": 5.0})
    jobs = [hold(scheduler, f"J{i}", f"U{i}", minute=i) for i in range(4)]
    assert all(j.status == JobStatus.RUNNING for j in jobs)
    assert len({j.assigned_gpu_ids[0] for j in jobs}) == 4
    first_two = sorted(j.assigned_gpu_ids[0] for j in jobs[:2])
    assert first_two == ["GPU-2", "GPU-4"]  # the two least-utilized GPUs went first


def test_multiple_waiting_jobs_only_the_ones_that_fit_are_routed():
    scheduler = sched()
    add_gpus(scheduler, 2)
    jobs = [hold(scheduler, f"J{i}", f"U{i}", minute=i) for i in range(5)]
    running = [j for j in jobs if j.status == JobStatus.RUNNING]
    waiting = [j for j in jobs if j.status == JobStatus.WAITING]
    assert len(running) == 2 and len(waiting) == 3


# ---- 18. Large pool -----------------------------------------------------------

def test_large_gpu_pool_routes_to_the_least_utilized_free_gpus():
    scheduler = sched()
    n = 60
    add_gpus(scheduler, n)
    readings = {f"GPU-{i}": float((i * 17) % 100) for i in range(1, n + 1)}
    set_utilization(scheduler, readings)
    job = hold(scheduler, "J1", "U1")
    expected = min(readings, key=lambda gid: (readings[gid], gid))
    assert job.assigned_gpu_ids == [expected]


# ---- 19/20. Decision trace and events ----------------------------------------

def test_routing_decision_trace_lists_every_candidate_with_eligibility_and_reason():
    scheduler = sched()
    add_gpus(scheduler, 2)
    hold(scheduler, "HOLD", "A")
    set_utilization(scheduler, {"GPU-1": 90.0})
    waiter = hold(scheduler, "W1", "B", minute=1)

    trace = scheduler.last_routing_decision
    assert trace.outcome == RoutingOutcome.ROUTED
    assert trace.selected_gpu_id == "GPU-2"
    ids = {c.gpu_id for c in trace.candidates}
    assert ids == {"GPU-1", "GPU-2"}
    by_id = {c.gpu_id: c for c in trace.candidates}
    assert by_id["GPU-1"].available is False
    assert by_id["GPU-2"].available is True
    assert "GPU-2" in trace.reason


def test_balancing_trace_reports_cooldown_status_and_skip_reasons():
    scheduler = sched(cooldown_minutes=5)
    add_gpus(scheduler, 3)
    hold(scheduler, "HOLD", "A", gpu_count=3)
    set_utilization(scheduler, {"GPU-1": 80.0, "GPU-2": 80.0, "GPU-3": 1.0})
    hold(scheduler, "W1", "B", gpu_count=2, minute=1)

    traces = [t for t in scheduler.last_balancing_traces if t.path == ReallocationPath.EXCESS_CAPACITY]
    assert len(traces) == 1
    trace = traces[0]
    assert trace.job_id == "W1" and trace.deficit == 2
    by_id = {c.gpu_id: c for c in trace.candidates}
    assert by_id["GPU-3"].eligible is True and by_id["GPU-3"].selected is True
    assert by_id["GPU-1"].eligible is False and by_id["GPU-1"].skip_reason == "not underutilized enough"
    assert all(c.in_cooldown is False for c in trace.candidates)  # nothing asked yet this run

    scheduler.respond_to_prompt("GPU-3", YES, now=at(1.5))
    scheduler.try_allocate_all(now=at(1.5))
    hold(scheduler, "W2", "C", gpu_count=2, minute=2)
    # W1 is still WAITING (its own ask was declined) and gets
    # re-evaluated on this same pass alongside the new W2 - filter by
    # job_id, not just path, to get W2's own trace specifically.
    trace2 = next(t for t in scheduler.last_balancing_traces
                  if t.path == ReallocationPath.EXCESS_CAPACITY and t.job_id == "W2")
    by_id2 = {c.gpu_id: c for c in trace2.candidates}
    assert by_id2["GPU-3"].in_cooldown is True
    assert by_id2["GPU-3"].skip_reason == "in cooldown"


def test_balance_events_are_logged_for_every_routing_decision():
    scheduler = sched()
    add_gpus(scheduler, 2)
    hold(scheduler, "J1", "U1")
    events = [e for e in scheduler.state.events if e.event_type.value == "BALANCE"]
    assert events and "GPU-1" in events[-1].message or "evaluated" in events[0].message.lower()


# ---- 21. State consistency after balancing ------------------------------------

def test_state_stays_consistent_after_a_reallocation_ask_and_reroute():
    scheduler = sched(cooldown_minutes=5)
    add_gpus(scheduler, 2)
    holder = hold(scheduler, "HOLD", "A", gpu_count=2)
    set_utilization(scheduler, {"GPU-1": 80.0, "GPU-2": 1.0})
    waiter = hold(scheduler, "W1", "B", gpu_count=2, minute=1)   # gpu_count>1: excess-capacity path applies

    scheduler.respond_to_prompt("GPU-2", NO, now=at(1.5))         # A releases GPU-2
    scheduler.try_allocate_all(now=at(1.5))

    assert holder.assigned_gpu_ids == ["GPU-1"]                   # A keeps its actively-used GPU
    assert scheduler.state.get_gpu("GPU-1").assigned_user_id == "A"
    assert scheduler.state.get_gpu("GPU-2").assigned_user_id == "B"
    assert waiter.assigned_gpu_ids == ["GPU-2"] and waiter.gpus_still_needed == 1
    assert waiter.status == JobStatus.WAITING                      # still short one GPU
    assert scheduler.allocation_engine.get_gpus_for_user("A") == ["GPU-1"]
    assert scheduler.allocation_engine.get_gpus_for_user("B") == ["GPU-2"]
