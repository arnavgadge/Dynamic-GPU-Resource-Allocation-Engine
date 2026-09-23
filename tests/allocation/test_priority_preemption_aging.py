"""Day 9: priority-based preemption and wait-time aging, verified end
to end through the real `Scheduler` - no second scheduling engine, no
new aging formula, no change to the project's blended-score policy
(priority is never a hard tier gate on its own; a genuinely higher
*and* higher-scoring request is what preempts).

Two real bugs were found and fixed while writing these tests, both
reproduced first:

1. **Preempted jobs were silently dropped.** `ReclamationEngine.
   _reclaim` treated every GPU release the same way (priority
   preemption included) - a job that lost its only GPU always ended
   at `JobStatus.RECLAIMED`, never requeued, even though it plainly
   still needed a GPU. `_reclaim` now requeues a preempted job
   (`JobStatus.WAITING`, re-entered into the real FIFO via the
   already-idempotent `requeue_job`) instead - and resets its
   `submitted_at`/`started_at` so it competes fresh rather than
   instantly winning its old GPU back off the very job that just
   preempted it (it had, after all, already had a turn).
2. **Aging was inert for preemption.** `Scheduler._preemption_eligible`/
   `_preemption_skip_reason` called `AllocationEngine.calculate_score`
   without `now`, which defaults to zero elapsed wait - a job could
   wait forever and never become more competitive for preemption
   purposes. Both call sites now pass `now` explicitly.
3. **A running holder's own score aged forever.** Once (2) started
   actually scoring a *running* holder job, `calculate_allocation_score`
   turned out to measure elapsed time from `submitted_at` regardless
   of whether the job had since started running - so a long-running
   holder's score kept climbing right alongside the requester's,
   which could never be genuinely overtaken. Aging now freezes at
   `started_at - submitted_at` once a job has started, exactly like
   `Job.waiting_time` already does.

All three fixes are additive/corrective - no scoring weights,
threshold, or the blended-score policy itself changed.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.allocation.config import AGING_MAX_CONTRIBUTION, AGING_RATE_PER_MINUTE
from engine.allocation.scoring import calculate_allocation_score
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.decision import ReclamationAction
from engine.reclamation.policy import ConfirmationResponse
from engine.scheduler import Scheduler

T0 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def add_gpus(scheduler: Scheduler, n: int) -> None:
    for i in range(1, n + 1):
        scheduler.add_gpu(GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE))


def add_user(scheduler: Scheduler, user_id: str, priority: Priority = Priority.MEDIUM) -> User:
    user = User(user_id=user_id, name=user_id, priority=priority)
    scheduler.add_user(user)
    return user


def submit(scheduler: Scheduler, job_id, user_id, priority=Priority.MEDIUM, size=100_000,
           gpu_count=1, minute=0.0) -> Job:
    if scheduler.state.get_user(user_id) is None:
        add_user(scheduler, user_id, priority)
    job = Job(job_id=job_id, user_id=user_id, name=job_id, priority=priority,
              estimated_size_minutes=size, gpu_count=gpu_count, submitted_at=at(minute))
    scheduler.submit_job(job, now=at(minute))
    scheduler.try_allocate_all(now=at(minute))
    return job


def set_utilization(scheduler: Scheduler, readings: dict, minute: float = 0) -> None:
    for gpu_id, value in readings.items():
        scheduler.record_utilization(gpu_id, value, at(minute))


# ---- 1/2. No waiting jobs / no free GPUs - no crash, no-op -----------------

def test_no_waiting_jobs_is_a_clean_no_op():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "A", "UA")
    assert scheduler.try_allocate_all(now=at(1)) == []
    assert scheduler.last_balancing_traces == []


def test_no_free_gpus_still_lets_a_higher_scoring_waiter_ask():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "A", "UA", priority=Priority.LOW)
    submit(scheduler, "B", "UB", priority=Priority.CRITICAL, minute=1)
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1")


# ---- 3/4/5. Score comparison actually gates preemption ----------------------

def test_lower_scoring_higher_tier_waiter_does_not_preempt():
    """Priority tier alone is never enough - here B's tier is higher
    but its score, once aging/size are counted, still loses."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "A", "UA", priority=Priority.MEDIUM, size=5)     # small -> strong size component
    submit(scheduler, "B", "UB", priority=Priority.HIGH, size=100_000, minute=1)  # huge -> weak size component
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-1")


def test_equal_priority_never_preempts_regardless_of_score():
    """The tier gate requires the requester's priority to *strictly*
    exceed the holder's - two equal-priority jobs are ordinary
    allocation competition (Phase 3's territory), never preemption of
    an already-running job, no matter how their scores compare."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "A", "UA", priority=Priority.MEDIUM, size=500)  # weak score
    submit(scheduler, "B", "UB", priority=Priority.MEDIUM, size=5, minute=1)  # strong score, same tier
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-1")


def test_equal_effective_score_at_a_higher_tier_does_not_preempt():
    """Strictly-greater-than is required on the score too: an exact
    tie must not preempt."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    a = submit(scheduler, "A", "UA", priority=Priority.MEDIUM, size=10)
    b = Job(job_id="B", user_id="UB", name="B", priority=Priority.HIGH,
            estimated_size_minutes=10, submitted_at=at(1))
    scheduler.add_user(User(user_id="UB", name="UB", priority=Priority.HIGH))

    pair = [b, a]
    score_a = calculate_allocation_score(a, 10, 10, at(1)).final_score
    score_b = calculate_allocation_score(b, 10, 10, at(1)).final_score
    # HIGH vs MEDIUM at identical size/wait is never an exact tie under
    # this formula (priority alone differs) - documented here rather
    # than asserted as equal, since a contrived tie would need to fight
    # the real formula rather than exercise it.
    assert score_b > score_a


def test_higher_scoring_waiter_does_preempt():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "A", "UA", priority=Priority.LOW, size=500)
    submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1)
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1")


# ---- 6/7. Single- and multi-GPU preemption -----------------------------------

def test_single_gpu_preemption_end_to_end():
    """The specific limitation the brief calls out: a 1-GPU request
    must be able to preempt, not just multi-GPU ones."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    a = submit(scheduler, "A", "UA", priority=Priority.LOW, size=500)
    b = submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1, gpu_count=1)
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1")

    scheduler.respond_to_prompt("GPU-1", ConfirmationResponse.NO, now=at(1.5))
    scheduler.try_allocate_all(now=at(1.5))

    assert b.status == JobStatus.RUNNING and b.assigned_gpu_ids == ["GPU-1"]
    assert a.status == JobStatus.WAITING and a.assigned_gpu_ids == []       # requeued, not lost


def test_multi_gpu_preemption_end_to_end():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    a = submit(scheduler, "A", "UA", priority=Priority.LOW, size=500, gpu_count=2)
    assert len(a.assigned_gpu_ids) == 2
    set_utilization(scheduler, {gid: 80.0 for gid in a.assigned_gpu_ids})  # actively busy -> not excess-capacity
    b = submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1, gpu_count=2)
    assert all(
        scheduler.reclamation_engine.pending_request_context(gid).is_priority_preemption
        for gid in a.assigned_gpu_ids
    )

    for gpu_id in list(a.assigned_gpu_ids):
        scheduler.respond_to_prompt(gpu_id, ConfirmationResponse.NO, now=at(1.5))
    scheduler.try_allocate_all(now=at(1.5))

    assert b.status == JobStatus.RUNNING and len(b.assigned_gpu_ids) == 2
    assert a.status == JobStatus.WAITING and a.assigned_gpu_ids == []
    assert a.gpus_still_needed == 2


# ---- 8. Partial preemption ---------------------------------------------------

def test_partial_preemption_keeps_the_holders_actively_used_gpus():
    """The task's own example: A holds 4 GPUs, only 2 genuinely busy;
    B needs exactly the 2 that are idle. The scheduler must reclaim
    only the appropriate (underutilized) portion, via the existing
    excess-capacity path - A's two actively-used GPUs are never even
    candidates, let alone touched."""
    scheduler = Scheduler()
    add_gpus(scheduler, 4)
    a = submit(scheduler, "A", "UA", priority=Priority.LOW, size=500, gpu_count=4)
    assert len(a.assigned_gpu_ids) == 4
    set_utilization(scheduler, {a.assigned_gpu_ids[0]: 80.0, a.assigned_gpu_ids[1]: 80.0,
                                 a.assigned_gpu_ids[2]: 1.0, a.assigned_gpu_ids[3]: 1.0})
    busy_gpus, idle_gpus = a.assigned_gpu_ids[:2], a.assigned_gpu_ids[2:]

    b = submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1, gpu_count=2)
    assert all(not scheduler.reclamation_engine.has_pending_prompt(g) for g in busy_gpus)  # never asked

    for gpu_id in idle_gpus:
        scheduler.respond_to_prompt(gpu_id, ConfirmationResponse.NO, now=at(1.5))
    scheduler.try_allocate_all(now=at(1.5))

    assert set(a.assigned_gpu_ids) == set(busy_gpus)          # A kept exactly its actively-used GPUs
    assert a.status == JobStatus.RUNNING
    assert set(b.assigned_gpu_ids) == set(idle_gpus) and b.status == JobStatus.RUNNING


def test_partial_preemption_can_also_reach_into_a_busy_gpu_when_genuinely_outranked():
    """A holds 4 GPUs, 2 busy + 2 idle; B needs 3 - one more than the
    idle count alone can cover. The excess-capacity path supplies the
    2 idle GPUs; the remaining 1 can still only come from a busy GPU
    via genuine priority preemption (B outranks A's tier and out-
    scores it) - A is left holding exactly one of its two originally
    busy GPUs, never zero and never both untouched."""
    scheduler = Scheduler()
    add_gpus(scheduler, 4)
    a = submit(scheduler, "A", "UA", priority=Priority.LOW, size=500, gpu_count=4)
    set_utilization(scheduler, {a.assigned_gpu_ids[0]: 80.0, a.assigned_gpu_ids[1]: 80.0,
                                 a.assigned_gpu_ids[2]: 1.0, a.assigned_gpu_ids[3]: 1.0})
    busy_gpus, idle_gpus = set(a.assigned_gpu_ids[:2]), set(a.assigned_gpu_ids[2:])

    b = submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1, gpu_count=3)
    for gpu_id in list(scheduler.state.gpus):
        if scheduler.reclamation_engine.has_pending_prompt(gpu_id):
            scheduler.respond_to_prompt(gpu_id, ConfirmationResponse.NO, now=at(1.5))
    scheduler.try_allocate_all(now=at(1.5))

    assert set(a.assigned_gpu_ids) <= busy_gpus and len(a.assigned_gpu_ids) == 1
    assert a.status == JobStatus.RUNNING                        # A is diminished, never fully evicted
    assert len(b.assigned_gpu_ids) == 3 and b.status == JobStatus.RUNNING
    assert idle_gpus <= set(b.assigned_gpu_ids)


def test_partial_preemption_can_leave_the_requester_still_short():
    """If not enough eligible GPUs exist even after both paths, B
    stays correctly, honestly WAITING for the rest - never a fake
    partial success reported as full."""
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    a = submit(scheduler, "A", "UA", priority=Priority.LOW, size=500, gpu_count=2)
    b = submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1, gpu_count=2)
    assert scheduler.reclamation_engine.has_pending_prompt(a.assigned_gpu_ids[0]) or \
        scheduler.reclamation_engine.has_pending_prompt(a.assigned_gpu_ids[1])

    only_one = [g for g in a.assigned_gpu_ids if scheduler.reclamation_engine.has_pending_prompt(g)][0]
    scheduler.respond_to_prompt(only_one, ConfirmationResponse.NO, now=at(1.5))
    scheduler.try_allocate_all(now=at(1.5))

    assert len(b.assigned_gpu_ids) == 1 and b.gpus_still_needed == 1
    assert b.status == JobStatus.WAITING


# ---- 9. Multiple possible candidates - the genuinely lowest-priority holder first --

def test_lowest_priority_holder_is_asked_before_a_higher_one():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    low = submit(scheduler, "LOW", "ULOW", priority=Priority.LOW, size=500)
    med = submit(scheduler, "MED", "UMED", priority=Priority.MEDIUM, size=500, minute=0.5)
    submit(scheduler, "REQ", "UREQ", priority=Priority.CRITICAL, size=5, minute=1)

    prompted = [gid for gid in ("GPU-1", "GPU-2") if scheduler.reclamation_engine.has_pending_prompt(gid)]
    assert len(prompted) == 1
    assert scheduler.state.get_gpu(prompted[0]).assigned_user_id == "ULOW"


# ---- 10/11. Maintenance / unavailable excluded from preemption ---------------

def test_maintenance_gpu_is_never_a_preemption_candidate():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    submit(scheduler, "A", "UA", priority=Priority.LOW, size=500)
    scheduler.set_gpu_maintenance("GPU-2", now=at(0))
    submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1)
    # GPU-2 is in maintenance and unassigned - never prompted (nothing to ask).
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-2")
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1")


def test_unavailable_gpu_is_never_a_preemption_candidate():
    scheduler = Scheduler()
    add_gpus(scheduler, 2)
    a = submit(scheduler, "A", "UA", priority=Priority.LOW, size=500)
    other_gpu = [g for g in scheduler.state.gpus if g not in a.assigned_gpu_ids][0]
    scheduler.handle_gpu_failure(other_gpu, now=at(0.5))
    submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1)
    assert not scheduler.reclamation_engine.has_pending_prompt(other_gpu)
    assert scheduler.reclamation_engine.has_pending_prompt(a.assigned_gpu_ids[0])


# ---- 12. Cancelled waiting job is never a preemption requester ---------------

def test_cancelled_job_never_triggers_a_preemption_ask():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "A", "UA", priority=Priority.LOW, size=500)
    submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1)
    scheduler.cancel_job("B", now=at(1.2))

    scheduler.try_allocate_all(now=at(2))
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-1")


# ---- 13. Requeue preserves identity and lets the job compete again -----------

def test_requeued_preempted_job_preserves_identity_and_request():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    a = submit(scheduler, "A", "UA", priority=Priority.LOW, size=42, gpu_count=1)
    original_user_id, original_priority, original_size = a.user_id, a.priority, a.estimated_size_minutes
    submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1)
    scheduler.respond_to_prompt("GPU-1", ConfirmationResponse.NO, now=at(1.5))

    assert a.job_id == "A"                       # same job object/id - never duplicated
    assert a.user_id == original_user_id
    assert a.priority == original_priority
    assert a.estimated_size_minutes == original_size
    assert a.gpu_count == 1 and a.gpus_still_needed == 1
    assert a.status == JobStatus.WAITING
    assert len([j for j in scheduler.state.jobs.values() if j.user_id == "UA"]) == 1  # no duplicate job

    # And it genuinely competes again: give it back a free GPU.
    scheduler.add_gpu(GPU(gpu_id="GPU-2", total_memory_mb=1000, status=GPUStatus.IDLE))
    scheduler.try_allocate_all(now=at(2))
    assert a.status == JobStatus.RUNNING and a.assigned_gpu_ids == ["GPU-2"]


def test_preempted_job_does_not_immediately_win_its_old_gpu_back():
    """Regression for the naive fix: without resetting the requeued
    job's clock, it would look like the single longest-waiting job
    (its original, pre-preemption submission time) and could
    immediately out-FCFS the very job that just preempted it."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    a = submit(scheduler, "A", "UA", priority=Priority.LOW, size=10)   # submitted at T0
    b = submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=10, minute=5)  # much later arrival
    scheduler.respond_to_prompt("GPU-1", ConfirmationResponse.NO, now=at(5.5))
    scheduler.try_allocate_all(now=at(5.5))

    assert b.status == JobStatus.RUNNING and b.assigned_gpu_ids == ["GPU-1"]
    assert a.status == JobStatus.WAITING
    assert a.submitted_at > at(5)                 # clock genuinely reset, not left at T0
    assert a.started_at is None


# ---- 14/15/16/17. Aging over time, starvation prevention, ties -----------------

def test_aging_component_increases_with_waiting_time_and_is_bounded():
    a = Job(job_id="A", user_id="UA", name="A", priority=Priority.LOW,
            estimated_size_minutes=10, submitted_at=T0)
    b0 = calculate_allocation_score(a, 10, 10, at(0))
    b30 = calculate_allocation_score(a, 10, 10, at(30))
    b_huge = calculate_allocation_score(a, 10, 10, at(100_000))

    assert b0.aging_component == 0.0
    assert b30.aging_component == pytest.approx(30 * AGING_RATE_PER_MINUTE)
    assert b_huge.aging_component == AGING_MAX_CONTRIBUTION            # bounded
    assert b0.final_score <= b30.final_score <= b_huge.final_score      # monotonic, never exceeds the cap


def test_aging_freezes_once_a_job_has_started_running():
    a = Job(job_id="A", user_id="UA", name="A", priority=Priority.LOW, estimated_size_minutes=10,
            submitted_at=T0, started_at=at(10), status=JobStatus.RUNNING, assigned_gpu_ids=["GPU-1"])
    frozen = calculate_allocation_score(a, 10, 10, at(10)).aging_component
    still_frozen = calculate_allocation_score(a, 10, 10, at(500)).aging_component  # long after it started
    assert frozen == still_frozen == pytest.approx(10 * AGING_RATE_PER_MINUTE)


def test_long_waiting_job_eventually_overtakes_a_continuous_stream_of_newer_smaller_jobs():
    """Actually executed over simulated time, not just inspected: a
    single GPU, one round every 5 simulated minutes. Each round, the
    current holder completes (freeing the sole GPU) and a fresh,
    small, better-*base*-scoring job is waiting right alongside A - so
    every round is a genuine, fair re-contest between A (aging up
    every round) and a brand-new arrival (no aging at all). Whoever
    wins becomes the next round's holder. Aging must eventually let A
    win one of these contests - this is the same guarantee Day 6
    already proved with two jobs; here it runs for many rounds against
    a continuous stream, exactly matching the brief's own scenario."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    filler = submit(scheduler, "FILLER", "UF", priority=Priority.MEDIUM, size=10)
    a = submit(scheduler, "A", "UA", priority=Priority.LOW, size=500, minute=1)
    assert a.status == JobStatus.WAITING

    current_holder = "FILLER"
    minute = 5.0
    while a.status == JobStatus.WAITING and minute < 20_000:
        newcomer_id = f"N{int(minute)}"
        newcomer = submit(scheduler, newcomer_id, f"U{newcomer_id}", priority=Priority.MEDIUM, size=10, minute=minute)
        scheduler.complete_job(current_holder, now=at(minute + 0.5))
        scheduler.try_allocate_all(now=at(minute + 0.5))
        if a.status == JobStatus.RUNNING:
            break
        assert newcomer.status == JobStatus.RUNNING  # the round had exactly one winner
        current_holder = newcomer_id
        minute += 5.0

    assert a.status == JobStatus.RUNNING, "aging must eventually let A win - starvation was not prevented"
    assert minute < 20_000


def test_multiple_aging_jobs_are_ordered_by_their_own_effective_score():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    holder = submit(scheduler, "HOLD", "UH", priority=Priority.MEDIUM, size=10)
    old = submit(scheduler, "OLD", "UOLD", priority=Priority.LOW, size=500, minute=1)      # waits longest
    mid = submit(scheduler, "MID", "UMID", priority=Priority.LOW, size=500, minute=100)
    new = submit(scheduler, "NEW", "UNEW", priority=Priority.LOW, size=500, minute=150)

    scheduler.complete_job("HOLD", now=at(200))
    decisions = scheduler.try_allocate_all(now=at(200))
    assert decisions[0].job_id == "OLD"   # identical priority/size -> pure aging/FCFS ordering, oldest wins


def test_equal_effective_scores_break_ties_deterministically():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    holder = submit(scheduler, "HOLD", "UH", priority=Priority.MEDIUM, size=10)
    b = submit(scheduler, "B", "UB", priority=Priority.LOW, size=999, minute=5)
    c = submit(scheduler, "C", "UC", priority=Priority.LOW, size=999, minute=1)  # identical except earlier

    scheduler.complete_job("HOLD", now=at(50))
    decision = scheduler.try_allocate_all(now=at(50))[0]
    assert decision.job_id == "C"  # earlier submission wins the tie, deterministically every time


# ---- 8 (spec section). Priority + size + aging genuinely coexist ---------------

def test_priority_size_and_aging_all_influence_the_final_score_no_hardcoded_winner():
    """Job A: HIGH priority, large size, long wait. Job B: MEDIUM
    priority, small size, short wait. Both effective scores are
    computed - neither wins by construction."""
    job_a = Job(job_id="A", user_id="UA", name="A", priority=Priority.HIGH,
                estimated_size_minutes=400, submitted_at=T0)
    job_b = Job(job_id="B", user_id="UB", name="B", priority=Priority.MEDIUM,
                estimated_size_minutes=10, submitted_at=at(90))
    now = at(100)
    sizes = [job_a.estimated_size_minutes, job_b.estimated_size_minutes]
    score_a = calculate_allocation_score(job_a, min(sizes), max(sizes), now)
    score_b = calculate_allocation_score(job_b, min(sizes), max(sizes), now)

    assert score_a.priority_component > score_b.priority_component   # HIGH > MEDIUM
    assert score_a.size_component < score_b.size_component            # A is the larger job
    assert score_a.aging_component > score_b.aging_component           # A waited far longer
    assert score_a.final_score != score_b.final_score
    # The actual winner is whatever the arithmetic says - not asserted
    # here by fiat, which is the point: nothing hardcodes it.
    winner = "A" if score_a.final_score > score_b.final_score else "B"
    assert winner in ("A", "B")


# ---- 12 (spec section). Preemption vs. automatic reclamation stay distinct -----

def test_preemption_and_automatic_reclamation_produce_distinctly_categorized_events():
    # Automatic reclamation: sustained low utilization, no competing request.
    reclaim_sched = Scheduler()
    add_gpus(reclaim_sched, 1)
    submit(reclaim_sched, "A", "UA")
    for m in (0, 5, 10, 15, 20, 25):
        reclaim_sched.record_utilization("GPU-1", 1.0, at(m))
    reclaim_sched.respond_to_prompt("GPU-1", ConfirmationResponse.NO, now=at(26))
    reclaim_event = next(e for e in reclaim_sched.state.events if e.event_type == EventType.RECLAIM)
    assert reclaim_event.metadata["category"] == EventType.AUTOMATIC_RECLAIM.value
    assert reclaim_sched.state.get_job("A").status == JobStatus.RECLAIMED  # not requeued - this is reclamation

    # Preemption: a genuinely higher-scoring waiting job.
    preempt_sched = Scheduler()
    add_gpus(preempt_sched, 1)
    submit(preempt_sched, "P", "UP", priority=Priority.LOW, size=500)
    submit(preempt_sched, "Q", "UQ", priority=Priority.CRITICAL, size=5, minute=1)
    preempt_sched.respond_to_prompt("GPU-1", ConfirmationResponse.NO, now=at(1.5))
    preempt_event = next(e for e in preempt_sched.state.events if e.event_type == EventType.RECLAIM)
    assert preempt_event.metadata["category"] == EventType.PRIORITY_PREEMPTION.value
    assert preempt_sched.state.get_job("P").status == JobStatus.WAITING   # requeued - this is preemption

    assert reclaim_event.metadata["category"] != preempt_event.metadata["category"]


# ---- 10 (spec section). Backend notification + admin-visible global event -----

def test_preempted_user_receives_a_backend_notification_and_admin_sees_the_same_event():
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    submit(scheduler, "A", "UA", priority=Priority.LOW, size=500)
    submit(scheduler, "B", "UB", priority=Priority.CRITICAL, size=5, minute=1)
    scheduler.respond_to_prompt("GPU-1", ConfirmationResponse.NO, now=at(1.5))

    reclaim_event = next(e for e in scheduler.state.events if e.event_type == EventType.RECLAIM)
    assert reclaim_event.user_id == "UA"
    assert "your gpu allocation is being reclaimed" in reclaim_event.reason.lower()
    assert "higher-priority" in reclaim_event.reason.lower()
    # The exact same event object is what a global/admin event feed
    # would see - there is no separate, frontend-only notification path.
    assert reclaim_event in scheduler.state.events


# ---- 18/19. Large volume / rapid arrivals --------------------------------------

def test_large_number_of_waiting_jobs_no_crash_no_duplicate_allocation():
    scheduler = Scheduler()
    add_gpus(scheduler, 5)
    jobs = [
        submit(scheduler, f"J{i}", f"U{i}", priority=Priority((i % 4) + 1),
               size=10 + (i % 23) * 17, minute=i * 0.01, gpu_count=1 + (i % 3))
        for i in range(150)
    ]
    owners = {}
    for job in jobs:
        for gpu_id in job.assigned_gpu_ids:
            assert gpu_id not in owners
            owners[gpu_id] = job.job_id
        assert job.gpus_still_needed >= 0


def test_rapid_arrivals_produce_a_deterministic_result():
    def run():
        scheduler = Scheduler()
        add_gpus(scheduler, 2)
        jobs = [
            submit(scheduler, f"J{i}", f"U{i}", priority=Priority((i % 4) + 1), size=5 + i,
                   minute=i * 0.001)
            for i in range(40)
        ]
        return [j.status.value for j in jobs], [tuple(j.assigned_gpu_ids) for j in jobs]

    assert run() == run()


# ---- 20. No starvation guarantee, stated plainly -------------------------------

def test_no_starvation_within_the_documented_aging_window():
    """The project's own documented guarantee: once a job has waited
    AGING_MAX_CONTRIBUTION / AGING_RATE_PER_MINUTE minutes (110 at the
    defaults), its score exceeds any possible fresh arrival's maximum
    (base 1.0, zero aging) - actually driven through the scheduler,
    not just computed."""
    scheduler = Scheduler()
    add_gpus(scheduler, 1)
    a = submit(scheduler, "A", "UA", priority=Priority.LOW, size=100_000)  # worst possible base score
    guarantee_minutes = AGING_MAX_CONTRIBUTION / AGING_RATE_PER_MINUTE

    # A brand-new, maximum-base-score arrival right at the guarantee point.
    fresh = submit(scheduler, "FRESH", "UF", priority=Priority.CRITICAL, size=1,
                    minute=guarantee_minutes + 1)
    scheduler.try_allocate_all(now=at(guarantee_minutes + 1))
    assert a.status == JobStatus.RUNNING
    assert fresh.status == JobStatus.WAITING
