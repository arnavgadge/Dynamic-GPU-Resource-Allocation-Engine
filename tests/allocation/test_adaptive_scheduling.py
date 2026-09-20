"""Day 6: the adaptive FCFS / SJF scheduling policy, end to end.

The policy (unchanged - this file verifies it, it does not redefine
it): waiting jobs whose sizes are within the 20% similarity threshold
are served FCFS (earliest `submitted_at`); jobs whose sizes differ by
more are served by the blended allocation score (0.6 x priority +
0.4 x size + aging) via the Priority Queue. Priority is *not* a hard
gate in either mode - the blended-score policy is deliberate (the
`ml_video` scenario and `test_engine.py::test_scenario_2` document
it), and nothing here changes that.

Every contest goes through the real `Scheduler`: one GPU is held by a
"holder" job, the contenders queue up, the holder completes, and the
single decision that follows is inspected - including its structured
trace (`AllocationDecision.explain()`, `size_spread`, per-candidate
arrival position and score breakdown). Deterministic data throughout.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.allocation.config import JOB_SIZE_SIMILARITY_THRESHOLD
from engine.allocation.decision import AllocationPolicy
from engine.allocation.similarity import are_job_sizes_similar, relative_size_spread, sizes_are_similar
from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.scheduler import Scheduler

NOW = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def job(job_id, size, priority=Priority.MEDIUM, at=0, gpu_count=1, user=None) -> Job:
    return Job(job_id=job_id, user_id=user or f"U-{job_id}", name=job_id, priority=priority,
               estimated_size_minutes=size, gpu_count=gpu_count,
               submitted_at=NOW + timedelta(minutes=at))


def scheduler_with(n_gpus: int) -> Scheduler:
    scheduler = Scheduler()
    for i in range(1, n_gpus + 1):
        scheduler.add_gpu(GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE))
    return scheduler


def enqueue(scheduler: Scheduler, *jobs: Job) -> None:
    for j in jobs:
        if scheduler.state.get_user(j.user_id) is None:
            scheduler.add_user(User(user_id=j.user_id, name=j.user_id, priority=Priority.MEDIUM))
        scheduler.submit_job(j, now=j.submitted_at)


def contest(*contenders: Job, decide_at_minutes: float = 10):
    """One GPU, held by a holder job that then completes, so the
    contenders are genuinely waiting together when the GPU frees.
    Returns the single allocation decision that follows."""
    scheduler = scheduler_with(1)
    holder = job("HOLDER", 30, at=-1000)
    enqueue(scheduler, holder)
    scheduler.try_allocate_all(now=NOW - timedelta(minutes=999))
    enqueue(scheduler, *contenders)
    decide_at = NOW + timedelta(minutes=decide_at_minutes)
    scheduler.complete_job("HOLDER", now=decide_at)
    decisions = scheduler.try_allocate_all(now=decide_at)
    return scheduler, decisions


# ---- the 20% rule itself ---------------------------------------------------

def test_are_job_sizes_similar_examples_from_the_spec():
    assert are_job_sizes_similar([job("A", 10), job("B", 12)]) is True    # 16.7%
    assert are_job_sizes_similar([job("A", 15), job("B", 60)]) is False   # 75%


def test_threshold_is_explicit_and_configurable():
    jobs = [job("A", 40), job("B", 60)]  # spread 33.3%
    assert JOB_SIZE_SIMILARITY_THRESHOLD == 0.20
    assert are_job_sizes_similar(jobs) is False
    assert are_job_sizes_similar(jobs, threshold=0.5) is True
    assert sizes_are_similar([40, 60], threshold=0.5) is True


def test_boundary_exactly_at_20_percent_is_similar_and_just_above_is_not():
    assert relative_size_spread([80, 100]) == pytest.approx(0.20)
    assert are_job_sizes_similar([job("A", 80), job("B", 100)]) is True       # exactly 20% -> FCFS
    assert are_job_sizes_similar([job("A", 79), job("B", 100)]) is False      # 21% -> score-based


# ---- FCFS mode -------------------------------------------------------------

def test_one_waiting_job_is_trivially_fcfs():
    _, decisions = contest(job("A", 30, at=1))
    assert decisions[0].job_id == "A"
    assert decisions[0].policy == AllocationPolicy.FCFS


def test_two_equal_size_jobs_earliest_arrival_wins():
    _, decisions = contest(job("LATE", 30, at=5), job("EARLY", 30, at=1))
    assert decisions[0].policy == AllocationPolicy.FCFS
    assert decisions[0].job_id == "EARLY"


def test_two_jobs_within_20_percent_use_fcfs():
    _, decisions = contest(job("A", 10, at=1), job("B", 12, at=2))
    assert decisions[0].policy == AllocationPolicy.FCFS
    assert decisions[0].job_id == "A"


def test_exactly_at_the_20_percent_boundary_still_uses_fcfs():
    _, decisions = contest(job("A", 100, at=1), job("B", 80, at=2))
    assert decisions[0].policy == AllocationPolicy.FCFS
    assert decisions[0].job_id == "A"  # the *larger* job wins because it arrived first


def test_same_size_different_priority_is_still_fcfs_priority_is_not_a_hard_gate():
    _, decisions = contest(job("LOW-EARLY", 30, priority=Priority.LOW, at=1),
                           job("HIGH-LATE", 30, priority=Priority.HIGH, at=2))
    assert decisions[0].policy == AllocationPolicy.FCFS
    assert decisions[0].job_id == "LOW-EARLY"


def test_same_size_different_waiting_time_longest_wait_wins():
    _, decisions = contest(job("W5", 30, at=5), job("W1", 30, at=1), job("W3", 30, at=3))
    assert decisions[0].job_id == "W1"


def test_equal_submission_times_resolve_deterministically_by_queue_order():
    first = contest(job("FIRST", 30, at=2), job("SECOND", 30, at=2))[1][0].job_id
    again = contest(job("FIRST", 30, at=2), job("SECOND", 30, at=2))[1][0].job_id
    assert first == again == "FIRST"


# ---- SJF / weighted mode -----------------------------------------------------

def test_slightly_above_20_percent_switches_to_score_based():
    _, decisions = contest(job("BIG", 100, at=1), job("SMALL", 79, at=2))
    assert decisions[0].policy == AllocationPolicy.SCORE_BASED
    assert decisions[0].job_id == "SMALL"  # equal priority -> the smaller job's size term wins


def test_very_different_sizes_smaller_job_wins_at_equal_priority():
    _, decisions = contest(job("HUGE", 600, at=1), job("TINY", 5, at=2), job("MID", 60, at=3))
    assert decisions[0].policy == AllocationPolicy.SCORE_BASED
    assert decisions[0].job_id == "TINY"


def test_different_size_and_priority_blend_priority_can_beat_a_smaller_job():
    """README example: HIGH/20min beats MEDIUM/15min (the *smaller*
    job) because priority carries 60% - not pure SJF."""
    _, decisions = contest(job("J1", 20, Priority.HIGH, at=1), job("J2", 15, Priority.MEDIUM, at=2),
                           job("J3", 200, Priority.LOW, at=3))
    assert decisions[0].policy == AllocationPolicy.SCORE_BASED
    assert decisions[0].job_id == "J1"


def test_blended_policy_is_preserved_a_much_smaller_lower_priority_job_can_win():
    """The retained blended-score decision (not a HIGH > MEDIUM > LOW
    gate): MEDIUM/15min outscores HIGH/180min."""
    _, decisions = contest(job("HIGH-BIG", 180, Priority.HIGH, at=1), job("MED-SMALL", 15, Priority.MEDIUM, at=2))
    assert decisions[0].policy == AllocationPolicy.SCORE_BASED
    assert decisions[0].job_id == "MED-SMALL"


def test_three_job_competition_is_decided_by_score():
    _, decisions = contest(job("A", 15, at=1), job("B", 60, at=2), job("C", 180, at=3))
    assert decisions[0].policy == AllocationPolicy.SCORE_BASED
    assert [c.job_id for c in sorted(decisions[0].candidates, key=lambda c: -c.score)] == ["A", "B", "C"]
    assert decisions[0].job_id == "A"


def test_equal_scores_break_ties_by_earliest_submission():
    # A and B identical priority+size (identical score); C's size forces score mode.
    _, decisions = contest(job("A", 10, at=8), job("B", 10, at=2), job("C", 1000, at=9))
    assert decisions[0].policy == AllocationPolicy.SCORE_BASED
    assert decisions[0].job_id == "B"


# ---- aging coexists with the policy ---------------------------------------------

def test_aging_lets_a_long_waiting_low_priority_job_win_in_score_mode():
    old = job("OLD-LOW-BIG", 500, Priority.LOW, at=-300)      # waited ~5h
    fresh = job("FRESH-HIGH-SMALL", 5, Priority.HIGH, at=10)  # not CRITICAL: that tier would gate everyone else out
    _, decisions = contest(old, fresh, decide_at_minutes=10)
    decision = decisions[0]
    assert decision.policy == AllocationPolicy.SCORE_BASED
    assert decision.job_id == "OLD-LOW-BIG"
    winner = next(c for c in decision.candidates if c.job_id == decision.job_id)
    loser = next(c for c in decision.candidates if c.job_id != decision.job_id)
    assert winner.aging_component > loser.aging_component
    assert winner.base_score < loser.base_score  # it won on aging, not on base score


def test_aging_grows_with_waiting_time_and_is_not_a_second_formula():
    _, short = contest(job("A", 10, at=0), job("B", 500, at=0), decide_at_minutes=5)
    _, long = contest(job("A", 10, at=0), job("B", 500, at=0), decide_at_minutes=60)
    a_short = next(c for c in short[0].candidates if c.job_id == "B")
    a_long = next(c for c in long[0].candidates if c.job_id == "B")
    assert 0 < a_short.aging_component < a_long.aging_component
    assert a_long.aging_component == pytest.approx(60 * 0.01)  # the one existing rate


def test_fcfs_mode_needs_no_aging_it_already_serves_the_longest_waiter():
    _, decisions = contest(job("A", 30, at=-200), job("B", 30, at=0), decide_at_minutes=10)
    assert decisions[0].policy == AllocationPolicy.FCFS
    assert decisions[0].job_id == "A"
    assert all(c.aging_component is None for c in decisions[0].candidates)  # score never computed


# ---- queue synchronization: cancel / requeue -------------------------------------------

def test_cancelled_job_is_skipped_and_the_next_earliest_wins():
    scheduler = scheduler_with(1)
    enqueue(scheduler, job("HOLDER", 30, at=-1000))
    scheduler.try_allocate_all(now=NOW - timedelta(minutes=999))
    enqueue(scheduler, job("A", 30, at=1), job("B", 30, at=2), job("C", 30, at=3))
    scheduler.cancel_job("A", now=NOW + timedelta(minutes=5))
    scheduler.complete_job("HOLDER", now=NOW + timedelta(minutes=6))
    decision = scheduler.try_allocate_all(now=NOW + timedelta(minutes=6))[0]
    assert decision.job_id == "B"
    assert "A" not in {c.job_id for c in decision.candidates}


def test_requeued_job_keeps_its_original_arrival_position():
    scheduler = scheduler_with(1)
    enqueue(scheduler, job("A", 30, at=1))
    scheduler.try_allocate_all(now=NOW + timedelta(minutes=2))       # A runs on GPU-1
    enqueue(scheduler, job("B", 30, at=3), job("C", 30, at=4))
    scheduler.handle_gpu_failure("GPU-1", now=NOW + timedelta(minutes=5))  # A requeued
    scheduler.add_gpu(GPU(gpu_id="GPU-2", total_memory_mb=1000, status=GPUStatus.IDLE))
    decision = scheduler.try_allocate_all(now=NOW + timedelta(minutes=6))[0]
    assert decision.policy == AllocationPolicy.FCFS
    assert decision.job_id == "A"  # requeued to the queue's tail, but FCFS reads submitted_at
    assert next(c for c in decision.candidates if c.job_id == "A").arrival_position == 1


# ---- multi-GPU / partial / no GPUs / many jobs -----------------------------------------------

def test_multi_gpu_job_keeps_winning_until_fully_allocated_under_fcfs():
    scheduler = scheduler_with(2)
    a = job("A", 30, at=1, gpu_count=2)
    b = job("B", 30, at=2)
    enqueue(scheduler, a, b)
    decisions = scheduler.try_allocate_all(now=NOW + timedelta(minutes=3))
    assert [d.job_id for d in decisions] == ["A", "A"]
    assert a.status == JobStatus.RUNNING and a.gpus_still_needed == 0
    assert b.status == JobStatus.WAITING


def test_partial_availability_earlier_job_holds_its_place_later_job_does_not_jump_it():
    scheduler = scheduler_with(1)
    a = job("A", 30, at=1, gpu_count=3)
    b = job("B", 30, at=2)
    enqueue(scheduler, a, b)
    scheduler.try_allocate_all(now=NOW + timedelta(minutes=3))
    assert (len(a.assigned_gpu_ids), a.gpus_still_needed) == (1, 2)
    assert a.status == JobStatus.WAITING
    assert b.assigned_gpu_ids == []  # FCFS: A is still the earliest waiter
    assert scheduler.allocation_engine.waiting_job_ids() == ["A", "B"]


def test_no_available_gpus_produces_no_decision_and_no_crash():
    scheduler = scheduler_with(0)
    enqueue(scheduler, job("A", 10, at=1), job("B", 500, at=2))
    assert scheduler.try_allocate_all(now=NOW + timedelta(minutes=3)) == []
    assert scheduler.allocation_engine.select_next_job() is not None  # still decidable, just nowhere to go


def test_many_similar_jobs_are_served_strictly_in_arrival_order():
    scheduler = scheduler_with(3)
    jobs = [job(f"J{i:03d}", 100 + (i % 10), at=i) for i in range(100)]  # sizes 100..109 (<10% spread)
    enqueue(scheduler, *jobs)
    decisions = scheduler.try_allocate_all(now=NOW + timedelta(minutes=200))
    assert [d.job_id for d in decisions] == ["J000", "J001", "J002"]
    assert all(d.policy == AllocationPolicy.FCFS for d in decisions)


def test_many_varied_jobs_decisions_are_deterministic_across_runs():
    def run():
        scheduler = scheduler_with(5)
        jobs = [job(f"J{i:03d}", 5 + (i * 37) % 400, Priority((i % 4) + 1), at=i) for i in range(120)]
        enqueue(scheduler, *jobs)
        return [d.job_id for d in scheduler.try_allocate_all(now=NOW + timedelta(minutes=30))]

    assert run() == run()
    assert len(set(run())) == 5


# ---- decision trace ----------------------------------------------------------------------------------

def test_fcfs_trace_explains_mode_spread_arrival_order_and_winner():
    _, decisions = contest(job("A", 10, at=1), job("B", 12, at=2))
    decision = decisions[0]
    assert decision.similarity_threshold == JOB_SIZE_SIMILARITY_THRESHOLD
    assert decision.size_spread == pytest.approx((12 - 10) / 12)
    assert [(c.job_id, c.arrival_position) for c in sorted(decision.candidates, key=lambda c: c.arrival_position)] == [
        ("A", 1), ("B", 2)]

    text = decision.explain()
    assert "Policy: FCFS" in text
    assert "within the 20% threshold" in text
    assert "job sizes within 20%" in text and "has waited longest" in text
    assert "#1 A" in text and "#2 B" in text
    assert "size=10min" in text and "MEDIUM" in text and "waited=" in text
    assert "Selected: A -> GPU-1" in text


def test_score_based_trace_explains_mode_breakdown_and_winner():
    _, decisions = contest(job("A", 15, Priority.MEDIUM, at=1), job("B", 60, Priority.MEDIUM, at=2))
    decision = decisions[0]
    assert decision.size_spread == pytest.approx(0.75)

    text = decision.explain()
    assert "Policy: SJF / Weighted" in text
    assert "beyond the 20% threshold" in text
    assert "scored highest" in text
    assert "priority component" in text and "size component" in text and "aging" in text
    assert "Selected: A -> GPU-1" in text
    for c in decision.candidates:
        assert c.score == pytest.approx(c.base_score + c.aging_component)


def test_trace_is_available_on_the_schedulers_last_decision():
    scheduler, decisions = contest(job("A", 10, at=1), job("B", 12, at=2))
    assert scheduler.last_allocation_decision is decisions[-1]
    assert "Policy: FCFS" in scheduler.last_allocation_decision.explain()


def test_trace_waiting_time_uses_the_decisions_own_clock_not_the_wall_clock():
    """Regression: `Job.waiting_time` measures against the wall clock
    for a not-yet-started job, so a simulated-clock decision used to
    report e.g. "waited 261 days". The candidate's waiting time must
    be measured against the decision's own `now`."""
    _, decisions = contest(job("A", 15, at=1), job("B", 60, at=2), decide_at_minutes=10)
    by_id = {c.job_id: c for c in decisions[0].candidates}
    assert by_id["A"].waiting_time == timedelta(minutes=9)
    assert by_id["B"].waiting_time == timedelta(minutes=8)
    assert by_id["A"].waiting_minutes == pytest.approx(9.0)
