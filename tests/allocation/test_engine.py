from datetime import datetime, timedelta, timezone

import pytest

from engine.allocation.decision import AllocationPolicy
from engine.allocation.engine import AllocationEngine
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState
from engine.models.user import User

BASE = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


def make_engine() -> AllocationEngine:
    return AllocationEngine(SchedulerState())


def make_gpu(gpu_id: str) -> GPU:
    return GPU(gpu_id=gpu_id, total_memory_mb=24_576)


def make_job(job_id: str, user_id: str, size_minutes: float, priority: Priority = Priority.MEDIUM,
            submitted_at: datetime = BASE) -> Job:
    return Job(job_id=job_id, user_id=user_id, name=job_id, priority=priority,
               estimated_size_minutes=size_minutes, submitted_at=submitted_at)


def add_user(engine: AllocationEngine, user_id: str, priority: Priority = Priority.MEDIUM) -> User:
    user = User(user_id=user_id, name=user_id, priority=priority)
    engine.state.add_user(user)
    return user


# ------------------------------------------------------------------
# Scenario 1 - simple allocation
# ------------------------------------------------------------------

def test_scenario_1_single_gpu_single_job():
    engine = make_engine()
    add_user(engine, "U1")
    engine.add_gpu(make_gpu("GPU0"))
    engine.submit_job(make_job("J1", "U1", size_minutes=30))

    decision = engine.allocate_next()

    assert decision is not None
    assert decision.gpu_id == "GPU0"
    assert decision.job_id == "J1"
    assert decision.user_id == "U1"
    assert decision.policy == AllocationPolicy.FCFS  # only one candidate


# ------------------------------------------------------------------
# Scenario 2 - priority difference, similar sizes
# ------------------------------------------------------------------

def test_scenario_2_priority_difference_with_similar_sizes_uses_fcfs():
    # Similar sizes -> FCFS applies regardless of priority; the
    # project's rule is size-similarity-first, not priority-first,
    # for choosing FCFS vs score-based.
    engine = make_engine()
    add_user(engine, "U1")
    add_user(engine, "U2")
    engine.add_gpu(make_gpu("GPU0"))

    high = make_job("J_high", "U1", size_minutes=20, priority=Priority.HIGH, submitted_at=BASE + timedelta(minutes=5))
    low = make_job("J_low", "U2", size_minutes=21, priority=Priority.LOW, submitted_at=BASE)
    engine.submit_job(high)
    engine.submit_job(low)

    decision = engine.allocate_next()

    assert decision.policy == AllocationPolicy.FCFS
    assert decision.job_id == "J_low"  # submitted first, waited longest


# ------------------------------------------------------------------
# Scenario 3 - FCFS among similar sizes
# ------------------------------------------------------------------

def test_scenario_3_fcfs_selects_longest_waiting_job():
    engine = make_engine()
    for uid in ("U1", "U2", "U3"):
        add_user(engine, uid)
    engine.add_gpu(make_gpu("GPU0"))

    job_a = make_job("A", "U1", size_minutes=10, submitted_at=BASE)
    job_b = make_job("B", "U2", size_minutes=12, submitted_at=BASE + timedelta(minutes=3))
    job_c = make_job("C", "U3", size_minutes=11, submitted_at=BASE + timedelta(minutes=5))
    for job in (job_a, job_b, job_c):
        engine.submit_job(job)

    decision = engine.allocate_next()

    assert decision.policy == AllocationPolicy.FCFS
    assert decision.job_id == "A"
    assert len(decision.candidates) == 3
    assert all(c.score is None for c in decision.candidates)


# ------------------------------------------------------------------
# Scenario 4 - clearly different job sizes -> score-based
# ------------------------------------------------------------------

def test_scenario_4_different_sizes_use_score_based_selection():
    engine = make_engine()
    for uid in ("U1", "U2", "U3"):
        add_user(engine, uid)
    engine.add_gpu(make_gpu("GPU0"))

    small = make_job("SMALL", "U1", size_minutes=15, priority=Priority.MEDIUM, submitted_at=BASE)
    medium = make_job("MEDIUM", "U2", size_minutes=60, priority=Priority.MEDIUM, submitted_at=BASE)
    large = make_job("LARGE", "U3", size_minutes=180, priority=Priority.MEDIUM, submitted_at=BASE)
    for job in (small, medium, large):
        engine.submit_job(job)

    decision = engine.allocate_next()

    assert decision.policy == AllocationPolicy.SCORE_BASED
    # Equal priority -> smallest job has the highest size component and wins.
    assert decision.job_id == "SMALL"
    assert all(c.score is not None for c in decision.candidates)


# ------------------------------------------------------------------
# Scenario 5 - three competing jobs, explicit score check
# ------------------------------------------------------------------

def test_scenario_5_three_competing_jobs_score_matches_formula():
    engine = make_engine()
    for uid in ("U1", "U2", "U3"):
        add_user(engine, uid)
    engine.add_gpu(make_gpu("GPU0"))

    j1 = make_job("J1", "U1", size_minutes=20, priority=Priority.HIGH, submitted_at=BASE)
    j2 = make_job("J2", "U2", size_minutes=15, priority=Priority.MEDIUM, submitted_at=BASE + timedelta(minutes=1))
    j3 = make_job("J3", "U3", size_minutes=200, priority=Priority.LOW, submitted_at=BASE + timedelta(minutes=2))
    for job in (j1, j2, j3):
        engine.submit_job(job)

    decision = engine.allocate_next()

    assert decision.policy == AllocationPolicy.SCORE_BASED
    scores = {c.job_id: c.score for c in decision.candidates}

    # min=15, max=200
    expected_j1 = 0.6 * (2 / 3) + 0.4 * ((200 - 20) / (200 - 15))
    expected_j2 = 0.6 * (1 / 3) + 0.4 * ((200 - 15) / (200 - 15))
    expected_j3 = 0.6 * 0.0 + 0.4 * ((200 - 200) / (200 - 15))

    assert scores["J1"] == pytest.approx(expected_j1)
    assert scores["J2"] == pytest.approx(expected_j2)
    assert scores["J3"] == pytest.approx(expected_j3)
    assert decision.job_id == max(scores, key=scores.get)


# ------------------------------------------------------------------
# Scenario 6 - critical job precedence
# ------------------------------------------------------------------

def test_scenario_6_critical_job_takes_precedence_over_larger_normal_jobs():
    engine = make_engine()
    for uid in ("U1", "U2"):
        add_user(engine, uid)
    engine.add_gpu(make_gpu("GPU0"))

    # Without the critical job, this HIGH-priority tiny job would win outright.
    normal_high = make_job("NORMAL_HIGH", "U1", size_minutes=5, priority=Priority.HIGH, submitted_at=BASE)
    critical = make_job("CRITICAL", "U2", size_minutes=500, priority=Priority.CRITICAL,
                         submitted_at=BASE + timedelta(minutes=10))
    engine.submit_job(normal_high)
    engine.submit_job(critical)

    decision = engine.allocate_next()

    assert decision.job_id == "CRITICAL"
    # Only the critical job was scored - the normal job never entered the comparison.
    assert [c.job_id for c in decision.candidates] == ["CRITICAL"]
    assert "critical" in decision.reason.lower()


def test_scenario_6b_multiple_critical_jobs_still_apply_similarity_rule():
    engine = make_engine()
    for uid in ("U1", "U2"):
        add_user(engine, uid)
    engine.add_gpu(make_gpu("GPU0"))

    crit_a = make_job("CRIT_A", "U1", size_minutes=10, priority=Priority.CRITICAL, submitted_at=BASE)
    crit_b = make_job("CRIT_B", "U2", size_minutes=11, priority=Priority.CRITICAL,
                       submitted_at=BASE + timedelta(minutes=1))
    normal = make_job("NORMAL", "U1", size_minutes=1, priority=Priority.HIGH,
                       submitted_at=BASE - timedelta(minutes=5))
    for job in (normal, crit_a, crit_b):
        engine.submit_job(job)

    decision = engine.allocate_next()

    # Both critical jobs are similar in size -> FCFS between them; the
    # normal job (even though it arrived first overall) is excluded.
    assert decision.policy == AllocationPolicy.FCFS
    assert decision.job_id == "CRIT_A"
    assert {c.job_id for c in decision.candidates} == {"CRIT_A", "CRIT_B"}


# ------------------------------------------------------------------
# Scenario 7 - multiple GPUs, repeated allocation
# ------------------------------------------------------------------

def test_scenario_7_allocates_until_gpus_or_jobs_run_out():
    engine = make_engine()
    for uid in ("U1", "U2", "U3"):
        add_user(engine, uid)
    for gid in ("GPU0", "GPU1"):
        engine.add_gpu(make_gpu(gid))

    engine.submit_job(make_job("J1", "U1", size_minutes=10, submitted_at=BASE))
    engine.submit_job(make_job("J2", "U2", size_minutes=10, submitted_at=BASE + timedelta(minutes=1)))
    engine.submit_job(make_job("J3", "U3", size_minutes=10, submitted_at=BASE + timedelta(minutes=2)))

    decisions = engine.allocate_all()

    # Two GPUs, three jobs -> exactly two allocations, one job left waiting.
    assert len(decisions) == 2
    assert {d.gpu_id for d in decisions} == {"GPU0", "GPU1"}
    assert engine.waiting_job_ids() == ["J3"]
    assert engine.available_gpu_count() == 0


def test_scenario_7b_more_gpus_than_jobs_leaves_gpus_available():
    engine = make_engine()
    add_user(engine, "U1")
    for gid in ("GPU0", "GPU1", "GPU2"):
        engine.add_gpu(make_gpu(gid))
    engine.submit_job(make_job("J1", "U1", size_minutes=10))

    decisions = engine.allocate_all()

    assert len(decisions) == 1
    assert engine.available_gpu_count() == 2
    assert engine.waiting_job_ids() == []


# ------------------------------------------------------------------
# State consistency after assignment
# ------------------------------------------------------------------

def test_assignment_keeps_gpu_job_user_and_index_consistent():
    engine = make_engine()
    add_user(engine, "U1")
    engine.add_gpu(make_gpu("GPU0"))
    engine.submit_job(make_job("J1", "U1", size_minutes=10))

    decision = engine.allocate_next()

    gpu = engine.state.get_gpu("GPU0")
    job = engine.state.get_job("J1")
    user = engine.state.get_user("U1")

    assert gpu.assigned_user_id == "U1"
    assert gpu.assigned_job_id == "J1"
    assert gpu.status == GPUStatus.ACTIVE

    assert job.status == JobStatus.RUNNING
    assert job.assigned_gpu_id == "GPU0"
    assert job.started_at is not None

    assert "GPU0" in user.assigned_gpu_ids
    assert "J1" in user.running_job_ids

    assert engine.get_gpus_for_user("U1") == ["GPU0"]

    assignment = engine.state.get_active_assignment_for_gpu("GPU0")
    assert assignment is not None
    assert assignment.job_id == "J1"
    assert assignment.user_id == "U1"

    assert decision.event.event_type == EventType.ALLOC
    assert decision.event in engine.state.events


def test_job_leaves_waiting_queue_once_assigned():
    engine = make_engine()
    add_user(engine, "U1")
    engine.add_gpu(make_gpu("GPU0"))
    engine.submit_job(make_job("J1", "U1", size_minutes=10))

    assert engine.waiting_job_ids() == ["J1"]
    engine.allocate_next()
    assert engine.waiting_job_ids() == []


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_no_available_gpus_returns_none():
    engine = make_engine()
    add_user(engine, "U1")
    engine.submit_job(make_job("J1", "U1", size_minutes=10))

    assert engine.allocate_next() is None
    assert engine.waiting_job_ids() == ["J1"]


def test_no_waiting_jobs_returns_none_and_gpu_stays_available():
    engine = make_engine()
    engine.add_gpu(make_gpu("GPU0"))

    assert engine.allocate_next() is None
    assert engine.available_gpu_count() == 1


def test_duplicate_job_submission_raises():
    engine = make_engine()
    add_user(engine, "U1")
    engine.submit_job(make_job("J1", "U1", size_minutes=10))

    with pytest.raises(ValueError):
        engine.submit_job(make_job("J1", "U1", size_minutes=20))


def test_submitting_an_already_running_job_raises():
    engine = make_engine()
    add_user(engine, "U1")
    running_job = make_job("J1", "U1", size_minutes=10)
    running_job.status = JobStatus.RUNNING

    with pytest.raises(ValueError):
        engine.submit_job(running_job)


def test_gpu_that_is_already_assigned_is_not_immediately_available():
    engine = make_engine()
    add_user(engine, "U1")
    gpu = make_gpu("GPU0")
    gpu.assigned_user_id = "U1"
    gpu.assigned_job_id = "SOME_JOB"

    engine.add_gpu(gpu)

    assert engine.available_gpu_count() == 0


def test_mark_gpu_available_rejects_unknown_or_still_assigned_gpu():
    engine = make_engine()
    gpu = make_gpu("GPU0")
    gpu.assigned_user_id = "U1"
    engine.add_gpu(gpu)  # already assigned -> not auto-available

    assert engine.mark_gpu_available("GPU0") is False
    assert engine.mark_gpu_available("does-not-exist") is False

    gpu.assigned_user_id = None
    assert engine.mark_gpu_available("GPU0") is True
    assert engine.available_gpu_count() == 1


def test_zero_and_negative_job_sizes_do_not_crash_allocation():
    engine = make_engine()
    for uid in ("U1", "U2"):
        add_user(engine, uid)
    engine.add_gpu(make_gpu("GPU0"))

    engine.submit_job(make_job("ZERO", "U1", size_minutes=0, submitted_at=BASE))
    engine.submit_job(make_job("BIG", "U2", size_minutes=500, submitted_at=BASE + timedelta(minutes=1)))

    decision = engine.allocate_next()

    assert decision is not None
    assert decision.job_id in ("ZERO", "BIG")


def test_removed_gpu_is_not_handed_out():
    engine = make_engine()
    add_user(engine, "U1")
    engine.add_gpu(make_gpu("GPU0"))
    engine.add_gpu(make_gpu("GPU1"))
    engine.remove_gpu("GPU0")
    engine.submit_job(make_job("J1", "U1", size_minutes=10))

    decision = engine.allocate_next()

    assert decision.gpu_id == "GPU1"
    assert engine.state.get_gpu("GPU0") is None


def test_equal_priority_and_equal_size_jobs_still_resolve_deterministically():
    engine = make_engine()
    for uid in ("U1", "U2"):
        add_user(engine, uid)
    engine.add_gpu(make_gpu("GPU0"))

    job_a = make_job("A", "U1", size_minutes=10, submitted_at=BASE)
    job_b = make_job("B", "U2", size_minutes=10, submitted_at=BASE + timedelta(seconds=1))
    engine.submit_job(job_a)
    engine.submit_job(job_b)

    decision = engine.allocate_next()

    # Equal sizes -> similarity rule applies -> FCFS -> earliest arrival wins.
    assert decision.policy == AllocationPolicy.FCFS
    assert decision.job_id == "A"


def test_user_can_hold_multiple_jobs_gpus_across_allocations():
    engine = make_engine()
    add_user(engine, "U1")
    engine.add_gpu(make_gpu("GPU0"))
    engine.add_gpu(make_gpu("GPU1"))
    engine.submit_job(make_job("J1", "U1", size_minutes=10, submitted_at=BASE))
    engine.submit_job(make_job("J2", "U1", size_minutes=10, submitted_at=BASE + timedelta(minutes=1)))

    engine.allocate_all()

    assert sorted(engine.get_gpus_for_user("U1")) == ["GPU0", "GPU1"]
    user = engine.state.get_user("U1")
    assert sorted(user.running_job_ids) == ["J1", "J2"]


# ------------------------------------------------------------------
# Audit regression: the score-based winner must actually be decided
# by the Phase 2 PriorityQueue/MaxHeap, not a plain Python min()/max()
# standing in for it. (A prior audit found MaxHeap/PriorityQueue were
# fully implemented and unit-tested but never imported by any engine.)
# ------------------------------------------------------------------

def test_score_based_selection_genuinely_uses_the_priority_queue(monkeypatch):
    import engine.allocation.engine as allocation_engine_module

    calls = {"insert": 0, "pop_best": 0}
    real_priority_queue_cls = allocation_engine_module.PriorityQueue

    class SpyingPriorityQueue(real_priority_queue_cls):
        def insert(self, item):
            calls["insert"] += 1
            return super().insert(item)

        def pop_best(self):
            calls["pop_best"] += 1
            return super().pop_best()

    monkeypatch.setattr(allocation_engine_module, "PriorityQueue", SpyingPriorityQueue)

    engine = make_engine()
    for uid in ("U1", "U2", "U3"):
        add_user(engine, uid)
    engine.add_gpu(make_gpu("GPU0"))

    # Sizes well beyond the 20% threshold -> forces the score-based branch.
    engine.submit_job(make_job("SMALL", "U1", size_minutes=15, priority=Priority.MEDIUM, submitted_at=BASE))
    engine.submit_job(make_job("MEDIUM", "U2", size_minutes=60, priority=Priority.MEDIUM, submitted_at=BASE))
    engine.submit_job(make_job("LARGE", "U3", size_minutes=180, priority=Priority.MEDIUM, submitted_at=BASE))

    decision = engine.allocate_next()

    assert decision.policy == AllocationPolicy.SCORE_BASED
    assert decision.job_id == "SMALL"
    # The real PriorityQueue/MaxHeap machinery was actually exercised -
    # one insert per candidate, one pop for the winner - not bypassed
    # by a plain min()/max() over a Python list.
    assert calls["insert"] == 3
    assert calls["pop_best"] == 1
