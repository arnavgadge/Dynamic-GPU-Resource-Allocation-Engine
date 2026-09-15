from datetime import datetime, timedelta, timezone

import pytest

from engine.allocation.engine import AllocationEngine
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState
from engine.models.user import User
from engine.models.utilization import UtilizationObservation
from engine.reclamation.decision import ReclamationAction
from engine.reclamation.engine import ReclamationEngine
from engine.reclamation.policy import (
    ConfirmationResponse,
    ReclamationPolicy,
    ReclamationTier,
    ReclamationTierPolicy,
)

BASE = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

# A fast policy for tests: same shape as the real one, tiny durations.
TEST_POLICY = ReclamationPolicy(
    tier1=ReclamationTierPolicy(
        tier=ReclamationTier.TIER_1, label="likely completed job",
        utilization_threshold_percent=2.0, sustained_duration=timedelta(minutes=20),
    ),
    tier2=ReclamationTierPolicy(
        tier=ReclamationTier.TIER_2, label="likely inactive user",
        utilization_threshold_percent=15.0, sustained_duration=timedelta(minutes=60),
    ),
    no_response_grace_period=timedelta(minutes=5),
)


def build_state_with_running_job():
    state = SchedulerState()
    state.add_user(User(user_id="U1", name="Alice", priority=Priority.HIGH))
    gpu = GPU(gpu_id="GPU0", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
              assigned_user_id="U1", assigned_job_id="J1")
    job = Job(job_id="J1", user_id="U1", name="ML Training", priority=Priority.HIGH,
              estimated_size_minutes=180, status=JobStatus.RUNNING,
              started_at=BASE, assigned_gpu_ids=["GPU0"])
    state.add_gpu(gpu)
    state.add_job(job)
    user = state.get_user("U1")
    user.assigned_gpu_ids.append("GPU0")
    user.running_job_ids.append("J1")
    return state, gpu, job, user


def feed_low_utilization(engine: ReclamationEngine, gpu: GPU, minutes: int, percent: float, start_minute: int = 0):
    last = None
    for minute in range(start_minute, start_minute + minutes + 1):
        observation = UtilizationObservation(utilization_percent=percent, timestamp=BASE + timedelta(minutes=minute))
        last = engine.record_utilization(gpu, observation)
    return last


# ------------------------------------------------------------------
# Sustained vs. brief dips
# ------------------------------------------------------------------

def test_brief_dip_does_not_trigger_a_prompt():
    state, gpu, job, user = build_state_with_running_job()
    engine = ReclamationEngine(state, policy=TEST_POLICY)

    readings = [10.0, 9.0, 4.0, 1.0, 8.0, 14.0, 20.0, 30.0]
    decision = None
    for minute, percent in enumerate(readings):
        observation = UtilizationObservation(utilization_percent=percent, timestamp=BASE + timedelta(minutes=minute))
        decision = engine.record_utilization(gpu, observation)

    assert decision is None
    assert gpu.status == GPUStatus.ACTIVE
    assert engine.has_pending_prompt("GPU0") is False


def test_sustained_tier1_breach_raises_a_prompt():
    state, gpu, job, user = build_state_with_running_job()
    engine = ReclamationEngine(state, policy=TEST_POLICY)

    decision = feed_low_utilization(engine, gpu, minutes=20, percent=1.0)

    assert decision is not None
    assert decision.action == ReclamationAction.PROMPTED
    assert decision.tier == ReclamationTier.TIER_1
    assert gpu.status == GPUStatus.IDLE_WARNING
    assert engine.has_pending_prompt("GPU0") is True
    assert decision.event.event_type == EventType.PROMPT
    assert decision.event in state.events


def test_sustained_tier2_breach_raises_a_prompt_when_above_tier1_threshold():
    state, gpu, job, user = build_state_with_running_job()
    engine = ReclamationEngine(state, policy=TEST_POLICY)

    # 10% is above Tier 1's 2% threshold, so only Tier 2 can fire.
    decision = feed_low_utilization(engine, gpu, minutes=60, percent=10.0)

    assert decision is not None
    assert decision.tier == ReclamationTier.TIER_2


def test_no_prompt_while_utilization_stays_healthy():
    state, gpu, job, user = build_state_with_running_job()
    engine = ReclamationEngine(state, policy=TEST_POLICY)

    decision = feed_low_utilization(engine, gpu, minutes=60, percent=80.0)

    assert decision is None
    assert gpu.status == GPUStatus.ACTIVE


# ------------------------------------------------------------------
# Confirmation flow: YES / NO / no response
# ------------------------------------------------------------------

def test_yes_response_backs_off_and_resets_the_timer():
    state, gpu, job, user = build_state_with_running_job()
    engine = ReclamationEngine(state, policy=TEST_POLICY)
    feed_low_utilization(engine, gpu, minutes=20, percent=1.0)
    assert engine.has_pending_prompt("GPU0") is True

    decision = engine.respond("GPU0", ConfirmationResponse.YES, now=BASE + timedelta(minutes=20))

    assert decision.action == ReclamationAction.BACKED_OFF
    assert gpu.status == GPUStatus.ACTIVE
    assert engine.has_pending_prompt("GPU0") is False
    # GPU/job/user assignment is untouched by a "yes".
    assert gpu.assigned_job_id == "J1"
    assert job.status == JobStatus.RUNNING


def test_after_yes_a_new_full_duration_breach_is_required_before_re_prompting():
    state, gpu, job, user = build_state_with_running_job()
    engine = ReclamationEngine(state, policy=TEST_POLICY)
    feed_low_utilization(engine, gpu, minutes=20, percent=1.0)
    engine.respond("GPU0", ConfirmationResponse.YES, now=BASE + timedelta(minutes=20))

    # Immediately after backing off, a few more low readings should
    # NOT instantly re-trigger - the timer was reset.
    decision = feed_low_utilization(engine, gpu, minutes=5, percent=1.0, start_minute=21)
    assert decision is None
    assert gpu.status == GPUStatus.ACTIVE

    # But a full new sustained window after the reset does re-trigger.
    decision = feed_low_utilization(engine, gpu, minutes=20, percent=1.0, start_minute=21)
    assert decision is not None
    assert decision.action == ReclamationAction.PROMPTED


def test_no_response_reclaims_immediately_and_updates_state():
    state, gpu, job, user = build_state_with_running_job()
    allocation_engine = AllocationEngine(state)
    allocation_engine.add_gpu(gpu)  # already assigned -> not immediately available
    engine = ReclamationEngine(state, allocation_engine=allocation_engine, policy=TEST_POLICY)

    feed_low_utilization(engine, gpu, minutes=20, percent=1.0)
    decision = engine.respond("GPU0", ConfirmationResponse.NO, now=BASE + timedelta(minutes=20))

    assert decision.action == ReclamationAction.RECLAIMED
    assert gpu.status == GPUStatus.IDLE
    assert gpu.assigned_user_id is None
    assert gpu.assigned_job_id is None
    assert job.status == JobStatus.RECLAIMED
    assert job.assigned_gpu_id is None
    assert "GPU0" not in user.assigned_gpu_ids
    assert "J1" not in user.running_job_ids
    assert decision.event.event_type == EventType.RECLAIM

    assignment = state.get_active_assignment_for_gpu("GPU0")
    assert assignment is None  # ended, not active

    # The reclaimed GPU goes back into the allocation engine's pool.
    assert allocation_engine.available_gpu_count() == 1
    assert engine.last_reclaim() is decision.event


def test_no_response_within_grace_period_does_not_auto_reclaim():
    state, gpu, job, user = build_state_with_running_job()
    engine = ReclamationEngine(state, policy=TEST_POLICY)
    feed_low_utilization(engine, gpu, minutes=20, percent=1.0)

    decisions = engine.check_timeouts(now=BASE + timedelta(minutes=22))  # grace period is 5 min

    assert decisions == []
    assert engine.has_pending_prompt("GPU0") is True
    assert gpu.status == GPUStatus.IDLE_WARNING


def test_no_response_past_grace_period_auto_reclaims():
    state, gpu, job, user = build_state_with_running_job()
    allocation_engine = AllocationEngine(state)
    allocation_engine.add_gpu(gpu)
    engine = ReclamationEngine(state, allocation_engine=allocation_engine, policy=TEST_POLICY)

    feed_low_utilization(engine, gpu, minutes=20, percent=1.0)
    assert engine.has_pending_prompt("GPU0") is True

    decisions = engine.check_timeouts(now=BASE + timedelta(minutes=26))  # 6 min after prompt > 5 min grace

    assert len(decisions) == 1
    assert decisions[0].action == ReclamationAction.RECLAIMED
    assert "no response" in decisions[0].reason.lower()
    assert gpu.status == GPUStatus.IDLE
    assert allocation_engine.available_gpu_count() == 1


def test_responding_with_no_pending_prompt_raises():
    state, gpu, job, user = build_state_with_running_job()
    engine = ReclamationEngine(state, policy=TEST_POLICY)

    with pytest.raises(ValueError):
        engine.respond("GPU0", ConfirmationResponse.YES)


def test_new_utilization_readings_do_not_raise_a_second_prompt_while_one_is_pending():
    state, gpu, job, user = build_state_with_running_job()
    engine = ReclamationEngine(state, policy=TEST_POLICY)
    feed_low_utilization(engine, gpu, minutes=20, percent=1.0)
    assert engine.has_pending_prompt("GPU0") is True

    decision = engine.record_utilization(
        gpu, UtilizationObservation(utilization_percent=1.0, timestamp=BASE + timedelta(minutes=21))
    )

    assert decision is None  # still waiting on the same prompt, not a new one


def test_unassigned_gpu_is_never_prompted():
    state = SchedulerState()
    gpu = GPU(gpu_id="GPU0", total_memory_mb=24_576, status=GPUStatus.IDLE)
    state.add_gpu(gpu)
    engine = ReclamationEngine(state, policy=TEST_POLICY)

    decision = feed_low_utilization(engine, gpu, minutes=60, percent=0.0)

    assert decision is None
    assert gpu.status == GPUStatus.IDLE


# ------------------------------------------------------------------
# Audit regression: a reclaim must also clear the stale user->GPU
# entry in AllocationEngine's HashMap-backed index (UserGPUIndex),
# not just the real GPU/User model fields. A prior audit found the
# HashMap kept reporting a GPU the user no longer held after a
# legitimate reclaim, because ReclamationEngine never told
# AllocationEngine's index about it.
# ------------------------------------------------------------------

def test_reclaim_clears_the_allocation_engines_stale_hashmap_entry_too():
    from engine.models.assignment import GPUAssignment

    state, gpu, job, user = build_state_with_running_job()
    allocation_engine = AllocationEngine(state)
    allocation_engine.add_gpu(gpu)
    allocation_engine._user_index.add_assignment(
        GPUAssignment(assignment_id="seed", gpu_id="GPU0", user_id="U1", job_id="J1", created_at=BASE)
    )
    assert allocation_engine.get_gpus_for_user("U1") == ["GPU0"]

    engine = ReclamationEngine(state, allocation_engine=allocation_engine, policy=TEST_POLICY)
    feed_low_utilization(engine, gpu, minutes=20, percent=1.0)
    engine.respond("GPU0", ConfirmationResponse.NO, now=BASE + timedelta(minutes=20))

    assert allocation_engine.get_gpus_for_user("U1") == []
