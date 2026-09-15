"""Direct unit tests for `engine.scheduler.Scheduler` - the thin
orchestrator over Allocation/Reclamation/Balancing. Most of its
methods were previously only exercised indirectly (via `Simulator` or
the API layer); this file tests it on its own.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.decision import ReclamationAction
from engine.reclamation.policy import ConfirmationResponse
from engine.scheduler import Scheduler

BASE = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def running_job_on_gpu(scheduler, gpu_id, user_id, job_id, size_minutes, priority=Priority.MEDIUM, util=50.0):
    scheduler.add_user(User(user_id=user_id, name=user_id, priority=priority))
    gpu = GPU(gpu_id=gpu_id, total_memory_mb=1000, utilization_percent=util, status=GPUStatus.ACTIVE,
               assigned_user_id=user_id, assigned_job_id=job_id)
    job = Job(job_id=job_id, user_id=user_id, name="job", priority=priority, estimated_size_minutes=size_minutes,
               status=JobStatus.RUNNING, started_at=BASE, submitted_at=BASE, assigned_gpu_ids=[gpu_id])
    scheduler.state.add_job(job)
    scheduler.state.get_user(user_id).assigned_gpu_ids.append(gpu_id)
    scheduler.state.get_user(user_id).running_job_ids.append(job_id)
    scheduler.allocation_engine.add_gpu(gpu)
    return gpu, job


def test_submit_job_and_try_allocate_all_reach_a_waiting_job():
    scheduler = Scheduler()
    scheduler.add_user(User(user_id="u1", name="U1", priority=Priority.HIGH))
    scheduler.add_gpu(GPU(gpu_id="GPU-1", total_memory_mb=1000, status=GPUStatus.IDLE))

    event = scheduler.submit_job(Job(job_id="J1", user_id="u1", name="job", priority=Priority.HIGH,
                                       estimated_size_minutes=10, submitted_at=BASE), now=BASE)
    assert event.event_type.value == "REQUEST"

    decisions = scheduler.try_allocate_all(now=BASE)
    assert len(decisions) == 1
    assert scheduler.state.get_job("J1").status == JobStatus.RUNNING


def test_complete_job_releases_gpu_and_lets_a_waiting_job_take_it():
    scheduler = Scheduler()
    gpu, job = running_job_on_gpu(scheduler, "GPU-1", "u1", "J1", size_minutes=30)
    scheduler.add_user(User(user_id="u2", name="U2", priority=Priority.MEDIUM))
    scheduler.submit_job(Job(job_id="J2", user_id="u2", name="job2", priority=Priority.MEDIUM,
                               estimated_size_minutes=10, submitted_at=BASE), now=BASE)

    event = scheduler.complete_job("J1", now=BASE + timedelta(minutes=30))
    # Phase 5 (100-scenario validation): a normal completion is its own
    # distinct event type now, not the generic STATUS - the original
    # expectation predated JOB_COMPLETION existing at all.
    assert event.event_type.value == "JOB_COMPLETION"
    assert job.status == JobStatus.COMPLETED
    assert gpu.assigned_user_id is None

    scheduler.try_allocate_all(now=BASE + timedelta(minutes=30))
    assert scheduler.state.get_job("J2").status == JobStatus.RUNNING
    assert scheduler.state.get_job("J2").assigned_gpu_id == "GPU-1"


def test_complete_job_rejects_a_job_that_is_not_running():
    scheduler = Scheduler()
    scheduler.add_user(User(user_id="u1", name="U1", priority=Priority.MEDIUM))
    scheduler.state.add_job(Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
                                   estimated_size_minutes=10, submitted_at=BASE))  # WAITING
    with pytest.raises(ValueError):
        scheduler.complete_job("J1", now=BASE)


def test_complete_job_rejects_an_unknown_job():
    scheduler = Scheduler()
    with pytest.raises(ValueError):
        scheduler.complete_job("NOPE", now=BASE)


# ------------------------------------------------------------------
# check_estimated_completions
# ------------------------------------------------------------------

def test_estimated_completion_prompts_through_the_real_confirmation_flow_not_by_deleting_the_job():
    scheduler = Scheduler()
    gpu, job = running_job_on_gpu(scheduler, "GPU-1", "u1", "J1", size_minutes=20, util=80.0)

    # Before the estimated duration elapses: nothing happens, even
    # though nothing here looks at utilization at all.
    decisions = scheduler.check_estimated_completions(BASE + timedelta(minutes=10))
    assert decisions == []
    assert gpu.status == GPUStatus.ACTIVE
    assert job.status == JobStatus.RUNNING  # never silently marked complete

    decisions = scheduler.check_estimated_completions(BASE + timedelta(minutes=20))
    assert len(decisions) == 1
    assert decisions[0].action == ReclamationAction.PROMPTED
    assert decisions[0].tier is None  # not a utilization tier - honestly reported as such
    assert gpu.status == GPUStatus.IDLE_WARNING
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1") is True


def test_estimated_completion_prompt_reuses_yes_no_exactly_like_a_tier_prompt():
    scheduler = Scheduler()
    gpu, job = running_job_on_gpu(scheduler, "GPU-1", "u1", "J1", size_minutes=20)
    scheduler.check_estimated_completions(BASE + timedelta(minutes=20))

    decision = scheduler.respond_to_prompt("GPU-1", ConfirmationResponse.NO, now=BASE + timedelta(minutes=21))
    assert decision.action == ReclamationAction.RECLAIMED
    assert gpu.status == GPUStatus.IDLE
    assert job.status == JobStatus.RECLAIMED


def test_estimated_completion_does_not_double_prompt_a_gpu_already_prompted():
    scheduler = Scheduler()
    gpu, job = running_job_on_gpu(scheduler, "GPU-1", "u1", "J1", size_minutes=20)
    first = scheduler.check_estimated_completions(BASE + timedelta(minutes=20))
    assert len(first) == 1

    second = scheduler.check_estimated_completions(BASE + timedelta(minutes=21))
    assert second == []  # already pending - not re-prompted


def test_try_allocate_all_records_the_last_decision_pair_for_a_decision_trace():
    scheduler = Scheduler()
    scheduler.add_user(User(user_id="u1", name="U1", priority=Priority.HIGH))
    scheduler.add_gpu(GPU(gpu_id="GPU-1", total_memory_mb=1000, status=GPUStatus.IDLE))
    scheduler.add_gpu(GPU(gpu_id="GPU-2", total_memory_mb=1000, status=GPUStatus.IDLE))

    assert scheduler.last_allocation_decision is None
    assert scheduler.last_routing_decision is None

    scheduler.submit_job(Job(job_id="J1", user_id="u1", name="job", priority=Priority.HIGH,
                               estimated_size_minutes=10, submitted_at=BASE), now=BASE)
    scheduler.try_allocate_all(now=BASE)

    assert scheduler.last_allocation_decision is not None
    assert scheduler.last_allocation_decision.job_id == "J1"
    assert scheduler.last_routing_decision is not None
    assert scheduler.last_routing_decision.selected_gpu_id == scheduler.last_allocation_decision.gpu_id


def test_estimated_completion_ignores_waiting_and_completed_jobs():
    scheduler = Scheduler()
    scheduler.add_user(User(user_id="u1", name="U1", priority=Priority.MEDIUM))
    scheduler.state.add_job(Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
                                   estimated_size_minutes=5, submitted_at=BASE))  # WAITING, no GPU
    decisions = scheduler.check_estimated_completions(BASE + timedelta(minutes=100))
    assert decisions == []


def test_available_gpu_count_matches_the_real_free_count_through_the_real_allocation_path():
    """100-scenario validation, Phase 13: available_gpu_count() must
    never drift from SchedulerState's own truth, specifically along
    the path Scheduler.try_allocate_all actually uses (the router,
    not AllocationEngine.allocate_next)."""
    scheduler = Scheduler()
    for i in range(1, 5):
        scheduler.add_gpu(GPU(gpu_id=f"GPU-{i}", total_memory_mb=1000, status=GPUStatus.IDLE))

    def real_free_count():
        return sum(1 for g in scheduler.state.gpus.values() if not g.is_assigned)

    assert scheduler.allocation_engine.available_gpu_count() == real_free_count() == 4

    scheduler.add_user(User(user_id="u1", name="U1", priority=Priority.MEDIUM))
    scheduler.add_user(User(user_id="u2", name="U2", priority=Priority.MEDIUM))
    scheduler.submit_job(Job(job_id="J1", user_id="u1", name="j1", priority=Priority.MEDIUM,
                               estimated_size_minutes=10, submitted_at=BASE), now=BASE)
    scheduler.submit_job(Job(job_id="J2", user_id="u2", name="j2", priority=Priority.MEDIUM,
                               estimated_size_minutes=10, submitted_at=BASE), now=BASE)
    scheduler.try_allocate_all(now=BASE)  # the real path - router + finalize_assignment, not allocate_next

    assert scheduler.allocation_engine.available_gpu_count() == real_free_count() == 2

    scheduler.complete_job("J1", now=BASE + timedelta(minutes=10))
    assert scheduler.allocation_engine.available_gpu_count() == real_free_count() == 3
