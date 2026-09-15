"""Direct tests for the new admin-facing Scheduler actions added while
fixing the 100-scenario validation report: cancel_job (Phase 7),
force_reclaim (Phase 8), change_job_priority (Phase 9), and the
maintenance-mode pair (Phase 14).
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.scheduler import Scheduler

BASE = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def _scheduler(n_gpus=1):
    s = Scheduler()
    for i in range(1, n_gpus + 1):
        s.add_gpu(GPU(gpu_id=f"GPU-{i}", total_memory_mb=1000, status=GPUStatus.IDLE))
    return s


def _submit(s, job, now):
    """`Scheduler.submit_job` only registers a job - it never
    allocates on its own (that is `try_allocate_all`'s job). Every
    test below wants the immediate real-allocation attempt too."""
    s.submit_job(job, now=now)
    s.try_allocate_all(now=now)
    return job


# -- Phase 7: cancel_job -----------------------------------------------

def test_cancel_job_withdraws_a_waiting_job():
    s = _scheduler(1)
    s.add_user(User(user_id="A", name="A", priority=Priority.MEDIUM))
    _submit(s, Job(job_id="J1", user_id="A", name="j1", priority=Priority.MEDIUM,
                    estimated_size_minutes=10, submitted_at=BASE), BASE)  # occupies the only GPU
    s.add_user(User(user_id="B", name="B", priority=Priority.MEDIUM))
    jb = Job(job_id="J2", user_id="B", name="j2", priority=Priority.MEDIUM, estimated_size_minutes=10, submitted_at=BASE)
    _submit(s, jb, BASE)
    assert jb.status == JobStatus.WAITING

    event = s.cancel_job("J2", now=BASE + timedelta(minutes=1))
    assert event.event_type == EventType.JOB_CANCELLED
    assert jb.status == JobStatus.CANCELLED
    assert "J2" not in s.allocation_engine.waiting_job_ids()

    # cancelled job is never later allocated even once a GPU frees.
    s.complete_job("J1", now=BASE + timedelta(minutes=5))
    s.try_allocate_all(now=BASE + timedelta(minutes=5))
    assert jb.status == JobStatus.CANCELLED


def test_cancel_job_rejects_a_running_job():
    s = _scheduler(1)
    s.add_user(User(user_id="A", name="A", priority=Priority.MEDIUM))
    ja = Job(job_id="J1", user_id="A", name="j1", priority=Priority.MEDIUM, estimated_size_minutes=10, submitted_at=BASE)
    _submit(s, ja, BASE)
    assert ja.status == JobStatus.RUNNING
    with pytest.raises(ValueError):
        s.cancel_job("J1", now=BASE)


# -- Phase 8: force_reclaim ---------------------------------------------

def test_force_reclaim_bypasses_confirmation_and_reallocates():
    s = _scheduler(1)
    s.add_user(User(user_id="A", name="A", priority=Priority.MEDIUM))
    ja = Job(job_id="J1", user_id="A", name="j1", priority=Priority.MEDIUM, estimated_size_minutes=999, submitted_at=BASE)
    _submit(s, ja, BASE)
    s.add_user(User(user_id="B", name="B", priority=Priority.MEDIUM))
    jb = Job(job_id="J2", user_id="B", name="j2", priority=Priority.MEDIUM, estimated_size_minutes=10, submitted_at=BASE)
    _submit(s, jb, BASE)
    assert jb.status == JobStatus.WAITING

    event = s.force_reclaim("GPU-1", now=BASE + timedelta(seconds=1))  # no prompt, no wait
    assert event.event_type == EventType.ADMIN_FORCE_RECLAIM
    assert ja.status == JobStatus.RECLAIMED
    assert jb.status == JobStatus.RUNNING  # reevaluated and allocated immediately
    assert jb.assigned_gpu_id == "GPU-1"


def test_force_reclaim_rejects_an_unassigned_gpu():
    s = _scheduler(1)
    with pytest.raises(ValueError):
        s.force_reclaim("GPU-1", now=BASE)


# -- Phase 9: change_job_priority ---------------------------------------

def test_change_job_priority_lets_a_low_priority_job_win_after_upgrade():
    s = _scheduler(1)
    s.add_user(User(user_id="occupant", name="occupant", priority=Priority.MEDIUM))
    occ = Job(job_id="occ", user_id="occupant", name="occ", priority=Priority.MEDIUM, estimated_size_minutes=10, submitted_at=BASE)
    _submit(s, occ, BASE)  # occupies the only GPU

    s.add_user(User(user_id="A", name="A", priority=Priority.LOW))
    s.add_user(User(user_id="B", name="B", priority=Priority.MEDIUM))
    ja = Job(job_id="JA", user_id="A", name="ja", priority=Priority.LOW, estimated_size_minutes=60, submitted_at=BASE)
    jb = Job(job_id="JB", user_id="B", name="jb", priority=Priority.MEDIUM, estimated_size_minutes=60, submitted_at=BASE + timedelta(minutes=1))
    _submit(s, ja, BASE)
    _submit(s, jb, jb.submitted_at)

    event = s.change_job_priority("JA", Priority.CRITICAL, now=BASE + timedelta(minutes=2))
    assert event.event_type == EventType.PRIORITY_CHANGED
    assert ja.priority == Priority.CRITICAL

    s.complete_job("occ", now=BASE + timedelta(minutes=10))
    s.try_allocate_all(now=BASE + timedelta(minutes=10))
    assert ja.status == JobStatus.RUNNING  # now CRITICAL -> wins the critical-tier restriction outright
    assert jb.status == JobStatus.WAITING


def test_change_job_priority_rejects_a_running_job():
    s = _scheduler(1)
    s.add_user(User(user_id="A", name="A", priority=Priority.LOW))
    ja = Job(job_id="J1", user_id="A", name="j1", priority=Priority.LOW, estimated_size_minutes=10, submitted_at=BASE)
    _submit(s, ja, BASE)
    with pytest.raises(ValueError):
        s.change_job_priority("J1", Priority.CRITICAL, now=BASE)


# -- Phase 14: maintenance mode ------------------------------------------

def test_maintenance_removes_capacity_and_clearing_it_restores_and_reallocates():
    s = _scheduler(2)
    event = s.set_gpu_maintenance("GPU-2", now=BASE)
    assert event.event_type == EventType.SYSTEM
    assert s.state.get_gpu("GPU-2").status == GPUStatus.MAINTENANCE
    assert s.allocation_engine.available_gpu_count() == 1  # GPU-1 only

    s.add_user(User(user_id="A", name="A", priority=Priority.MEDIUM))
    ja = Job(job_id="J1", user_id="A", name="j1", priority=Priority.MEDIUM, estimated_size_minutes=10, submitted_at=BASE)
    _submit(s, ja, BASE)
    assert ja.assigned_gpu_id == "GPU-1"  # never routed onto the maintenance GPU

    s.add_user(User(user_id="B", name="B", priority=Priority.MEDIUM))
    jb = Job(job_id="J2", user_id="B", name="j2", priority=Priority.MEDIUM, estimated_size_minutes=10, submitted_at=BASE)
    _submit(s, jb, BASE)
    assert jb.status == JobStatus.WAITING  # GPU-2 unavailable, no other free GPU

    s.clear_gpu_maintenance("GPU-2", now=BASE + timedelta(minutes=1))
    assert s.state.get_gpu("GPU-2").status in (GPUStatus.IDLE, GPUStatus.ACTIVE)
    assert jb.status == JobStatus.RUNNING  # reevaluated immediately
    assert jb.assigned_gpu_id == "GPU-2"


def test_maintenance_rejects_a_currently_assigned_gpu():
    s = _scheduler(1)
    s.add_user(User(user_id="A", name="A", priority=Priority.MEDIUM))
    _submit(s, Job(job_id="J1", user_id="A", name="j1", priority=Priority.MEDIUM,
                    estimated_size_minutes=10, submitted_at=BASE), BASE)
    with pytest.raises(ValueError):
        s.set_gpu_maintenance("GPU-1", now=BASE)


# -- Phase 6: hardware failure handling -----------------------------------

def test_handle_gpu_failure_marks_unavailable_and_requeues_the_affected_job():
    s = _scheduler(2)
    s.add_user(User(user_id="A", name="A", priority=Priority.MEDIUM))
    ja = Job(job_id="J1", user_id="A", name="j1", priority=Priority.MEDIUM, estimated_size_minutes=999, submitted_at=BASE)
    _submit(s, ja, BASE)
    assert ja.assigned_gpu_id == "GPU-1"

    event = s.handle_gpu_failure("GPU-1", now=BASE + timedelta(minutes=1))
    assert event.event_type == EventType.HARDWARE_FAILURE
    assert s.state.get_gpu("GPU-1").status == GPUStatus.UNAVAILABLE
    assert "GPU-1" in s.state.gpus  # kept, never deleted
    assert "GPU-1" not in s.state.get_user("A").assigned_gpu_ids

    # handle_gpu_failure reevaluates the waiting queue itself - since
    # GPU-2 was genuinely free, the job goes WAITING and is
    # immediately re-allocated onto it in the very same call, exactly
    # like any other freed-capacity reevaluation in this project.
    assert ja.status == JobStatus.RUNNING
    assert ja.assigned_gpu_id == "GPU-2"


def test_handle_gpu_failure_on_a_multi_gpu_job_keeps_its_other_gpus():
    s = _scheduler(3)
    s.add_user(User(user_id="A", name="A", priority=Priority.MEDIUM))
    ja = Job(job_id="J1", user_id="A", name="j1", priority=Priority.MEDIUM, estimated_size_minutes=999,
              submitted_at=BASE, gpu_count=2)
    _submit(s, ja, BASE)
    assert sorted(ja.assigned_gpu_ids) == ["GPU-1", "GPU-2"]

    s.handle_gpu_failure("GPU-1", now=BASE + timedelta(minutes=1))
    assert ja.status == JobStatus.RUNNING  # still holds GPU-2
    assert ja.assigned_gpu_ids == ["GPU-2"]
    assert ja.gpus_still_needed == 1


def test_handle_gpu_failure_is_a_noop_for_an_already_unavailable_gpu():
    s = _scheduler(1)
    s.handle_gpu_failure("GPU-1", now=BASE)
    assert s.handle_gpu_failure("GPU-1", now=BASE + timedelta(minutes=1)) is None
