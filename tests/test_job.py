from datetime import datetime, timedelta, timezone

from engine.models.enums import JobStatus, Priority
from engine.models.job import Job


def test_job_can_be_created_correctly():
    job = Job(job_id="J1", user_id="U1", name="ML Training",
              priority=Priority.HIGH, estimated_size_minutes=180)

    assert job.job_id == "J1"
    assert job.user_id == "U1"
    assert job.name == "ML Training"
    assert job.priority == Priority.HIGH
    assert job.estimated_size_minutes == 180
    assert job.status == JobStatus.WAITING
    assert job.assigned_gpu_id is None
    assert job.started_at is None


def test_job_status_can_represent_every_defined_state():
    for status in JobStatus:
        job = Job(job_id="J1", user_id="U1", name="Job", priority=Priority.LOW,
                  estimated_size_minutes=10, status=status)
        assert job.status is status


def test_waiting_time_uses_now_while_job_has_not_started():
    submitted = datetime.now(timezone.utc) - timedelta(minutes=5)
    job = Job(job_id="J1", user_id="U1", name="Job", priority=Priority.LOW,
              estimated_size_minutes=10, submitted_at=submitted)

    assert job.waiting_time >= timedelta(minutes=5)


def test_waiting_time_uses_started_at_once_running():
    submitted = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    started = datetime(2026, 1, 1, 0, 4, 0, tzinfo=timezone.utc)
    job = Job(job_id="J1", user_id="U1", name="Job", priority=Priority.LOW,
              estimated_size_minutes=10, submitted_at=submitted, started_at=started,
              status=JobStatus.RUNNING, assigned_gpu_ids=["GPU0"])

    assert job.waiting_time == timedelta(minutes=4)
