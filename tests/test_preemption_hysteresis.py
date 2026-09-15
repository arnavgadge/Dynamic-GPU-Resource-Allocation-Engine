"""Phase 12 of the 100-scenario fix set: a resource-request/preemption
ask on a GPU cannot be repeated (for any job, not just the one that
was already declined) within the configured cooldown window - the
actual anti-thrashing guard against utilization merely oscillating
near a threshold.
"""

from datetime import datetime, timedelta, timezone

from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.policy import ConfirmationResponse
from engine.scheduler import Scheduler

START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def _submit(s, job, now):
    s.submit_job(job, now=now)
    s.try_allocate_all(now=now)
    return job


def test_a_different_jobs_ask_is_blocked_by_the_same_gpus_cooldown_then_allowed_after_it_elapses():
    s = Scheduler()
    s.add_gpu(GPU(gpu_id="GPU-1", total_memory_mb=1000, status=GPUStatus.IDLE))
    s.add_user(User(user_id="A", name="A", priority=Priority.LOW))
    _submit(s, Job(job_id="ja", user_id="A", name="ja", priority=Priority.LOW,
                     estimated_size_minutes=999, submitted_at=START), START)

    s.add_user(User(user_id="B", name="B", priority=Priority.HIGH))
    jb = Job(job_id="jb", user_id="B", name="jb", priority=Priority.HIGH,
              estimated_size_minutes=10, submitted_at=START + timedelta(minutes=1))
    _submit(s, jb, jb.submitted_at)
    assert s.reclamation_engine.has_pending_prompt("GPU-1")
    s.respond_to_prompt("GPU-1", ConfirmationResponse.YES, now=jb.submitted_at + timedelta(seconds=1))
    s.try_allocate_all(now=jb.submitted_at + timedelta(seconds=1))

    s.add_user(User(user_id="C", name="C", priority=Priority.HIGH))
    jc = Job(job_id="jc", user_id="C", name="jc", priority=Priority.HIGH,
              estimated_size_minutes=10, submitted_at=jb.submitted_at + timedelta(minutes=1))
    _submit(s, jc, jc.submitted_at)
    assert not s.reclamation_engine.has_pending_prompt("GPU-1"), (
        "a different job's ask must still be blocked by the same GPU's cooldown, not just the original decliner's"
    )

    s.add_user(User(user_id="D", name="D", priority=Priority.HIGH))
    jd = Job(job_id="jd", user_id="D", name="jd", priority=Priority.HIGH,
              estimated_size_minutes=10, submitted_at=jb.submitted_at + timedelta(minutes=15))
    _submit(s, jd, jd.submitted_at)
    assert s.reclamation_engine.has_pending_prompt("GPU-1"), "once the cooldown elapses, a fresh ask must be allowed again"
