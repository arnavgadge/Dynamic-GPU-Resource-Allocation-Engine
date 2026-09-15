"""Phase 15 of the 100-scenario fix set: SchedulerState.events and
ReclaimHistory must not grow without bound - verified by actually
running enough real breach-to-reclaim cycles to exceed both caps, not
by inspecting the eviction code and assuming it works.
"""

from datetime import datetime, timedelta, timezone

from engine.history_config import MAX_EVENT_HISTORY, MAX_RECLAIM_HISTORY
from engine.models.enums import GPUStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.policy import ConfirmationResponse
from engine.scheduler import Scheduler

BASE = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def test_reclaim_history_stays_bounded_across_many_real_cycles():
    s = Scheduler()
    s.add_gpu(GPU(gpu_id="GPU-1", total_memory_mb=1000, status=GPUStatus.IDLE))
    s.add_user(User(user_id="A", name="A", priority=Priority.MEDIUM))

    t = BASE
    cycles = MAX_RECLAIM_HISTORY + 50  # deliberately exceed the cap
    for i in range(cycles):
        job = Job(job_id=f"j{i}", user_id="A", name=f"j{i}", priority=Priority.MEDIUM,
                    estimated_size_minutes=999, submitted_at=t)
        s.submit_job(job, now=t)
        s.try_allocate_all(now=t)
        for m in (0, 5, 10, 15, 20, 25):
            s.record_utilization("GPU-1", 1.0, t + timedelta(minutes=m))
        s.respond_to_prompt("GPU-1", ConfirmationResponse.NO, now=t + timedelta(minutes=26))
        s.try_allocate_all(now=t + timedelta(minutes=26))
        t += timedelta(minutes=30)

    assert s.reclamation_engine._history.size() == MAX_RECLAIM_HISTORY
    # the most recent reclaim is still exactly right - eviction never
    # touches the end undo_last/peek_last actually read from.
    last = s.reclamation_engine.last_reclaim()
    assert last is not None and last.job_id == f"j{cycles - 1}"


def test_event_log_stays_bounded_across_many_real_events():
    s = Scheduler()
    s.add_gpu(GPU(gpu_id="GPU-1", total_memory_mb=1000, status=GPUStatus.IDLE))
    s.add_user(User(user_id="A", name="A", priority=Priority.MEDIUM))

    t = BASE
    # each cycle logs several real events (REQUEST, BALANCE, ALLOC,
    # STATUS, JOB_COMPLETION) - enough iterations to exceed the cap.
    iterations = (MAX_EVENT_HISTORY // 4) + 50
    for i in range(iterations):
        job = Job(job_id=f"j{i}", user_id="A", name=f"j{i}", priority=Priority.MEDIUM,
                    estimated_size_minutes=10, submitted_at=t)
        s.submit_job(job, now=t)
        s.try_allocate_all(now=t)
        s.complete_job(f"j{i}", now=t + timedelta(minutes=10))
        t += timedelta(minutes=11)

    assert len(s.state.events) == MAX_EVENT_HISTORY
    # newest events are the ones actually kept.
    assert s.state.events[-1].job_id == f"j{iterations - 1}"
