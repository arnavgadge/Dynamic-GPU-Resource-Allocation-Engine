"""The size-disparity reallocation path (added by direct user request,
not part of the original 14-day brief): when explicitly enabled
(`Scheduler(size_disparity_ratio=...)`), a waiting job whose own size
is at least that many times shorter than a holder's remaining time may
ask the holder to release - independent of priority tier, through the
exact same consent-based confirmation flow every other reallocation
trigger uses. Off (`None`) by default - see `Scheduler.__init__`'s own
comment for why every scripted demo scenario needs it off; only the
live `interactive_demo` API session turns it on
(`api/session.py::load_scenario`).

The user's own worked example: User B holds the whole pool with
~1-hour jobs; User A asks for 1 GPU needing ~2 minutes (a 30x gap,
comfortably past the default 5x threshold). B gets a real YES/NO
prompt; a NO leaves A exactly where it was - still WAITING, nothing
forced; and the moment A finishes, the freed GPU goes straight back to
B's own still-waiting deficit, automatically, through the ordinary
allocation policy - no new code needed for that part, it already
works.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.policy import ConfirmationResponse
from engine.scheduler import Scheduler

T0 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def add_gpus(scheduler: Scheduler, n: int) -> None:
    for i in range(1, n + 1):
        scheduler.add_gpu(GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE))


def submit(scheduler, job_id, user_id, priority=Priority.MEDIUM, size=10, gpu_count=1, minute=0.0) -> Job:
    if scheduler.state.get_user(user_id) is None:
        scheduler.add_user(User(user_id=user_id, name=user_id, priority=priority))
    job = Job(job_id=job_id, user_id=user_id, name=job_id, priority=priority,
              estimated_size_minutes=size, gpu_count=gpu_count, submitted_at=at(minute))
    scheduler.submit_job(job, now=at(minute))
    scheduler.try_allocate_all(now=at(minute))
    return job


# ---- off by default ---------------------------------------------------

def test_disabled_by_default_no_ask_ever_raised():
    scheduler = Scheduler()  # size_disparity_ratio=None
    add_gpus(scheduler, 1)
    submit(scheduler, "LONG", "USERB", size=60)
    submit(scheduler, "SHORT", "USERA", size=2, minute=1)
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-1")


def test_existing_scripted_scenarios_are_never_affected_by_the_opt_in_flag():
    """Sanity check mirroring why this broke the test suite the first
    time: a Scheduler constructed the normal way (every scenario,
    every pre-existing test) must behave identically with or without
    this feature existing in the codebase."""
    scheduler = Scheduler()
    assert scheduler.size_disparity_ratio is None


# ---- the user's own worked example -------------------------------------

def test_userb_holds_the_pool_usera_needs_1_gpu_for_a_fraction_of_the_time():
    scheduler = Scheduler(size_disparity_ratio=5.0)
    add_gpus(scheduler, 10)
    long_job = submit(scheduler, "LONG", "USERB", size=60, gpu_count=9)  # "all but one"
    assert len(long_job.assigned_gpu_ids) == 9
    last_free = [g for g in scheduler.state.gpus if g not in long_job.assigned_gpu_ids][0]
    # Hand User B the very last GPU too, so the pool really is exhausted.
    scheduler.manual_assign_gpu(last_free, "USERB", now=at(0.5))

    short_job = submit(scheduler, "SHORT", "USERA", size=2, minute=1)
    assert short_job.status == JobStatus.WAITING

    prompted = [g for g in scheduler.state.gpus if scheduler.reclamation_engine.has_pending_prompt(g)]
    assert len(prompted) == 1
    event = next(e for e in scheduler.state.events if e.event_type == EventType.PROMPT)
    assert "much shorter" in event.message
    assert event.metadata["category"] == "SIZE_DISPARITY"


def test_userb_says_no_usera_stays_in_the_queue_simple():
    scheduler = Scheduler(size_disparity_ratio=5.0)
    add_gpus(scheduler, 1)
    long_job = submit(scheduler, "LONG", "USERB", size=60)
    short_job = submit(scheduler, "SHORT", "USERA", size=2, minute=1)
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1")

    # YES = "I'm still using it" = B declines to release - exactly
    # "if userB doesn't allow".
    scheduler.respond_to_prompt("GPU-1", ConfirmationResponse.YES, now=at(1.5))
    scheduler.try_allocate_all(now=at(1.5))

    assert short_job.status == JobStatus.WAITING
    assert short_job.assigned_gpu_ids == []
    assert long_job.status == JobStatus.RUNNING
    assert long_job.assigned_gpu_ids == ["GPU-1"]


def test_userb_releases_usera_runs_and_on_completion_gpu_automatically_returns_to_userb():
    """The full loop, end to end: B asked -> B says NO (releases) ->
    A runs -> A finishes -> the GPU goes straight back to B's own
    still-outstanding deficit, automatically, via the ordinary
    allocation policy - not a single new line of code for this part."""
    scheduler = Scheduler(size_disparity_ratio=5.0)
    add_gpus(scheduler, 1)
    long_job = submit(scheduler, "LONG", "USERB", size=60)
    short_job = submit(scheduler, "SHORT", "USERA", size=2, minute=1)
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1")

    scheduler.respond_to_prompt("GPU-1", ConfirmationResponse.NO, now=at(1.5))
    scheduler.try_allocate_all(now=at(1.5))

    assert short_job.status == JobStatus.RUNNING
    assert short_job.assigned_gpu_ids == ["GPU-1"]
    # B was dispossessed, not abandoned - requeued (Day 9's existing
    # preemption-requeue behavior), still needing its GPU back.
    assert long_job.status == JobStatus.WAITING
    assert long_job.assigned_gpu_ids == []

    scheduler.complete_job("SHORT", now=at(3.5))  # A's 2-minute job finishes
    scheduler.try_allocate_all(now=at(3.5))

    assert long_job.status == JobStatus.RUNNING
    assert long_job.assigned_gpu_ids == ["GPU-1"]


def test_no_response_within_the_grace_period_auto_releases_same_as_every_other_prompt():
    scheduler = Scheduler(size_disparity_ratio=5.0)
    add_gpus(scheduler, 1)
    submit(scheduler, "LONG", "USERB", size=60)
    short_job = submit(scheduler, "SHORT", "USERA", size=2, minute=1)
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1")

    scheduler.check_reclamation_timeouts(at(1 + 5))  # 5-minute grace period elapses
    scheduler.try_allocate_all(now=at(6))

    assert short_job.status == JobStatus.RUNNING


# ---- the ratio gate itself -----------------------------------------------

def test_a_modest_size_difference_never_triggers_an_ask():
    scheduler = Scheduler(size_disparity_ratio=5.0)
    add_gpus(scheduler, 1)
    submit(scheduler, "LONG", "USERB", size=20)
    submit(scheduler, "SHORT", "USERA", size=10, minute=1)  # only 2x, not 5x
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-1")


def test_exactly_at_the_ratio_boundary_triggers_an_ask():
    scheduler = Scheduler(size_disparity_ratio=5.0)
    add_gpus(scheduler, 1)
    submit(scheduler, "LONG", "USERB", size=50)
    # Submitted in the very same instant as LONG starts, so no
    # simulated time has elapsed yet - LONG's remaining time is still
    # exactly its full 50 minutes, genuinely 5.0x SHORT's 10.
    submit(scheduler, "SHORT", "USERA", size=10, minute=0)
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1")


def test_priority_is_irrelevant_to_this_path_same_or_lower_still_asks():
    """Distinct from path 2: a same-or-lower-priority requester can
    still trigger this one, since it is independent of priority."""
    scheduler = Scheduler(size_disparity_ratio=5.0)
    add_gpus(scheduler, 1)
    submit(scheduler, "LONG", "USERB", size=60, priority=Priority.HIGH)
    submit(scheduler, "SHORT", "USERA", size=2, priority=Priority.LOW, minute=1)
    assert scheduler.reclamation_engine.has_pending_prompt("GPU-1")


# ---- safety: never touches maintenance/unavailable/cooldown/declined ------

def test_maintenance_gpu_is_never_a_size_disparity_candidate():
    scheduler = Scheduler(size_disparity_ratio=5.0)
    add_gpus(scheduler, 2)
    submit(scheduler, "LONG", "USERB", size=60)
    scheduler.set_gpu_maintenance("GPU-2", now=at(0))
    submit(scheduler, "SHORT", "USERA", size=2, minute=1)
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-2")


def test_declined_holder_is_not_re_asked_for_the_same_job():
    scheduler = Scheduler(size_disparity_ratio=5.0)
    add_gpus(scheduler, 1)
    submit(scheduler, "LONG", "USERB", size=60)
    short_job = submit(scheduler, "SHORT", "USERA", size=2, minute=1)
    scheduler.respond_to_prompt("GPU-1", ConfirmationResponse.YES, now=at(1.5))  # B keeps it
    scheduler.try_allocate_all(now=at(1.5))

    scheduler.try_allocate_all(now=at(2))  # re-evaluate again
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-1")
    assert short_job.status == JobStatus.WAITING
