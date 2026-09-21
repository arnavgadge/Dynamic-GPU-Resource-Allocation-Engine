"""Day 7: utilization-based GPU reclamation with the Sliding Window,
verified end to end through the real `Scheduler`.

Policy under test (unchanged - thresholds are the project's own):
Tier 1 = below 2% sustained 25 min; Tier 2 = below 15% sustained
2 h 30 min; a confirmation prompt ("are you still using this GPU?")
with a 5-minute no-response grace period; YES keeps the GPU and
resets the timer, NO (or silence past the grace period) reclaims it,
and the freed GPU goes back through the ordinary allocation policy.

Time is entirely simulated: every reading/response/timeout is stamped
explicitly (`at(minutes)`), so "25 minutes" costs microseconds and
nothing here depends on a wall clock. Utilization values are fixed
constants, never random.

While writing these tests two real bugs were found and fixed in
`ReclamationEngine` (both reproduced first): a GPU's idle history was
credited to *whoever held it next* - readings taken while the GPU was
unassigned, or while a since-completed job held it, made a brand-new
holder look "sustained idle" and prompted them within minutes.
Breach detection now only counts readings from the current holder's
tenure (`holder_since`), and `clear_watch_for_gpu` always restarts
the baseline even when no prompt was pending. The RECLAIM event (and
so the `ReclaimHistory` entry) also now records the GPU's previous and
resulting status.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engine.dsa.sliding_window import UtilizationSlidingWindow
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.models.utilization import UtilizationObservation
from engine.reclamation.decision import ReclamationAction
from engine.reclamation.monitor import is_sustained_breach
from engine.reclamation.policy import DEFAULT_RECLAMATION_POLICY, ConfirmationResponse, ReclamationTier
from engine.scheduler import Scheduler

T0 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
YES, NO = ConfirmationResponse.YES, ConfirmationResponse.NO


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def make_scheduler(n_gpus: int = 1) -> Scheduler:
    scheduler = Scheduler()
    for i in range(1, n_gpus + 1):
        scheduler.add_gpu(GPU(gpu_id=f"GPU-{i}", total_memory_mb=24_576, status=GPUStatus.IDLE))
    return scheduler


def start_job(scheduler: Scheduler, job_id: str, user_id: str, minute: float = 0, gpu_count: int = 1,
              size: float = 100_000) -> Job:
    if scheduler.state.get_user(user_id) is None:
        scheduler.add_user(User(user_id=user_id, name=user_id, priority=Priority.MEDIUM))
    job = Job(job_id=job_id, user_id=user_id, name=job_id, priority=Priority.MEDIUM,
              estimated_size_minutes=size, gpu_count=gpu_count, submitted_at=at(minute))
    scheduler.submit_job(job, now=at(minute))
    scheduler.try_allocate_all(now=at(minute))
    return job


def feed(scheduler: Scheduler, gpu_id: str, utilization: float, minutes):
    """One reading per listed minute; returns [(minute, decision)] for
    every reading that produced a decision."""
    out = []
    for minute in minutes:
        decision = scheduler.record_utilization(gpu_id, utilization, at(minute))
        if decision is not None:
            out.append((minute, decision))
    return out


def every(step: float, start: float, stop: float):
    minute = start
    while minute <= stop + 1e-9:
        yield minute
        minute += step


def prompt_for(scheduler: Scheduler, job_id="JA", user_id="A", gpu_id="GPU-1", util=1.0):
    """Run a job to its Tier-1 prompt on ``gpu_id`` (prompt at minute 25)."""
    job = start_job(scheduler, job_id, user_id)
    hits = feed(scheduler, gpu_id, util, every(5, 0, 25))
    assert [d.action for _, d in hits] == [ReclamationAction.PROMPTED]
    return job


# ---- 1/2/7. Normal load, brief dips, recovery -------------------------------

def test_normal_high_utilization_never_prompts():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    assert feed(scheduler, "GPU-1", 80.0, every(5, 0, 300)) == []
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.ACTIVE


def test_brief_dip_in_a_busy_gpu_does_not_reclaim():
    """The spec's own series: 80, 78, 2, 3, 75, 82."""
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    for minute, value in zip(every(5, 0, 25), (80.0, 78.0, 2.0, 3.0, 75.0, 82.0)):
        assert scheduler.record_utilization("GPU-1", value, at(minute)) is None
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.ACTIVE


def test_a_long_low_stretch_that_recovers_before_the_threshold_does_not_prompt():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    assert feed(scheduler, "GPU-1", 1.0, every(5, 0, 20)) == []      # 20 min < 25 min
    assert feed(scheduler, "GPU-1", 60.0, [22]) == []                # recovered
    assert feed(scheduler, "GPU-1", 1.0, every(5, 25, 45)) == []     # spike still inside the 25-min window
    hits = feed(scheduler, "GPU-1", 1.0, [50])                        # spike now older than the window
    assert [d.action for _, d in hits] == [ReclamationAction.PROMPTED]


def test_low_average_is_not_enough_a_single_spike_breaks_the_streak():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    assert feed(scheduler, "GPU-1", 1.0, every(5, 0, 15)) == []
    assert feed(scheduler, "GPU-1", 35.0, [20]) == []                # 1,1,1,1,35,1,1: mostly low, but the streak is broken
    assert feed(scheduler, "GPU-1", 1.0, [25, 30]) == []


# ---- 3/4. Tier 1 ---------------------------------------------------------------

def test_tier1_sustained_below_2_percent_prompts_with_the_confirmation_question():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    hits = feed(scheduler, "GPU-1", 1.0, every(5, 0, 25))
    assert [m for m, _ in hits] == [25]
    decision = hits[0][1]
    assert decision.action == ReclamationAction.PROMPTED
    assert decision.tier == ReclamationTier.TIER_1
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.IDLE_WARNING
    prompt = decision.event
    assert prompt.event_type == EventType.PROMPT
    assert "are you still using this GPU?" in prompt.message
    assert prompt.user_id == "A" and prompt.job_id == "JA"     # addressed to the affected user


def test_tier1_boundary_24m54s_no_25m_yes_and_exactly_2_percent_is_not_below():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    assert feed(scheduler, "GPU-1", 1.0, [0, 24.9]) == []                    # just short of 25 min
    assert [d.action for _, d in feed(scheduler, "GPU-1", 1.0, [25])] == [ReclamationAction.PROMPTED]

    other = make_scheduler()
    start_job(other, "JA", "A")
    assert feed(other, "GPU-1", 2.0, every(5, 0, 120)) == []                 # 2.0% is not < 2%... and not < 15% for 2.5h yet
    assert other.state.get_gpu("GPU-1").status == GPUStatus.ACTIVE


# ---- 5/6. Tier 2 --------------------------------------------------------------------

def test_tier2_sustained_below_15_percent_prompts_after_2h30():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    assert feed(scheduler, "GPU-1", 10.0, every(10, 0, 140)) == []
    hits = feed(scheduler, "GPU-1", 10.0, [150])
    assert [d.action for _, d in hits] == [ReclamationAction.PROMPTED]
    assert hits[0][1].tier == ReclamationTier.TIER_2
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.IDLE_WARNING


def test_tier2_boundary_149m_no_150m_yes_and_exactly_15_percent_is_not_below():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    assert feed(scheduler, "GPU-1", 14.9, [0, 149]) == []
    assert len(feed(scheduler, "GPU-1", 14.9, [150])) == 1

    at_threshold = make_scheduler()
    start_job(at_threshold, "JA", "A")
    assert feed(at_threshold, "GPU-1", 15.0, every(10, 0, 300)) == []      # 15.0% is not < 15%


def test_tier1_wins_when_both_tiers_are_breached():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    hits = feed(scheduler, "GPU-1", 1.0, every(5, 0, 160))
    assert hits[0][1].tier == ReclamationTier.TIER_1 and hits[0][0] == 25   # never waits for Tier 2


# ---- 8. YES ---------------------------------------------------------------------------

def test_yes_keeps_the_gpu_and_resets_the_timer():
    scheduler = make_scheduler()
    job = prompt_for(scheduler)
    decision = scheduler.respond_to_prompt("GPU-1", YES, now=at(26))

    assert decision.action == ReclamationAction.BACKED_OFF
    gpu = scheduler.state.get_gpu("GPU-1")
    assert gpu.status == GPUStatus.ACTIVE and gpu.assigned_job_id == "JA"
    assert job.status == JobStatus.RUNNING
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-1")
    assert scheduler.reclamation_engine.last_reclaim() is None

    # The timer restarted: readings from before the YES don't count, so
    # a full new 25 minutes of sustained idleness is required.
    assert feed(scheduler, "GPU-1", 1.0, every(5, 30, 50)) == []
    assert len(feed(scheduler, "GPU-1", 1.0, [55])) == 1


# ---- 9/12/13. NO / reclamation / release ---------------------------------------------------

def test_no_reclaims_and_releases_everything():
    scheduler = make_scheduler()
    job = prompt_for(scheduler)
    decision = scheduler.respond_to_prompt("GPU-1", NO, now=at(26))

    assert decision.action == ReclamationAction.RECLAIMED
    gpu = scheduler.state.get_gpu("GPU-1")
    assert gpu.status == GPUStatus.IDLE
    assert gpu.assigned_user_id is None and gpu.assigned_job_id is None
    assert job.status == JobStatus.RECLAIMED and job.assigned_gpu_ids == []
    user = scheduler.state.get_user("A")
    assert user.assigned_gpu_ids == [] and user.running_job_ids == []
    assert scheduler.allocation_engine.get_gpus_for_user("A") == []
    assert scheduler.state.get_active_assignment_for_gpu("GPU-1") is None
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-1")


# ---- 10/11. No response and the 5-minute warning ---------------------------------------------

def test_the_warning_period_is_five_minutes_not_five_seconds():
    assert DEFAULT_RECLAMATION_POLICY.no_response_grace_period == timedelta(minutes=5)

    from api.session import SimulationSession
    from engine.simulation import load_default_registry
    live = SimulationSession(load_default_registry(), "interactive_demo")
    assert live.simulator.scheduler.reclamation_engine.policy.no_response_grace_period == timedelta(minutes=5)


def test_no_response_reclaims_only_once_the_full_grace_period_has_elapsed():
    scheduler = make_scheduler()
    job = prompt_for(scheduler)                                   # prompted at minute 25

    assert scheduler.check_reclamation_timeouts(at(25) + timedelta(seconds=5)) == []     # not 5 seconds
    assert scheduler.check_reclamation_timeouts(at(25) + timedelta(minutes=4, seconds=59)) == []
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.IDLE_WARNING

    decisions = scheduler.check_reclamation_timeouts(at(25) + timedelta(minutes=5))
    assert [d.action for d in decisions] == [ReclamationAction.RECLAIMED]
    assert "no response within the grace period" in decisions[0].reason
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.IDLE
    assert job.status == JobStatus.RECLAIMED


# ---- 14/18. Reallocation after reclaim, multiple waiting jobs ------------------------------------

def test_reclaimed_gpu_is_reallocated_through_the_normal_policy():
    scheduler = make_scheduler()
    prompt_for(scheduler)
    waiting = start_job(scheduler, "JB", "B", minute=1)
    assert waiting.status == JobStatus.WAITING

    scheduler.respond_to_prompt("GPU-1", NO, now=at(26))
    decisions = scheduler.try_allocate_all(now=at(26))

    assert [d.job_id for d in decisions] == ["JB"]
    gpu = scheduler.state.get_gpu("GPU-1")
    assert gpu.status == GPUStatus.ACTIVE and gpu.assigned_user_id == "B"
    assert waiting.status == JobStatus.RUNNING

    types = [e.event_type for e in scheduler.state.events if e.gpu_id == "GPU-1"]
    assert types.index(EventType.PROMPT) < types.index(EventType.RESPONSE) < types.index(EventType.RECLAIM)
    assert types.index(EventType.RECLAIM) < len(types) - 1 - types[::-1].index(EventType.ALLOC)


def test_one_reclaimed_gpu_serves_exactly_one_of_several_waiting_jobs():
    scheduler = make_scheduler()
    prompt_for(scheduler)
    waiters = [start_job(scheduler, f"W{i}", f"U{i}", minute=i) for i in (1, 2, 3)]

    scheduler.respond_to_prompt("GPU-1", NO, now=at(26))
    decisions = scheduler.try_allocate_all(now=at(26))

    assert len(decisions) == 1 and decisions[0].job_id == "W1"      # FCFS: equal sizes, earliest wins
    assert [w.status for w in waiters] == [JobStatus.RUNNING, JobStatus.WAITING, JobStatus.WAITING]


# ---- 15/17. Multiple GPUs and users ----------------------------------------------------------------

def test_only_the_idle_gpu_of_several_is_prompted_and_only_its_user():
    scheduler = make_scheduler(3)
    for i, user in enumerate(("U1", "U2", "U3"), start=1):
        start_job(scheduler, f"J{i}", user)
    hits = {}
    for minute in every(5, 0, 25):
        for gpu_id, value in (("GPU-1", 70.0), ("GPU-2", 1.0), ("GPU-3", 70.0)):
            d = scheduler.record_utilization(gpu_id, value, at(minute))
            if d is not None:
                hits[gpu_id] = d
    idle_owner = scheduler.state.get_gpu("GPU-2").assigned_user_id
    assert list(hits) == ["GPU-2"]
    assert hits["GPU-2"].event.user_id == idle_owner
    assert scheduler.state.get_gpu("GPU-1").status == GPUStatus.ACTIVE
    assert scheduler.state.get_gpu("GPU-3").status == GPUStatus.ACTIVE


# ---- 16. Partial multi-GPU reclamation ------------------------------------------------------------------

def test_only_the_underutilized_part_of_a_multi_gpu_job_is_reclaimed():
    scheduler = make_scheduler(4)
    holder = start_job(scheduler, "JA", "A", gpu_count=4)
    assert len(holder.assigned_gpu_ids) == 4
    for gpu_id in ("GPU-1", "GPU-2", "GPU-3", "GPU-4"):
        feed(scheduler, gpu_id, 80.0, [0])
    waiter = start_job(scheduler, "JB", "B", minute=1, gpu_count=3)
    assert waiter.assigned_gpu_ids == []                                   # pool is exhausted

    prompted = set()
    for minute in every(5, 5, 30):
        for gpu_id, value in (("GPU-1", 1.0), ("GPU-2", 1.0), ("GPU-3", 80.0), ("GPU-4", 80.0)):
            if scheduler.record_utilization(gpu_id, value, at(minute)) is not None:
                prompted.add(gpu_id)
    assert prompted == {"GPU-1", "GPU-2"}                                   # busy GPUs never asked about

    for gpu_id in sorted(prompted):
        scheduler.respond_to_prompt(gpu_id, NO, now=at(31))
    scheduler.try_allocate_all(now=at(31))

    assert sorted(holder.assigned_gpu_ids) == ["GPU-3", "GPU-4"]           # A keeps what it actively uses
    assert holder.status == JobStatus.RUNNING                               # partial loss is not job loss
    assert sorted(waiter.assigned_gpu_ids) == ["GPU-1", "GPU-2"]
    assert (waiter.gpu_count, waiter.gpus_still_needed) == (3, 1)           # B still needs one more
    assert waiter.status == JobStatus.WAITING
    assert scheduler.state.get_user("A").assigned_gpu_ids.__len__() == 2
    assert scheduler.allocation_engine.get_gpus_for_user("A") == holder.assigned_gpu_ids or \
        sorted(scheduler.allocation_engine.get_gpus_for_user("A")) == ["GPU-3", "GPU-4"]


# ---- 19/20. Stale-watch safety -----------------------------------------------------------------------------

def test_completed_job_leaves_no_stale_watch_to_reclaim_the_next_holder():
    scheduler = make_scheduler()
    prompt_for(scheduler)                                                   # prompt outstanding at minute 25
    scheduler.complete_job("JA", now=at(26))
    assert not scheduler.reclamation_engine.has_pending_prompt("GPU-1")

    successor = start_job(scheduler, "JB", "B", minute=27)
    assert scheduler.check_reclamation_timeouts(at(40)) == []               # the old grace timer must not fire
    assert scheduler.state.get_gpu("GPU-1").assigned_job_id == "JB"
    assert successor.status == JobStatus.RUNNING


def test_cancelled_job_holds_no_gpu_and_leaves_no_watch():
    scheduler = make_scheduler()
    prompt_for(scheduler)
    waiter = start_job(scheduler, "JB", "B", minute=1)
    scheduler.cancel_job("JB", now=at(2))

    assert waiter.status == JobStatus.CANCELLED and waiter.assigned_gpu_ids == []
    scheduler.respond_to_prompt("GPU-1", NO, now=at(26))
    assert scheduler.try_allocate_all(now=at(26)) == []                     # the cancelled job never gets it
    assert scheduler.state.get_gpu("GPU-1").assigned_job_id is None
    assert not any(e.job_id == "JB" and e.event_type in (EventType.PROMPT, EventType.RECLAIM)
                   for e in scheduler.state.events)


def test_a_new_holder_is_not_judged_by_the_previous_holders_idle_history():
    """Regression: 20 idle minutes under JA, JA completes, JB gets the
    GPU - JB used to be prompted after 5 minutes."""
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    feed(scheduler, "GPU-1", 1.0, every(5, 0, 20))
    scheduler.complete_job("JA", now=at(21))
    start_job(scheduler, "JB", "B", minute=21)

    assert feed(scheduler, "GPU-1", 1.0, [25, 26, 30, 40, 45]) == []
    hits = feed(scheduler, "GPU-1", 1.0, [50])                              # 25 min after JB's first reading (25)
    assert len(hits) == 1 and hits[0][1].event.job_id == "JB"


def test_a_gpu_idle_while_unassigned_does_not_instantly_prompt_its_next_holder():
    """Regression: an hour of idle readings with nobody assigned, then
    a job arrives - the very next reading used to raise a prompt."""
    scheduler = make_scheduler()
    feed(scheduler, "GPU-1", 0.5, every(5, 0, 55))
    start_job(scheduler, "JB", "B", minute=60)
    assert feed(scheduler, "GPU-1", 0.5, every(5, 60, 80)) == []
    assert len(feed(scheduler, "GPU-1", 0.5, [85])) == 1


# ---- 21/22. Maintenance / unavailable ----------------------------------------------------------------------------

def test_maintenance_gpu_is_never_prompted_or_reclaimed():
    scheduler = make_scheduler(2)
    scheduler.set_gpu_maintenance("GPU-2", now=at(0))
    assert feed(scheduler, "GPU-2", 0.0, every(10, 0, 240)) == []
    gpu = scheduler.state.get_gpu("GPU-2")
    assert gpu.status == GPUStatus.MAINTENANCE and not gpu.is_assigned
    assert scheduler.check_reclamation_timeouts(at(300)) == []


def test_unavailable_gpu_is_handled_safely():
    scheduler = make_scheduler(2)
    start_job(scheduler, "JA", "A")
    failed = scheduler.state.get_job("JA").assigned_gpu_ids[0]
    scheduler.handle_gpu_failure(failed, now=at(1))

    assert feed(scheduler, failed, 0.0, every(10, 2, 240)) == []
    gpu = scheduler.state.get_gpu(failed)
    assert gpu.status == GPUStatus.UNAVAILABLE and not gpu.is_assigned
    assert not scheduler.reclamation_engine.has_pending_prompt(failed)


# ---- 23. Repeated cycles ---------------------------------------------------------------------------------------------

def test_repeated_reclaim_cycles_each_start_from_a_clean_baseline():
    scheduler = make_scheduler()
    jobs = [start_job(scheduler, f"J{i}", f"U{i}", minute=i) for i in range(4)]   # J0 runs, J1..J3 wait
    allocated_at = 0.0
    for cycle in range(3):
        holder = f"J{cycle}"
        early = feed(scheduler, "GPU-1", 1.0, every(5, allocated_at + 5, allocated_at + 25))
        assert early == [], f"cycle {cycle}: prompted before a full 25 minutes of this holder's own idleness"
        hits = feed(scheduler, "GPU-1", 1.0, [allocated_at + 30])
        assert len(hits) == 1 and hits[0][1].event.job_id == holder

        scheduler.respond_to_prompt("GPU-1", NO, now=at(allocated_at + 31))
        allocated_at += 31
        scheduler.try_allocate_all(now=at(allocated_at))

    assert [j.status for j in jobs[:3]] == [JobStatus.RECLAIMED] * 3
    assert jobs[3].status == JobStatus.RUNNING
    reclaimed_order = []
    while (event := scheduler.reclamation_engine.undo_last_reclaim()) is not None:
        reclaimed_order.append(event.job_id)
    assert reclaimed_order == ["J2", "J1", "J0"]                            # LIFO stack


# ---- 24/25. Events and ReclaimHistory ---------------------------------------------------------------------------------

def test_events_explain_the_whole_flow():
    scheduler = make_scheduler()
    prompt_for(scheduler)
    start_job(scheduler, "JB", "B", minute=1)
    scheduler.respond_to_prompt("GPU-1", NO, now=at(26))
    scheduler.try_allocate_all(now=at(26))

    events = [e for e in scheduler.state.events if e.gpu_id == "GPU-1"]
    by_type = {}
    for e in events:
        by_type.setdefault(e.event_type, []).append(e)
    for expected in (EventType.PROMPT, EventType.RESPONSE, EventType.RECLAIM, EventType.ALLOC, EventType.STATUS):
        assert expected in by_type, f"missing {expected.value} event"

    reclaim = by_type[EventType.RECLAIM][0]
    assert reclaim.user_id == "A" and reclaim.job_id == "JA" and reclaim.timestamp == at(26)
    assert reclaim.metadata["category"] == EventType.AUTOMATIC_RECLAIM.value
    assert "TIER_1" in reclaim.reason and reclaim.message == "Reclaimed GPU-1 from A"
    assert by_type[EventType.ALLOC][-1].user_id == "B"                        # the reallocation
    assert by_type[EventType.RESPONSE][0].reason == "user responded NO"


def test_reclaim_history_records_previous_and_resulting_state():
    scheduler = make_scheduler()
    prompt_for(scheduler)
    scheduler.respond_to_prompt("GPU-1", NO, now=at(26))

    last = scheduler.reclamation_engine.last_reclaim()
    logged = [e for e in scheduler.state.events if e.event_type == EventType.RECLAIM]
    assert last is logged[-1]
    assert (last.gpu_id, last.user_id, last.job_id, last.timestamp) == ("GPU-1", "A", "JA", at(26))
    assert last.metadata["previous_status"] == GPUStatus.IDLE_WARNING.value
    assert last.metadata["resulting_status"] == GPUStatus.IDLE.value
    assert last.reason


def test_normal_completion_is_never_classified_as_a_reclaim():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    event = scheduler.complete_job("JA", now=at(10))

    assert event.event_type == EventType.JOB_COMPLETION
    assert scheduler.reclamation_engine.last_reclaim() is None
    assert not any(e.event_type in (EventType.RECLAIM, EventType.AUTOMATIC_RECLAIM) for e in scheduler.state.events)


def test_manual_release_and_hardware_failure_are_not_automatic_reclaims():
    scheduler = make_scheduler(2)
    start_job(scheduler, "JA", "A")
    scheduler.force_reclaim(scheduler.state.get_job("JA").assigned_gpu_ids[0], now=at(5))
    start_job(scheduler, "JB", "B", minute=6)
    scheduler.handle_gpu_failure(scheduler.state.get_job("JB").assigned_gpu_ids[0], now=at(7))

    types = {e.event_type for e in scheduler.state.events}
    assert EventType.ADMIN_FORCE_RECLAIM in types and EventType.HARDWARE_FAILURE in types
    assert EventType.AUTOMATIC_RECLAIM not in types
    assert scheduler.reclamation_engine.last_reclaim() is None


# ---- 26/27. Sliding window and irregular samples -------------------------------------------------------------------------

def test_sliding_window_expires_old_samples_including_with_no_new_data():
    window = UtilizationSlidingWindow(timedelta(minutes=30))
    for minute in (0, 10, 20, 30, 45):                                        # irregular spacing
        window.add_observation(UtilizationObservation(1.0, at(minute)))
    assert [o.timestamp for o in window.observations()] == [at(20), at(30), at(45)]   # 0 and 10 expired
    assert window.window_span() == timedelta(minutes=25)

    assert window.evict_expired(at(70)) == 2                                   # ages forward without new data
    assert [o.timestamp for o in window.observations(at(70))] == [at(45)]
    assert window.evict_expired(at(500)) == 1 and window.is_empty()


def test_engine_window_is_bounded_by_the_longest_tier():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    feed(scheduler, "GPU-1", 80.0, every(5, 0, 600))                          # 10 hours of readings
    window = scheduler.reclamation_engine.window_for("GPU-1")
    span = window.window_span()
    assert span <= DEFAULT_RECLAMATION_POLICY.monitoring_window_duration
    assert window.size() <= 2 * 60 * 2.5 / 5 + 5                               # ~30 samples, not 120


def test_sustained_is_judged_by_timestamps_not_by_sample_counts():
    def obs(*minutes, value=1.0):
        return [UtilizationObservation(value, at(m)) for m in minutes]

    duration, threshold = timedelta(minutes=25), 2.0
    assert is_sustained_breach(obs(0, 1, 13, 24, 25), threshold, duration) is True       # 5 irregular samples, 25 min
    assert is_sustained_breach(obs(0, 0.5, 1, 1.5, 2), threshold, duration) is False      # 5 samples, only 2 min
    assert is_sustained_breach(obs(0, 30), threshold, duration) is False                  # a monitoring gap: 1 real sample in window
    assert is_sustained_breach(obs(0, 5, 10, 15, 20, 25, value=1.0)[:-1] + obs(25, value=9.0),
                               threshold, duration) is False                              # latest reading recovered


def test_irregular_sampling_through_the_engine_uses_real_elapsed_time():
    scheduler = make_scheduler()
    start_job(scheduler, "JA", "A")
    assert feed(scheduler, "GPU-1", 1.0, [0, 1, 13, 24]) == []
    hits = feed(scheduler, "GPU-1", 1.0, [25])
    assert len(hits) == 1 and hits[0][0] == 25
