from datetime import datetime, timedelta, timezone

from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.simulation.actions import AddJobAction, CompleteJobAction, UtilizationAction
from engine.simulation.scenario import Scenario
from engine.simulation.simulator import Simulator

START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def two_gpu_scenario(actions=None, initial_jobs=None) -> Scenario:
    return Scenario(
        scenario_id="test-two-gpu",
        name="test",
        description="test",
        start_time=START,
        gpus=[GPU(gpu_id="GPU-1", total_memory_mb=24_576, status=GPUStatus.IDLE),
              GPU(gpu_id="GPU-2", total_memory_mb=24_576, status=GPUStatus.IDLE)],
        users=[User(user_id="u1", name="U1", priority=Priority.MEDIUM),
               User(user_id="u2", name="U2", priority=Priority.MEDIUM)],
        initial_jobs=initial_jobs or [],
        actions=actions or [],
    )


# ------------------------------------------------------------------
# Test 3 - simulator starts at the expected initial time
# ------------------------------------------------------------------

def test_simulator_starts_at_the_scenario_start_time():
    sim = Simulator(two_gpu_scenario())
    assert sim.clock.now() == START


# ------------------------------------------------------------------
# Test 4 - advancing simulated time never sleeps
# ------------------------------------------------------------------

def test_advance_does_not_sleep_in_real_time():
    import time as real_time

    sim = Simulator(two_gpu_scenario())
    started = real_time.monotonic()
    sim.advance(timedelta(hours=3))
    elapsed = real_time.monotonic() - started

    assert sim.clock.now() == START + timedelta(hours=3)
    assert elapsed < 1.0


# ------------------------------------------------------------------
# Test 5 - job arrival happens at the configured simulated time
# ------------------------------------------------------------------

def test_job_arrives_exactly_at_its_configured_offset_not_before():
    job = Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
               estimated_size_minutes=10, submitted_at=START + timedelta(minutes=5))
    scenario = two_gpu_scenario(actions=[AddJobAction(offset=timedelta(minutes=5), job=job)])
    sim = Simulator(scenario)

    assert sim.snapshot().get_job("J1") is None  # not arrived yet

    sim.advance(timedelta(minutes=4))
    assert sim.snapshot().get_job("J1") is None  # still not due

    sim.advance(timedelta(minutes=1))  # now at +5 minutes
    assert sim.snapshot().get_job("J1") is not None


# ------------------------------------------------------------------
# Test 6 - a utilization observation reaches the actual scheduler
# ------------------------------------------------------------------

def test_utilization_action_reaches_the_real_gpu_model():
    scenario = two_gpu_scenario(actions=[
        UtilizationAction(offset=timedelta(minutes=1), gpu_id="GPU-1", utilization_percent=42.0),
    ])
    sim = Simulator(scenario)
    sim.advance(timedelta(minutes=1))

    gpu = sim.snapshot().get_gpu("GPU-1")
    assert gpu.utilization_percent == 42.0
    assert len(gpu.utilization_history) == 1
    assert gpu.utilization_history[0].utilization_percent == 42.0


# ------------------------------------------------------------------
# Test 7 - the allocation decision is produced by the real Allocation Engine
# ------------------------------------------------------------------

def test_allocation_decision_comes_from_the_real_allocation_engine():
    job = Job(job_id="J1", user_id="u1", name="job", priority=Priority.HIGH,
               estimated_size_minutes=10, submitted_at=START)
    scenario = two_gpu_scenario(initial_jobs=[job])
    sim = Simulator(scenario)

    state = sim.snapshot()
    assert state.get_job("J1").status == JobStatus.RUNNING

    alloc_events = [e for e in state.events if e.event_type == EventType.ALLOC]
    assert len(alloc_events) == 1
    assert "policy" in alloc_events[0].metadata
    assert alloc_events[0].metadata["policy"] in ("FCFS", "SCORE_BASED")


# ------------------------------------------------------------------
# Test 8 - load balancing is produced by the real balancing logic
# ------------------------------------------------------------------

def test_routing_decision_comes_from_the_real_load_balancer():
    job = Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
               estimated_size_minutes=10, submitted_at=START)
    scenario = two_gpu_scenario(initial_jobs=[job])
    sim = Simulator(scenario)

    balance_events = [e for e in sim.snapshot().events if e.event_type == EventType.BALANCE]
    assert len(balance_events) == 2  # "evaluated pool" + "selected"
    assert any("lowest utilization" in e.reason for e in balance_events)


# ------------------------------------------------------------------
# Test 9 - an assigned, underutilized GPU is never stolen
# ------------------------------------------------------------------

def test_assigned_underutilized_gpu_is_not_stolen_for_new_work():
    busy_gpu = GPU(gpu_id="GPU-1", total_memory_mb=24_576, utilization_percent=4.0,
                    status=GPUStatus.ACTIVE, assigned_user_id="u1", assigned_job_id="J-EXISTING")
    idle_gpu = GPU(gpu_id="GPU-2", total_memory_mb=24_576, utilization_percent=50.0, status=GPUStatus.IDLE)
    existing_job = Job(job_id="J-EXISTING", user_id="u1", name="existing", priority=Priority.MEDIUM,
                         estimated_size_minutes=120, status=JobStatus.RUNNING, started_at=START,
                         submitted_at=START, assigned_gpu_ids=["GPU-1"])
    new_job = Job(job_id="J-NEW", user_id="u2", name="new work", priority=Priority.MEDIUM,
                   estimated_size_minutes=10, submitted_at=START)

    scenario = Scenario(
        scenario_id="test-no-steal", name="test", description="test", start_time=START,
        gpus=[busy_gpu, idle_gpu],
        users=[User(user_id="u1", name="U1", priority=Priority.MEDIUM),
               User(user_id="u2", name="U2", priority=Priority.MEDIUM)],
        initial_jobs=[existing_job, new_job],
    )
    sim = Simulator(scenario)
    state = sim.snapshot()

    assert state.get_gpu("GPU-1").assigned_user_id == "u1"  # untouched
    assert state.get_gpu("GPU-1").utilization_percent == 4.0
    assert state.get_job("J-NEW").assigned_gpu_id == "GPU-2"


# ------------------------------------------------------------------
# Test 14 - a completed job releases its GPU
# ------------------------------------------------------------------

def test_completed_job_releases_its_gpu():
    running_job = Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
                        estimated_size_minutes=30, status=JobStatus.RUNNING, started_at=START,
                        submitted_at=START, assigned_gpu_ids=["GPU-1"])
    gpu = GPU(gpu_id="GPU-1", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
               assigned_user_id="u1", assigned_job_id="J1")
    scenario = Scenario(
        scenario_id="test-complete", name="test", description="test", start_time=START,
        gpus=[gpu], users=[User(user_id="u1", name="U1", priority=Priority.MEDIUM)],
        initial_jobs=[running_job],
        actions=[CompleteJobAction(offset=timedelta(minutes=10), job_id="J1")],
    )
    sim = Simulator(scenario)
    sim.advance(timedelta(minutes=10))

    state = sim.snapshot()
    assert state.get_job("J1").status == JobStatus.COMPLETED
    assert state.get_job("J1").assigned_gpu_id is None
    assert state.get_gpu("GPU-1").assigned_user_id is None
    assert state.get_gpu("GPU-1").status == GPUStatus.IDLE
    assert state.get_active_assignment_for_gpu("GPU-1") is None


def test_completing_a_job_lets_a_waiting_job_take_its_gpu():
    running_job = Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
                        estimated_size_minutes=30, status=JobStatus.RUNNING, started_at=START,
                        submitted_at=START, assigned_gpu_ids=["GPU-1"])
    gpu = GPU(gpu_id="GPU-1", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
               assigned_user_id="u1", assigned_job_id="J1")
    waiting_job = Job(job_id="J2", user_id="u2", name="job2", priority=Priority.MEDIUM,
                        estimated_size_minutes=10, submitted_at=START)
    scenario = Scenario(
        scenario_id="test-complete-2", name="test", description="test", start_time=START,
        gpus=[gpu],
        users=[User(user_id="u1", name="U1", priority=Priority.MEDIUM),
               User(user_id="u2", name="U2", priority=Priority.MEDIUM)],
        initial_jobs=[running_job, waiting_job],
        actions=[CompleteJobAction(offset=timedelta(minutes=10), job_id="J1")],
    )
    sim = Simulator(scenario)
    assert sim.snapshot().get_job("J2").status == JobStatus.WAITING

    sim.advance(timedelta(minutes=10))

    assert sim.snapshot().get_job("J2").status == JobStatus.RUNNING
    assert sim.snapshot().get_job("J2").assigned_gpu_id == "GPU-1"


# ------------------------------------------------------------------
# Test 17 - reset restores the scenario's initial state
# ------------------------------------------------------------------

def test_reset_restores_initial_state():
    job = Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
               estimated_size_minutes=10, submitted_at=START + timedelta(minutes=2))
    scenario = two_gpu_scenario(actions=[AddJobAction(offset=timedelta(minutes=2), job=job)])
    sim = Simulator(scenario)

    sim.advance(timedelta(minutes=5))
    assert sim.snapshot().get_job("J1") is not None
    assert sim.clock.now() == START + timedelta(minutes=5)

    sim.reset()

    assert sim.clock.now() == START
    assert sim.snapshot().get_job("J1") is None
    assert sim.snapshot().events == []
    assert sim.has_pending_actions() is True


# ------------------------------------------------------------------
# Test 19 - the event stream contains the expected categories
# ------------------------------------------------------------------

def test_event_stream_contains_expected_categories():
    job = Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
               estimated_size_minutes=10, submitted_at=START)
    scenario = two_gpu_scenario(initial_jobs=[job])
    sim = Simulator(scenario)

    types = {e.event_type for e in sim.snapshot().events}
    assert EventType.BALANCE in types
    assert EventType.ALLOC in types
    assert EventType.STATUS in types
