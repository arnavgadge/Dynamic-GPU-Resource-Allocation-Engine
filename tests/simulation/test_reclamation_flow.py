from datetime import datetime, timedelta, timezone

from engine.models.enums import GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.policy import ConfirmationResponse
from engine.simulation.actions import UserResponseAction, UtilizationAction
from engine.simulation.scenario import Scenario
from engine.simulation.simulator import Simulator

START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def one_gpu_scenario(actions) -> Scenario:
    gpu = GPU(gpu_id="GPU-1", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
               assigned_user_id="u1", assigned_job_id="J1")
    running_job = Job(job_id="J1", user_id="u1", name="job", priority=Priority.MEDIUM,
                        estimated_size_minutes=600, status=JobStatus.RUNNING, started_at=START,
                        submitted_at=START, assigned_gpu_ids=["GPU-1"])
    return Scenario(
        scenario_id="test-reclaim", name="test", description="test", start_time=START,
        gpus=[gpu], users=[User(user_id="u1", name="U1", priority=Priority.MEDIUM)],
        initial_jobs=[running_job], actions=actions,
    )


def sustained_breach_readings():
    """Six readings, five minutes apart, spanning exactly the default
    Tier 1 window (25 minutes)."""
    return [UtilizationAction(offset=timedelta(minutes=m), gpu_id="GPU-1", utilization_percent=1.0)
            for m in (0, 5, 10, 15, 20, 25)]


# ------------------------------------------------------------------
# Test 10 - the reclamation scenario reaches the actual threshold
# through simulated time (never real waiting)
# ------------------------------------------------------------------

def test_sustained_breach_is_reached_through_simulated_time_only():
    import time as real_time

    scenario = one_gpu_scenario(sustained_breach_readings())
    sim = Simulator(scenario)

    started = real_time.monotonic()
    sim.advance(timedelta(minutes=25))
    elapsed = real_time.monotonic() - started

    gpu = sim.snapshot().get_gpu("GPU-1")
    assert gpu.status == GPUStatus.IDLE_WARNING
    assert elapsed < 1.0  # nowhere close to 25 real minutes

    prompts = [e for e in sim.snapshot().events if e.event_type.value == "PROMPT"]
    assert len(prompts) == 1


def test_utilization_below_threshold_alone_does_not_trigger_reclamation_early():
    # Only three of the six required readings - not sustained yet.
    scenario = one_gpu_scenario(sustained_breach_readings()[:3])
    sim = Simulator(scenario)
    sim.advance(timedelta(minutes=10))

    assert sim.snapshot().get_gpu("GPU-1").status == GPUStatus.ACTIVE


# ------------------------------------------------------------------
# Test 11 - a YES response flows through the real Reclamation Engine
# ------------------------------------------------------------------

def test_yes_response_flows_through_the_real_engine_and_keeps_the_assignment():
    actions = sustained_breach_readings() + [
        UserResponseAction(offset=timedelta(minutes=26), gpu_id="GPU-1", response=ConfirmationResponse.YES),
    ]
    scenario = one_gpu_scenario(actions)
    sim = Simulator(scenario)
    sim.advance(timedelta(minutes=26))

    state = sim.snapshot()
    assert state.get_gpu("GPU-1").status == GPUStatus.ACTIVE
    assert state.get_gpu("GPU-1").assigned_user_id == "u1"  # kept, not reclaimed
    assert state.get_job("J1").status == JobStatus.RUNNING

    responses = [e for e in state.events if e.event_type.value == "RESPONSE"]
    assert len(responses) == 1
    assert "YES" in responses[0].message


# ------------------------------------------------------------------
# Test 12 - a NO response flows through the real Reclamation Engine
# ------------------------------------------------------------------

def test_no_response_flows_through_the_real_engine_and_reclaims():
    actions = sustained_breach_readings() + [
        UserResponseAction(offset=timedelta(minutes=26), gpu_id="GPU-1", response=ConfirmationResponse.NO),
    ]
    scenario = one_gpu_scenario(actions)
    sim = Simulator(scenario)
    sim.advance(timedelta(minutes=26))

    state = sim.snapshot()
    assert state.get_gpu("GPU-1").status == GPUStatus.IDLE
    assert state.get_gpu("GPU-1").assigned_user_id is None
    assert state.get_job("J1").status == JobStatus.RECLAIMED

    reclaims = [e for e in state.events if e.event_type.value == "RECLAIM"]
    assert len(reclaims) == 1


# ------------------------------------------------------------------
# Test 13 - no-response timeout, driven entirely by simulated time
# ------------------------------------------------------------------

def test_no_response_timeout_auto_reclaims_via_simulated_time():
    scenario = one_gpu_scenario(sustained_breach_readings())  # no UserResponseAction at all
    sim = Simulator(scenario)
    sim.advance(timedelta(minutes=25))
    assert sim.snapshot().get_gpu("GPU-1").status == GPUStatus.IDLE_WARNING

    sim.advance(timedelta(minutes=4))  # 4 min after prompt - within 5 min grace
    assert sim.snapshot().get_gpu("GPU-1").status == GPUStatus.IDLE_WARNING

    sim.advance(timedelta(minutes=2))  # now 6 min after prompt - past the grace period
    state = sim.snapshot()
    assert state.get_gpu("GPU-1").status == GPUStatus.IDLE
    assert state.get_job("J1").status == JobStatus.RECLAIMED

    reclaims = [e for e in state.events if e.event_type.value == "RECLAIM"]
    assert len(reclaims) == 1
    assert "no response" in reclaims[0].reason.lower()


# ------------------------------------------------------------------
# Test 15 / 16 - a reclaimed GPU becomes available, and a new waiting
# job can receive it
# ------------------------------------------------------------------

def test_reclaimed_gpu_becomes_available_for_a_new_waiting_job():
    from engine.simulation.actions import AddJobAction

    new_job = Job(job_id="J2", user_id="u2", name="new work", priority=Priority.MEDIUM,
                   estimated_size_minutes=10, submitted_at=START + timedelta(minutes=1))
    actions = sustained_breach_readings() + [
        AddJobAction(offset=timedelta(minutes=1), job=new_job),
        UserResponseAction(offset=timedelta(minutes=26), gpu_id="GPU-1", response=ConfirmationResponse.NO),
    ]
    scenario = one_gpu_scenario(actions)
    scenario.users.append(User(user_id="u2", name="U2", priority=Priority.MEDIUM))
    sim = Simulator(scenario)

    sim.advance(timedelta(minutes=1))
    assert sim.snapshot().get_job("J2").status == JobStatus.WAITING  # GPU still busy

    sim.advance(timedelta(minutes=25))  # through the sustained breach + response

    state = sim.snapshot()
    assert state.get_job("J2").status == JobStatus.RUNNING
    assert state.get_job("J2").assigned_gpu_id == "GPU-1"
    assert state.get_gpu("GPU-1").assigned_user_id == "u2"
