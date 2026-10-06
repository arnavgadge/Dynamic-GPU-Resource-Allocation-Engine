"""The size-disparity reallocation path, wired into the live demo.

Exercised through `SimulationSession` - the same object the API routes
call - so this proves the feature is actually reachable from the running
`interactive_demo` server, not just from a hand-built `Scheduler`.
"""

from engine.models.enums import JobStatus
from engine.reclamation.policy import ConfirmationResponse
from engine.simulation import load_default_registry
from api.session import SimulationSession


def _session(scenario_id: str = "interactive_demo") -> SimulationSession:
    return SimulationSession(load_default_registry(), scenario_id)


def test_interactive_demo_has_size_disparity_switched_on():
    session = _session()
    assert session.simulator.scheduler.size_disparity_ratio == 5.0


def test_scripted_scenarios_keep_size_disparity_off():
    session = _session("gta5_excel")
    assert session.simulator.scheduler.size_disparity_ratio is None


def test_setting_survives_a_reset():
    session = _session()
    session.reset()
    assert session.simulator.scheduler.size_disparity_ratio == 5.0


def test_live_worked_example_long_holder_asked_short_job_waits_then_resumes():
    session = _session()
    # A long holder takes the whole pool (9 + 1 = 10 GPUs, all 60 minutes).
    long_b = session.submit_user_request("user_b", "User B", "train", 9, 60, "MEDIUM")
    long_c = session.submit_user_request("user_c", "User C", "train", 1, 60, "MEDIUM")
    assert long_b.status == JobStatus.RUNNING
    assert long_c.status == JobStatus.RUNNING

    # User A asks for 1 GPU for 2 minutes: a 30x gap -> a holder is asked.
    short_a = session.submit_user_request("user_a", "User A", "quick", 1, 2, "HIGH")
    scheduler = session.simulator.scheduler
    prompted = [g for g in scheduler.state.gpus if scheduler.reclamation_engine.has_pending_prompt(g)]
    assert len(prompted) == 1
    assert short_a.status == JobStatus.WAITING

    # The holder refuses (YES = still using it): A stays queued, nothing forced.
    session.respond_to_prompt(prompted[0], ConfirmationResponse.YES.name)
    assert short_a.status == JobStatus.WAITING
    assert short_a.assigned_gpu_ids == []


def test_live_holder_releases_then_freed_gpu_returns_to_holder_on_completion():
    session = _session()
    long_b = session.submit_user_request("user_b", "User B", "train", 9, 60, "MEDIUM")
    session.submit_user_request("user_c", "User C", "train", 1, 60, "MEDIUM")

    short_a = session.submit_user_request("user_a", "User A", "quick", 1, 2, "HIGH")
    scheduler = session.simulator.scheduler
    prompted = [g for g in scheduler.state.gpus if scheduler.reclamation_engine.has_pending_prompt(g)]
    assert len(prompted) == 1
    gpu_id = prompted[0]

    session.respond_to_prompt(gpu_id, ConfirmationResponse.NO.name)
    assert short_a.status == JobStatus.RUNNING
    assert short_a.assigned_gpu_ids == [gpu_id]

    # A's 2-minute job finishes; the GPU goes back to the holder's deficit
    # automatically through the ordinary allocation policy.
    now = session.simulator.clock.now()
    scheduler.complete_job(short_a.job_id, now=now)
    scheduler.try_allocate_all(now=now)
    assert short_a.status == JobStatus.COMPLETED
    assert scheduler.state.gpus[gpu_id].assigned_user_id != "user_a"
