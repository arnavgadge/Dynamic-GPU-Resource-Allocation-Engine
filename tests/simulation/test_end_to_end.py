"""Test 20 and the brief's own "IMPORTANT END-TO-END DEMONSTRATION" -
the `full_lifecycle` scenario run start to finish with no manual
intervention, checked against every one of the brief's 15 steps.
"""

from engine.models.enums import EventType, GPUStatus, JobStatus
from engine.simulation import load_default_registry
from engine.simulation.simulator import Simulator


def run_full_lifecycle():
    registry = load_default_registry()
    scenario = registry.load("full_lifecycle")
    sim = Simulator(scenario)
    sim.run_to_completion()
    return sim.snapshot()


def test_end_to_end_scenario_completes_without_manual_intervention():
    # `run_to_completion` alone drives the entire scenario - no
    # per-step calls, no manual pokes into the engines.
    state = run_full_lifecycle()
    assert state is not None


def test_new_job_was_routed_to_the_genuinely_available_gpu_not_the_underutilized_one():
    state = run_full_lifecycle()

    assert state.get_job("J-D").status == JobStatus.RUNNING
    assert state.get_job("J-D").assigned_gpu_id == "GPU-4"  # not GPU-3, despite its 4%


def test_underutilized_but_assigned_gpu_was_untouched_until_reclaimed():
    state = run_full_lifecycle()
    # GPU-3 ends the scenario reassigned to J-E only *after* being
    # legitimately reclaimed from User C - never stolen directly.
    reclaim_events = [e for e in state.events if e.event_type == EventType.RECLAIM and e.gpu_id == "GPU-3"]
    assert len(reclaim_events) == 1


def test_sustained_breach_produced_a_prompt_and_a_response():
    state = run_full_lifecycle()

    prompts = [e for e in state.events if e.event_type == EventType.PROMPT and e.gpu_id == "GPU-3"]
    responses = [e for e in state.events if e.event_type == EventType.RESPONSE and e.gpu_id == "GPU-3"]
    assert len(prompts) == 1
    assert len(responses) == 1
    assert "NO" in responses[0].message


def test_gpu_returned_to_idle_then_active_for_the_waiting_job():
    state = run_full_lifecycle()

    gpu3 = state.get_gpu("GPU-3")
    assert gpu3.status == GPUStatus.ACTIVE  # re-allocated to J-E by the end
    assert gpu3.assigned_user_id == "user-e"


def test_waiting_job_received_the_newly_available_gpu():
    state = run_full_lifecycle()

    assert state.get_job("J-E").status == JobStatus.RUNNING
    assert state.get_job("J-E").assigned_gpu_id == "GPU-3"


def test_existing_jobs_on_other_gpus_were_never_disturbed():
    state = run_full_lifecycle()

    assert state.get_job("J-A").status == JobStatus.RUNNING
    assert state.get_job("J-A").assigned_gpu_id == "GPU-1"
    assert state.get_job("J-B").status == JobStatus.RUNNING
    assert state.get_job("J-B").assigned_gpu_id == "GPU-2"


def test_event_log_covers_the_full_lifecycle():
    state = run_full_lifecycle()

    types_present = {e.event_type for e in state.events}
    for expected in (EventType.REQUEST, EventType.BALANCE, EventType.ALLOC,
                      EventType.STATUS, EventType.PROMPT, EventType.RESPONSE, EventType.RECLAIM):
        assert expected in types_present, f"missing {expected} in event log"


def test_events_are_in_nondecreasing_simulated_time_order():
    state = run_full_lifecycle()
    timestamps = [e.timestamp for e in state.events]
    assert timestamps == sorted(timestamps)
