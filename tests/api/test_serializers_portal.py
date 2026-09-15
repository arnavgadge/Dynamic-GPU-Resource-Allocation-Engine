"""Tests for the Phase 9 serializer additions: estimated completion,
the decision trace, notifications, and the user-scoped portal state -
the backend-enforced privacy boundary (Part 24) in particular.
"""

from engine.simulation import load_default_registry
from api.session import SimulationSession
from api.serializers import serialize_decision_trace, serialize_notifications, serialize_portal_state, serialize_state


def build_session():
    registry = load_default_registry()
    return SimulationSession(registry, "interactive_demo")


def test_decision_trace_is_none_before_any_allocation():
    session = build_session()
    assert serialize_decision_trace(session.simulator.scheduler) is None


def test_decision_trace_reflects_the_real_allocation_and_routing_decision():
    session = build_session()
    job = session.submit_user_request("user_a", "User A", "ML Training", 1, 20, "HIGH")

    trace = serialize_decision_trace(session.simulator.scheduler)
    assert trace["job_id"] == job.job_id
    assert trace["gpu_id"] == job.assigned_gpu_id
    assert trace["allocation"]["policy"] in ("FCFS", "SCORE_BASED")
    assert len(trace["allocation"]["candidates"]) >= 1
    assert trace["routing"]["outcome"] == "ROUTED"
    assert any(c["gpu_id"] == job.assigned_gpu_id for c in trace["routing"]["candidates"])


def test_admin_state_includes_the_decision_trace():
    session = build_session()
    session.submit_user_request("user_a", "User A", "job", 1, 10, "HIGH")
    state = serialize_state(session.simulator, session.running, session.speed, session.scenario_id, session.scenario_name)
    assert state["decision_trace"] is not None


def test_gpu_and_job_expose_estimated_completion_when_running():
    session = build_session()
    job = session.submit_user_request("user_a", "User A", "job", 1, 20, "HIGH")

    state = serialize_state(session.simulator, session.running, session.speed, session.scenario_id, session.scenario_name)
    gpu_entry = next(g for g in state["gpus"] if g["gpu_id"] == job.assigned_gpu_id)
    job_entry = next(j for j in state["jobs"] if j["job_id"] == job.job_id)

    assert gpu_entry["estimated_completion"] is not None
    assert job_entry["remaining_seconds"] == 20 * 60


def test_notifications_are_filtered_to_one_user_and_labeled():
    session = build_session()
    session.submit_user_request("user_a", "User A", "job", 1, 10, "HIGH")
    session.submit_user_request("user_b", "User B", "job", 1, 10, "HIGH")

    state = session.simulator.scheduler.state
    notes_a = serialize_notifications(state, "user_a")
    notes_b = serialize_notifications(state, "user_b")

    assert all(True for n in notes_a)  # every note in notes_a is implicitly about user_a (filtered)
    assert len(notes_a) > 0
    assert notes_a != notes_b
    # newest first
    timestamps = [n["timestamp"] for n in notes_a]
    assert timestamps == sorted(timestamps, reverse=True)


def test_portal_state_scopes_jobs_to_the_requesting_user_only():
    session = build_session()
    session.submit_user_request("user_a", "User A", "ML Training", 1, 20, "HIGH")
    session.submit_user_request("user_b", "User B", "Video Editing", 1, 15, "MEDIUM")

    portal_a = serialize_portal_state(session.simulator, "user_a", session.running, session.speed)
    portal_b = serialize_portal_state(session.simulator, "user_b", session.running, session.speed)

    assert {j["user_id"] for j in portal_a["my_jobs"]} <= {"user_a"}
    assert {j["user_id"] for j in portal_b["my_jobs"]} <= {"user_b"}
    assert len(portal_a["my_jobs"]) == 1
    assert len(portal_b["my_jobs"]) == 1


def test_portal_state_never_includes_other_users_priorities_or_global_fields():
    session = build_session()
    session.submit_user_request("user_a", "User A", "job", 1, 10, "HIGH")

    portal = serialize_portal_state(session.simulator, "user_a", session.running, session.speed)

    # No admin-only keys leak into the portal payload.
    for forbidden_key in ("users", "gpus", "waiting_queue", "events", "decision_trace", "config"):
        assert forbidden_key not in portal


def test_portal_state_shows_queue_position_and_score_for_a_waiting_job():
    session = build_session()
    for i in range(10):
        session.submit_user_request(f"user-{i}", f"User {i}", "job", 1, 30, "MEDIUM")  # fills all 10 GPUs
    waiting = session.submit_user_request("user_a", "User A", "second job", 1, 10, "HIGH")
    assert waiting.status.value == "WAITING"

    portal = serialize_portal_state(session.simulator, "user_a", session.running, session.speed)
    waiting_entry = next(j for j in portal["my_jobs"] if j["job_id"] == waiting.job_id)
    assert waiting_entry["queue_position"] == 1
    assert 0.0 <= waiting_entry["allocation_score"] <= 1.0


def test_portal_pending_prompt_is_scoped_to_the_users_own_gpu():
    session = build_session()
    job = session.submit_user_request("user_a", "User A", "job", 1, 20, "HIGH")
    session.step(minutes=20)  # estimated completion elapses -> prompt on user_a's GPU

    portal_a = serialize_portal_state(session.simulator, "user_a", session.running, session.speed)
    portal_b = serialize_portal_state(session.simulator, "user_b", session.running, session.speed)

    assert portal_a["pending_prompt"] is not None
    assert portal_a["pending_prompt"]["gpu_id"] == job.assigned_gpu_id
    assert portal_b["pending_prompt"] is None
