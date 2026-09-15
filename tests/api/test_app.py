"""Tests for the FastAPI adapter (`api/app.py`).

Each test resets the shared session to a known scenario first - the
session is a single, process-wide "live simulator" (exactly one
dashboard's worth of state, matching the project's own "backend is
the source of truth" rule), so tests must not assume a particular
starting scenario left over from a previous test.
"""

import pytest
from fastapi.testclient import TestClient

from api.app import app

REQUIRED_SCENARIO_IDS = {"gta5_excel", "ml_video", "multiple_ml", "idle_user", "imbalance", "full_lifecycle"}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        test_client.post("/api/scenarios/gta5_excel/load")
        yield test_client


# ------------------------------------------------------------------
# Scenario list / load
# ------------------------------------------------------------------

def test_scenario_list_includes_every_required_scenario(client):
    response = client.get("/api/scenarios")
    assert response.status_code == 200

    ids = {entry["scenario_id"] for entry in response.json()}
    assert REQUIRED_SCENARIO_IDS.issubset(ids)
    for entry in response.json():
        assert entry["name"]
        assert entry["description"]


def test_loading_a_scenario_returns_its_initial_state(client):
    response = client.post("/api/scenarios/idle_user/load")
    assert response.status_code == 200

    state = response.json()
    assert state["simulation"]["scenario_id"] == "idle_user"
    assert any(gpu["gpu_id"] == "GPU-1" for gpu in state["gpus"])


def test_loading_an_unknown_scenario_returns_404(client):
    response = client.post("/api/scenarios/does-not-exist/load")
    assert response.status_code == 404


# ------------------------------------------------------------------
# State endpoint returns a valid scheduler state
# ------------------------------------------------------------------

def test_state_endpoint_returns_the_full_expected_shape(client):
    response = client.get("/api/state")
    assert response.status_code == 200
    state = response.json()

    for key in ("simulated_time", "engine_status", "gpus", "users", "jobs",
                 "waiting_queue", "events", "pending_prompts", "config", "simulation"):
        assert key in state

    assert len(state["gpus"]) >= 1
    assert len(state["users"]) >= 1


def test_gpu_count_reflects_the_loaded_scenario_not_a_hardcoded_number(client):
    two_gpu_state = client.get("/api/state").json()
    assert len(two_gpu_state["gpus"]) == 2

    client.post("/api/scenarios/multiple_ml/load")
    four_gpu_state = client.get("/api/state").json()
    assert len(four_gpu_state["gpus"]) == 4


# ------------------------------------------------------------------
# Scenario reset
# ------------------------------------------------------------------

def test_reset_restores_the_scenario_to_its_initial_state(client):
    client.post("/api/control/step", json={"minutes": 2})
    stepped_time = client.get("/api/state").json()["simulated_time"]

    response = client.post("/api/control/reset")
    assert response.status_code == 200
    reset_time = response.json()["simulated_time"]

    assert reset_time != stepped_time
    assert client.get("/api/state").json()["events"] != []  # scenario's own initial events replay


# ------------------------------------------------------------------
# Simulation controls
# ------------------------------------------------------------------

def test_start_and_pause_toggle_running_state(client):
    response = client.post("/api/control/start")
    assert response.json()["simulation"]["running"] is True

    response = client.post("/api/control/pause")
    assert response.json()["simulation"]["running"] is False


def test_step_advances_simulated_time_deterministically(client):
    before = client.get("/api/state").json()["simulated_time"]
    response = client.post("/api/control/step", json={"minutes": 5})
    after = response.json()["simulated_time"]

    assert after != before


def test_speed_configuration_reaches_the_simulator(client):
    response = client.post("/api/control/speed", json={"speed": 10})
    assert response.status_code == 200
    assert response.json()["simulation"]["speed"] == 10


def test_invalid_speed_is_rejected_safely(client):
    response = client.post("/api/control/speed", json={"speed": 3})
    assert response.status_code == 422
    # Rejected, not silently clamped or applied.
    assert client.get("/api/state").json()["simulation"]["speed"] != 3


# ------------------------------------------------------------------
# Prompt response reaches the real Reclamation Engine
# ------------------------------------------------------------------

def test_prompt_response_reaches_the_real_reclamation_engine(client):
    client.post("/api/scenarios/idle_user/load")
    client.post("/api/control/step", json={"minutes": 25})

    state = client.get("/api/state").json()
    gpu1 = next(g for g in state["gpus"] if g["gpu_id"] == "GPU-1")
    assert gpu1["status"] == "IDLE_WARNING"
    assert gpu1["has_pending_prompt"] is True
    assert len(state["pending_prompts"]) == 1

    response = client.post("/api/prompt/respond", json={"gpu_id": "GPU-1", "response": "NO"})
    assert response.status_code == 200
    new_state = response.json()

    reclaim_events = [e for e in new_state["events"] if e["event_type"] == "RECLAIM"]
    assert len(reclaim_events) == 1
    assert new_state["pending_prompts"] == []


def test_yes_response_keeps_the_assignment():
    with TestClient(app) as client:
        client.post("/api/scenarios/idle_user/load")
        client.post("/api/control/step", json={"minutes": 25})

        response = client.post("/api/prompt/respond", json={"gpu_id": "GPU-1", "response": "YES"})
        state = response.json()

        gpu1 = next(g for g in state["gpus"] if g["gpu_id"] == "GPU-1")
        assert gpu1["status"] == "ACTIVE"
        assert gpu1["assigned_user_id"] == "frank"


def test_responding_to_a_gpu_with_no_pending_prompt_is_rejected(client):
    response = client.post("/api/prompt/respond", json={"gpu_id": "GPU-1", "response": "NO"})
    assert response.status_code == 400


def test_responding_to_an_unknown_gpu_is_rejected(client):
    response = client.post("/api/prompt/respond", json={"gpu_id": "NOT-A-GPU", "response": "NO"})
    assert response.status_code == 404


def test_invalid_response_value_is_rejected_by_validation(client):
    response = client.post("/api/prompt/respond", json={"gpu_id": "GPU-1", "response": "MAYBE"})
    assert response.status_code == 422


# ------------------------------------------------------------------
# Waiting queue exposes the real backend-computed allocation score
# ------------------------------------------------------------------

def test_waiting_queue_scores_come_from_the_real_allocation_engine(client):
    client.post("/api/scenarios/ml_video/load")
    state = client.get("/api/state").json()

    assert len(state["waiting_queue"]) >= 1
    for entry in state["waiting_queue"]:
        assert "allocation_score" in entry
        assert 0.0 <= entry["allocation_score"] <= 1.0


# ------------------------------------------------------------------
# WebSocket sends state and reflects command effects
# ------------------------------------------------------------------

def test_websocket_sends_initial_state_on_connect(client):
    with client.websocket_connect("/ws/state") as websocket:
        message = websocket.receive_json()
    assert message["type"] == "state"
    assert "gpus" in message["payload"]


def test_websocket_broadcasts_after_a_control_command():
    with TestClient(app) as client:
        client.post("/api/scenarios/gta5_excel/load")
        with client.websocket_connect("/ws/state") as websocket:
            initial = websocket.receive_json()
            before_time = initial["payload"]["simulated_time"]

            client.post("/api/control/step", json={"minutes": 3})

            update = websocket.receive_json()
            assert update["type"] == "state"
            assert update["payload"]["simulated_time"] != before_time
