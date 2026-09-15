"""Phase 10: the admin-only manual-assignment endpoint, and Tests
10/11 from the brief - the User Portal and Admin Console both reading
the one real `SchedulerState`, never a separate copy.
"""

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        test_client.post("/api/scenarios/interactive_demo/load")
        yield test_client


def login(client, username):
    response = client.post("/api/auth/login", json={"username": username})
    assert response.status_code == 200
    return response.json()["token"]


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_manual_assign_requires_admin(client):
    token = login(client, "user_a")
    response = client.post(
        "/api/admin/assign", headers=auth_headers(token),
        json={"gpu_id": "GPU-1", "user_id": "user_a", "display_name": "User A"},
    )
    assert response.status_code == 403


def test_manual_assign_establishes_state_admin_sees(client):
    admin_token = login(client, "admin")
    response = client.post(
        "/api/admin/assign", headers=auth_headers(admin_token),
        json={"gpu_id": "GPU-3", "user_id": "user_c", "display_name": "User C"},
    )
    assert response.status_code == 200
    state = response.json()
    gpu = next(g for g in state["gpus"] if g["gpu_id"] == "GPU-3")
    assert gpu["assigned_user_id"] == "user_c"
    assert gpu["status"] == "ACTIVE"


def test_manual_assign_rejects_an_occupied_gpu(client):
    admin_token = login(client, "admin")
    client.post(
        "/api/admin/assign", headers=auth_headers(admin_token),
        json={"gpu_id": "GPU-1", "user_id": "user_a", "display_name": "User A"},
    )
    response = client.post(
        "/api/admin/assign", headers=auth_headers(admin_token),
        json={"gpu_id": "GPU-1", "user_id": "user_b", "display_name": "User B"},
    )
    assert response.status_code == 422


# -- Tests 10/11: user portal and admin console agree ------------------

def test_10_and_11_user_c_portal_and_admin_console_show_the_same_assignment(client):
    admin_token = login(client, "admin")
    client.post(
        "/api/admin/assign", headers=auth_headers(admin_token),
        json={"gpu_id": "GPU-5", "user_id": "user_c", "display_name": "User C"},
    )
    client.post(
        "/api/admin/assign", headers=auth_headers(admin_token),
        json={"gpu_id": "GPU-6", "user_id": "user_c", "display_name": "User C"},
    )

    admin_state = client.get("/api/state").json()
    admin_gpus_for_c = sorted(g["gpu_id"] for g in admin_state["gpus"] if g["assigned_user_id"] == "user_c")
    assert admin_gpus_for_c == ["GPU-5", "GPU-6"]

    user_c_token = login(client, "user_c")
    portal = client.get("/api/portal/state", headers=auth_headers(user_c_token)).json()
    # Two separate manual assignments create two separate placeholder
    # jobs (one per admin action) - the union of what they hold is
    # what matters for "what GPUs does User C actually have".
    running_jobs = [j for j in portal["my_jobs"] if j["status"] == "RUNNING"]
    held_gpus = sorted(gid for j in running_jobs for gid in j["assigned_gpu_ids"])
    assert held_gpus == ["GPU-5", "GPU-6"]

    # User Portal never leaks GPU-3/other users' data - only its own.
    for forbidden_key in ("gpus", "users", "waiting_queue", "decision_trace"):
        assert forbidden_key not in portal


# -- Full API round trip of the resource-request flow -------------------

def test_full_resource_request_round_trip_through_the_api(client):
    admin_token = login(client, "admin")
    # Occupy 9 of 10 GPUs under user_c, leaving exactly one free.
    user_c_token = login(client, "user_c")
    client.post(
        "/api/requests", headers=auth_headers(user_c_token),
        json={"workload": "existing work", "gpu_count": 9, "estimated_minutes": 500, "priority": "MEDIUM"},
    )

    user_b_token = login(client, "user_b")
    response = client.post(
        "/api/requests", headers=auth_headers(user_b_token),
        json={"workload": "job", "gpu_count": 2, "estimated_minutes": 10, "priority": "HIGH"},
    )
    assert response.status_code == 200
    b_job = next(j for j in response.json()["my_jobs"] if j["name"] == "job")
    assert b_job["status"] == "WAITING"
    assert len(b_job["assigned_gpu_ids"]) == 1

    # User C sees the resource-request prompt on their own portal.
    portal_c = client.get("/api/portal/state", headers=auth_headers(user_c_token)).json()
    prompt = portal_c["pending_prompt"]
    assert prompt is not None
    assert prompt["requested_by_user_id"] == "user_b"

    # User C releases it through the exact same respond endpoint every
    # other confirmation prompt uses.
    resp = client.post(
        "/api/prompt/respond", headers=auth_headers(user_c_token),
        json={"gpu_id": prompt["gpu_id"], "response": "NO"},
    )
    assert resp.status_code == 200

    # User B now holds both GPUs; the admin console agrees.
    portal_b = client.get("/api/portal/state", headers=auth_headers(user_b_token)).json()
    b_job_after = next(j for j in portal_b["my_jobs"] if j["name"] == "job")
    assert b_job_after["status"] == "RUNNING"
    assert len(b_job_after["assigned_gpu_ids"]) == 2

    admin_state = client.get("/api/state").json()
    assert len(admin_state["waiting_queue"]) == 0
    b_gpus_admin = sorted(g["gpu_id"] for g in admin_state["gpus"] if g["assigned_user_id"] == "user_b")
    assert b_gpus_admin == sorted(b_job_after["assigned_gpu_ids"])
