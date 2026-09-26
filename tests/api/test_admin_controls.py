"""Day 11: the administrative control REST endpoints - thin, admin-
only routes over the `Scheduler` operations already covered directly
in `tests/test_admin_controls.py`. This file only checks the API
layer's own job: admin-only access, request validation, correct HTTP
status mapping, and that the resulting admin state (and hence what
every connected dashboard would see) reflects the operation - no
scheduling logic lives in any handler under test here.
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


def admin(client):
    return auth_headers(login(client, "admin"))


ADMIN_ROUTES = [
    ("/api/admin/cancel-job", {"job_id": "NOPE"}),
    ("/api/admin/force-reclaim", {"gpu_id": "GPU-1"}),
    ("/api/admin/change-priority", {"job_id": "NOPE", "priority": "HIGH"}),
    ("/api/admin/gpu/maintenance/enable", {"gpu_id": "GPU-1"}),
    ("/api/admin/gpu/maintenance/disable", {"gpu_id": "GPU-1"}),
    ("/api/admin/gpu/failure", {"gpu_id": "GPU-1"}),
    ("/api/admin/gpu/recover", {"gpu_id": "GPU-1"}),
]


@pytest.mark.parametrize("route,body", ADMIN_ROUTES)
def test_every_admin_control_route_requires_admin(client, route, body):
    token = login(client, "user_a")
    response = client.post(route, headers=auth_headers(token), json=body)
    assert response.status_code == 403


@pytest.mark.parametrize("route,body", ADMIN_ROUTES)
def test_every_admin_control_route_rejects_no_token(client, route, body):
    response = client.post(route, json=body)
    assert response.status_code == 401


def _submit(client, token, workload="job", gpu_count=1, minutes=10, priority="MEDIUM"):
    response = client.post(
        "/api/requests", headers=auth_headers(token),
        json={"workload": workload, "gpu_count": gpu_count, "estimated_minutes": minutes, "priority": priority},
    )
    assert response.status_code == 200
    return next(j for j in response.json()["my_jobs"] if j["name"] == workload)


def test_admin_cancel_job_via_api(client):
    admin_headers = admin(client)
    # Fill the entire 10-GPU pool (max single request is 9) so the
    # next request genuinely waits.
    _submit(client, login(client, "user_a"), workload="filler-a", gpu_count=9, minutes=9999)
    _submit(client, login(client, "user_c"), workload="filler-c", gpu_count=1, minutes=9999)
    waiter_token = login(client, "user_b")
    waiting_job = _submit(client, waiter_token, workload="wait-job", gpu_count=1)
    assert waiting_job["status"] == "WAITING"

    response = client.post("/api/admin/cancel-job", headers=admin_headers, json={"job_id": waiting_job["job_id"]})
    assert response.status_code == 200
    state = response.json()
    cancelled = next(j for j in state["waiting_queue"] if j["job_id"] == waiting_job["job_id"]) if any(
        j["job_id"] == waiting_job["job_id"] for j in state["waiting_queue"]
    ) else None
    assert cancelled is None  # no longer in the waiting queue
    assert any(e["event_type"] == "JOB_CANCELLED" for e in state["events"])


def test_admin_cancel_job_unknown_id_returns_404(client):
    response = client.post("/api/admin/cancel-job", headers=admin(client), json={"job_id": "NOPE"})
    assert response.status_code == 404


def test_admin_force_reclaim_via_api(client):
    admin_headers = admin(client)
    holder_token = login(client, "user_a")
    _submit(client, holder_token, workload="held", gpu_count=1)

    state = client.get("/api/state").json()
    gpu_id = next(g["gpu_id"] for g in state["gpus"] if g["assigned_user_id"] == "user_a")

    response = client.post("/api/admin/force-reclaim", headers=admin_headers, json={"gpu_id": gpu_id})
    assert response.status_code == 200
    result = response.json()
    reclaimed_gpu = next(g for g in result["gpus"] if g["gpu_id"] == gpu_id)
    assert reclaimed_gpu["assigned_user_id"] is None
    assert any(e["event_type"] == "ADMIN_FORCE_RECLAIM" for e in result["events"])


def test_admin_force_reclaim_unassigned_gpu_returns_422(client):
    response = client.post("/api/admin/force-reclaim", headers=admin(client), json={"gpu_id": "GPU-1"})
    assert response.status_code == 422


def test_admin_force_reclaim_unknown_gpu_returns_404(client):
    response = client.post("/api/admin/force-reclaim", headers=admin(client), json={"gpu_id": "GPU-999"})
    assert response.status_code == 404


def test_admin_change_priority_via_api(client):
    admin_headers = admin(client)
    _submit(client, login(client, "user_a"), workload="filler-a", gpu_count=9, minutes=9999)
    _submit(client, login(client, "user_c"), workload="filler-c", gpu_count=1, minutes=9999)
    waiter_token = login(client, "user_b")
    waiting_job = _submit(client, waiter_token, workload="wait-job", gpu_count=1, priority="LOW")
    assert waiting_job["status"] == "WAITING"

    response = client.post(
        "/api/admin/change-priority", headers=admin_headers,
        json={"job_id": waiting_job["job_id"], "priority": "CRITICAL"},
    )
    assert response.status_code == 200
    result = response.json()
    assert any(e["event_type"] == "PRIORITY_CHANGED" for e in result["events"])

    portal = client.get("/api/portal/state", headers=auth_headers(waiter_token)).json()
    updated = next(j for j in portal["my_jobs"] if j["job_id"] == waiting_job["job_id"])
    assert updated["priority"] == "CRITICAL"


def test_admin_change_priority_rejects_running_job(client):
    holder_token = login(client, "user_a")
    running_job = _submit(client, holder_token, workload="running")
    response = client.post(
        "/api/admin/change-priority", headers=admin(client),
        json={"job_id": running_job["job_id"], "priority": "CRITICAL"},
    )
    assert response.status_code == 422


def test_admin_maintenance_enable_and_disable_via_api(client):
    admin_headers = admin(client)
    response = client.post("/api/admin/gpu/maintenance/enable", headers=admin_headers, json={"gpu_id": "GPU-1"})
    assert response.status_code == 200
    state = response.json()
    gpu = next(g for g in state["gpus"] if g["gpu_id"] == "GPU-1")
    assert gpu["status"] == "MAINTENANCE"
    assert any(e["event_type"] == "GPU_MAINTENANCE_ENABLED" for e in state["events"])

    response = client.post("/api/admin/gpu/maintenance/disable", headers=admin_headers, json={"gpu_id": "GPU-1"})
    assert response.status_code == 200
    state = response.json()
    gpu = next(g for g in state["gpus"] if g["gpu_id"] == "GPU-1")
    assert gpu["status"] in ("IDLE", "ACTIVE")
    assert any(e["event_type"] == "GPU_MAINTENANCE_DISABLED" for e in state["events"])


def test_admin_maintenance_enable_rejects_an_allocated_gpu(client):
    holder_token = login(client, "user_a")
    job = _submit(client, holder_token)
    response = client.post(
        "/api/admin/gpu/maintenance/enable", headers=admin(client), json={"gpu_id": job["assigned_gpu_ids"][0]},
    )
    assert response.status_code == 422


def test_admin_gpu_failure_and_recovery_round_trip_via_api(client):
    admin_headers = admin(client)
    response = client.post("/api/admin/gpu/failure", headers=admin_headers, json={"gpu_id": "GPU-1"})
    assert response.status_code == 200
    state = response.json()
    gpu = next(g for g in state["gpus"] if g["gpu_id"] == "GPU-1")
    assert gpu["status"] == "UNAVAILABLE"
    assert any(e["event_type"] == "HARDWARE_FAILURE" for e in state["events"])

    response = client.post("/api/admin/gpu/recover", headers=admin_headers, json={"gpu_id": "GPU-1"})
    assert response.status_code == 200
    state = response.json()
    gpu = next(g for g in state["gpus"] if g["gpu_id"] == "GPU-1")
    assert gpu["status"] in ("IDLE", "ACTIVE")
    assert any(e["event_type"] == "GPU_RECOVERED" for e in state["events"])


def test_admin_recover_rejects_a_healthy_gpu(client):
    response = client.post("/api/admin/gpu/recover", headers=admin(client), json={"gpu_id": "GPU-1"})
    assert response.status_code == 422
