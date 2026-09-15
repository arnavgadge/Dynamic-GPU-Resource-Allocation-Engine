"""Tests for the Phase 9 additions to the FastAPI app: login, the
User Portal endpoints, and the ownership-guarded prompt response.
Every test resets to a known state first, exactly like
`tests/api/test_app.py`.
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


# ------------------------------------------------------------------
# Login
# ------------------------------------------------------------------

def test_login_returns_a_token_and_role_for_each_seeded_account(client):
    for username, expected_role in [("admin", "ADMIN"), ("user_a", "USER"), ("user_b", "USER")]:
        response = client.post("/api/auth/login", json={"username": username})
        assert response.status_code == 200
        body = response.json()
        assert body["username"] == username
        assert body["role"] == expected_role
        assert body["token"]


def test_login_rejects_an_unknown_account(client):
    response = client.post("/api/auth/login", json={"username": "not_a_real_account"})
    assert response.status_code == 401


def test_whoami_requires_a_valid_token(client):
    assert client.get("/api/auth/me").status_code == 401

    token = login(client, "user_a")
    response = client.get("/api/auth/me", headers=auth_headers(token))
    assert response.status_code == 200
    assert response.json()["username"] == "user_a"


def test_logout_invalidates_the_token(client):
    token = login(client, "user_a")
    client.post("/api/auth/logout", headers=auth_headers(token))
    assert client.get("/api/auth/me", headers=auth_headers(token)).status_code == 401


# ------------------------------------------------------------------
# GPU request submission - reaches the real backend, no scheduling
# decision made by the request handler itself
# ------------------------------------------------------------------

def test_submitting_a_request_requires_authentication(client):
    response = client.post("/api/requests", json={
        "workload": "ML Training", "gpu_count": 1, "estimated_minutes": 20, "priority": "HIGH",
    })
    assert response.status_code == 401


def test_authenticated_request_is_allocated_immediately_when_a_gpu_is_free(client):
    token = login(client, "user_a")
    response = client.post("/api/requests", headers=auth_headers(token), json={
        "workload": "ML Training", "gpu_count": 1, "estimated_minutes": 20, "priority": "HIGH",
    })
    assert response.status_code == 200
    portal = response.json()
    job = portal["my_jobs"][0]
    assert job["status"] == "RUNNING"
    assert job["assigned_gpu_id"] is not None
    assert job["estimated_completion"] is not None


def test_request_identity_comes_from_the_token_never_the_request_body(client):
    # The request body has no user id field at all - there is nothing
    # for a client to lie about; the identity is 100% server-resolved.
    token_a = login(client, "user_a")
    client.post("/api/requests", headers=auth_headers(token_a), json={
        "workload": "job", "gpu_count": 1, "estimated_minutes": 10, "priority": "MEDIUM",
    })

    admin_state = client.get("/api/state").json()
    job = next(j for j in admin_state["jobs"] if j["name"] == "job")
    assert job["user_id"] == "user_a"


def test_multi_gpu_request_is_accepted_and_really_allocated(client):
    token = login(client, "user_a")
    response = client.post("/api/requests", headers=auth_headers(token), json={
        "workload": "job", "gpu_count": 2, "estimated_minutes": 10, "priority": "MEDIUM",
    })
    assert response.status_code == 200
    job = next(j for j in response.json()["my_jobs"] if j["name"] == "job")
    assert job["status"] == "RUNNING"
    assert len(job["assigned_gpu_ids"]) == 2


def test_full_pool_request_is_rejected_not_faked(client):
    token = login(client, "user_a")
    response = client.post("/api/requests", headers=auth_headers(token), json={
        "workload": "job", "gpu_count": 10, "estimated_minutes": 10, "priority": "MEDIUM",
    })
    assert response.status_code == 422


def test_unavailable_request_enters_the_waiting_queue_visible_to_admin(client):
    token = login(client, "user_a")
    # Fill all ten GPUs in interactive_demo across the four login accounts.
    for username, count in (("user_a", 3), ("user_b", 3), ("user_c", 2), ("user_d", 2)):
        t = login(client, username)
        client.post("/api/requests", headers=auth_headers(t), json={
            "workload": "job", "gpu_count": count, "estimated_minutes": 30, "priority": "MEDIUM",
        })

    response = client.post("/api/requests", headers=auth_headers(token), json={
        "workload": "second job", "gpu_count": 1, "estimated_minutes": 10, "priority": "HIGH",
    })
    portal = response.json()
    waiting_job = next(j for j in portal["my_jobs"] if j["name"] == "second job")
    assert waiting_job["status"] == "WAITING"
    assert waiting_job["queue_position"] == 1
    assert 0.0 <= waiting_job["allocation_score"] <= 1.0

    admin_state = client.get("/api/state").json()
    assert len(admin_state["waiting_queue"]) == 1


# ------------------------------------------------------------------
# Portal state privacy (Part 24/47)
# ------------------------------------------------------------------

def test_portal_state_requires_authentication(client):
    assert client.get("/api/portal/state").status_code == 401


def test_portal_state_only_shows_the_callers_own_jobs(client):
    token_a = login(client, "user_a")
    token_b = login(client, "user_b")
    client.post("/api/requests", headers=auth_headers(token_a), json={
        "workload": "A's job", "gpu_count": 1, "estimated_minutes": 10, "priority": "HIGH",
    })
    client.post("/api/requests", headers=auth_headers(token_b), json={
        "workload": "B's job", "gpu_count": 1, "estimated_minutes": 10, "priority": "MEDIUM",
    })

    portal_a = client.get("/api/portal/state", headers=auth_headers(token_a)).json()
    portal_b = client.get("/api/portal/state", headers=auth_headers(token_b)).json()

    assert [j["name"] for j in portal_a["my_jobs"]] == ["A's job"]
    assert [j["name"] for j in portal_b["my_jobs"]] == ["B's job"]
    for forbidden in ("users", "gpus", "waiting_queue", "events", "decision_trace"):
        assert forbidden not in portal_a


# ------------------------------------------------------------------
# Prompt response ownership (Part 47) - admin path stays unrestricted
# ------------------------------------------------------------------

def test_user_can_respond_to_a_prompt_on_their_own_gpu():
    with TestClient(app) as client:
        client.post("/api/scenarios/interactive_demo/load")
        token = login(client, "user_a")
        client.post("/api/requests", headers=auth_headers(token), json={
            "workload": "job", "gpu_count": 1, "estimated_minutes": 20, "priority": "HIGH",
        })
        client.post("/api/control/step", json={"minutes": 20})  # estimated completion elapses -> prompt

        response = client.post(
            "/api/prompt/respond", headers=auth_headers(token),
            json={"gpu_id": client.get("/api/portal/state", headers=auth_headers(token)).json()["my_jobs"][0]["assigned_gpu_id"], "response": "YES"},
        )
        assert response.status_code == 200
        assert "my_jobs" in response.json()  # got back the portal shape, not the admin one


def test_user_cannot_respond_to_a_prompt_on_someone_elses_gpu():
    with TestClient(app) as client:
        client.post("/api/scenarios/interactive_demo/load")
        token_a = login(client, "user_a")
        token_b = login(client, "user_b")
        client.post("/api/requests", headers=auth_headers(token_a), json={
            "workload": "job", "gpu_count": 1, "estimated_minutes": 20, "priority": "HIGH",
        })
        client.post("/api/control/step", json={"minutes": 20})

        gpu_id = client.get("/api/portal/state", headers=auth_headers(token_a)).json()["my_jobs"][0]["assigned_gpu_id"]
        response = client.post(
            "/api/prompt/respond", headers=auth_headers(token_b), json={"gpu_id": gpu_id, "response": "NO"},
        )
        assert response.status_code == 403


def test_admin_or_no_token_cannot_answer_on_behalf_of_a_logged_in_user(client):
    # Issue 6: once a GPU belongs to a real logged-in demo account,
    # the admin override no longer applies to it - only that user's
    # own token may answer their confirmation prompt.
    token = login(client, "user_a")
    client.post("/api/requests", headers=auth_headers(token), json={
        "workload": "job", "gpu_count": 1, "estimated_minutes": 20, "priority": "HIGH",
    })
    client.post("/api/control/step", json={"minutes": 20})

    gpu_id = client.get("/api/portal/state", headers=auth_headers(token)).json()["my_jobs"][0]["assigned_gpu_id"]
    # No Authorization header at all - the old Admin Console behavior.
    response = client.post("/api/prompt/respond", json={"gpu_id": gpu_id, "response": "NO"})
    assert response.status_code == 403

    admin_token = login(client, "admin")
    admin_response = client.post(
        "/api/prompt/respond", headers=auth_headers(admin_token), json={"gpu_id": gpu_id, "response": "NO"},
    )
    assert admin_response.status_code == 403

    # The actual owner can still answer normally.
    own_response = client.post(
        "/api/prompt/respond", headers=auth_headers(token), json={"gpu_id": gpu_id, "response": "NO"},
    )
    assert own_response.status_code == 200


def test_admin_override_still_works_for_scenarios_with_no_real_login(client):
    # A classic scenario's synthetic user ("frank") has no User
    # Portal at all - the admin override must still work for those,
    # exactly as before Issue 6.
    client.post("/api/scenarios/idle_user/load")
    client.post("/api/control/step", json={"minutes": 25})  # sustained breach -> prompt on GPU-1

    state = client.get("/api/state").json()
    assert any(p["gpu_id"] == "GPU-1" for p in state["pending_prompts"])

    response = client.post("/api/prompt/respond", json={"gpu_id": "GPU-1", "response": "NO"})
    assert response.status_code == 200
    assert "gpus" in response.json()
