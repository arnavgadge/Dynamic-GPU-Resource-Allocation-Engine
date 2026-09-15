"""Issue 6, Test 3: a reclamation confirmation for a GPU owned by a
real logged-in demo user must be answerable only from that user's own
User Portal - the Admin Console may observe it, never decide it.
"""

from fastapi.testclient import TestClient

from api.app import app


def login(client, username):
    return client.post("/api/auth/login", json={"username": username}).json()["token"]


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_3_gpu_c_owner_receives_the_prompt_admin_only_observes():
    with TestClient(app) as client:
        client.post("/api/scenarios/interactive_demo/load")
        token_c = login(client, "user_c")
        client.post("/api/requests", headers=auth_headers(token_c), json={
            "workload": "job", "gpu_count": 1, "estimated_minutes": 20, "priority": "HIGH",
        })
        client.post("/api/control/step", json={"minutes": 20})  # estimated completion elapses -> prompt

        gpu_id = client.get("/api/portal/state", headers=auth_headers(token_c)).json()["my_jobs"][0]["assigned_gpu_id"]

        # User C genuinely receives the prompt on their own portal.
        portal_c = client.get("/api/portal/state", headers=auth_headers(token_c)).json()
        assert portal_c["pending_prompt"] is not None
        assert portal_c["pending_prompt"]["gpu_id"] == gpu_id

        # Admin observes the same fact as a status, not a decision.
        admin_state = client.get("/api/state").json()
        admin_prompt = next(p for p in admin_state["pending_prompts"] if p["gpu_id"] == gpu_id)
        assert admin_prompt["owner_user_id"] == "user_c"
        assert admin_prompt["owner_has_portal"] is True

        # Admin cannot answer on User C's behalf - neither unauthenticated...
        no_token_resp = client.post("/api/prompt/respond", json={"gpu_id": gpu_id, "response": "YES"})
        assert no_token_resp.status_code == 403
        # ...nor with an actual ADMIN token.
        admin_token = login(client, "admin")
        admin_resp = client.post(
            "/api/prompt/respond", headers=auth_headers(admin_token), json={"gpu_id": gpu_id, "response": "YES"},
        )
        assert admin_resp.status_code == 403

        # Only User C can actually resolve it.
        own_resp = client.post(
            "/api/prompt/respond", headers=auth_headers(token_c), json={"gpu_id": gpu_id, "response": "YES"},
        )
        assert own_resp.status_code == 200


def test_3_generalizes_to_any_of_the_four_demo_users():
    for username in ("user_a", "user_b", "user_d"):
        with TestClient(app) as client:
            client.post("/api/scenarios/interactive_demo/load")
            token = login(client, username)
            client.post("/api/requests", headers=auth_headers(token), json={
                "workload": "job", "gpu_count": 1, "estimated_minutes": 15, "priority": "MEDIUM",
            })
            client.post("/api/control/step", json={"minutes": 15})
            gpu_id = client.get("/api/portal/state", headers=auth_headers(token)).json()["my_jobs"][0]["assigned_gpu_id"]

            blocked = client.post("/api/prompt/respond", json={"gpu_id": gpu_id, "response": "YES"})
            assert blocked.status_code == 403, f"admin override should be blocked for {username}"

            allowed = client.post(
                "/api/prompt/respond", headers=auth_headers(token), json={"gpu_id": gpu_id, "response": "YES"},
            )
            assert allowed.status_code == 200, f"{username} should be able to answer their own prompt"


def test_legacy_scenario_prompt_reports_no_portal_and_stays_admin_answerable():
    with TestClient(app) as client:
        client.post("/api/scenarios/idle_user/load")
        client.post("/api/control/step", json={"minutes": 25})

        state = client.get("/api/state").json()
        prompt = next(p for p in state["pending_prompts"] if p["gpu_id"] == "GPU-1")
        assert prompt["owner_user_id"] == "frank"
        assert prompt["owner_has_portal"] is False

        response = client.post("/api/prompt/respond", json={"gpu_id": "GPU-1", "response": "NO"})
        assert response.status_code == 200
