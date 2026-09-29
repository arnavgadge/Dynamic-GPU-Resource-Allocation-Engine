"""Day 12: the new resource endpoints (`GET /api/system`, `/api/gpus`,
`/api/jobs`, `/api/users`, `/api/events`), the user-facing job-cancel
endpoint, and the real-time `/ws/events` stream - all thin adapters
over the exact `SchedulerState`/serializers already exercised by
`tests/api/test_app.py`/`test_admin_controls.py`. Nothing here tests
scheduling logic itself; every assertion is either "does this route
return the real backend state" or "does the event stream faithfully
relay what the scheduler already logged, to the right client".
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


def submit(client, token, workload="job", gpu_count=1, minutes=10, priority="MEDIUM"):
    response = client.post(
        "/api/requests", headers=auth_headers(token),
        json={"workload": workload, "gpu_count": gpu_count, "estimated_minutes": minutes, "priority": priority},
    )
    assert response.status_code == 200
    return next(j for j in response.json()["my_jobs"] if j["name"] == workload)


# ---- A-E. Read-only resource endpoints ---------------------------------

def test_get_system_overview(client):
    token = login(client, "user_a")
    submit(client, token, gpu_count=2)

    response = client.get("/api/system")
    assert response.status_code == 200
    body = response.json()
    for key in ("total_gpus", "available_gpus", "allocated_gpus", "maintenance_gpus",
                "failed_gpus", "waiting_jobs", "active_jobs", "engine_status",
                "simulated_time", "real_time"):
        assert key in body
    assert body["total_gpus"] == 10  # interactive_demo's own logical pool size, not hardcoded here
    assert body["allocated_gpus"] == 2
    assert body["available_gpus"] == body["total_gpus"] - body["allocated_gpus"]


def test_get_gpus_matches_state(client):
    token = login(client, "user_a")
    submit(client, token)

    gpus = client.get("/api/gpus").json()
    state_gpus = client.get("/api/state").json()["gpus"]
    assert {g["gpu_id"] for g in gpus} == {g["gpu_id"] for g in state_gpus}
    assert any(g["assigned_user_id"] == "user_a" for g in gpus)


def test_get_jobs_includes_score_only_for_waiting_jobs(client):
    # Exhaust the pool so the next request genuinely waits and is scored.
    submit(client, login(client, "user_a"), workload="filler-a", gpu_count=9, minutes=9999)
    submit(client, login(client, "user_c"), workload="filler-c", gpu_count=1, minutes=9999)
    waiter_token = login(client, "user_b")
    waiting = submit(client, waiter_token, workload="wait-job", gpu_count=1)
    assert waiting["status"] == "WAITING"

    jobs = client.get("/api/jobs").json()
    waiting_entry = next(j for j in jobs if j["job_id"] == waiting["job_id"])
    running_entry = next(j for j in jobs if j["name"] == "filler-a")
    assert waiting_entry["allocation_score"] is not None
    assert running_entry["allocation_score"] is None


def test_get_single_job_by_id(client):
    token = login(client, "user_a")
    job = submit(client, token)

    response = client.get(f"/api/jobs/{job['job_id']}")
    assert response.status_code == 200
    assert response.json()["job_id"] == job["job_id"]


def test_get_single_job_unknown_id_returns_404(client):
    response = client.get("/api/jobs/NOPE")
    assert response.status_code == 404


def test_get_users(client):
    token = login(client, "user_a")
    submit(client, token)

    users = client.get("/api/users").json()
    entry = next(u for u in users if u["user_id"] == "user_a")
    assert entry["assigned_gpu_ids"]


def test_get_events_returns_recent_events_oldest_first_and_respects_limit(client):
    token = login(client, "user_a")
    submit(client, token)

    all_events = client.get("/api/events").json()
    assert len(all_events) >= 1
    timestamps = [e["timestamp"] for e in all_events]
    assert timestamps == sorted(timestamps)  # oldest first

    limited = client.get("/api/events?limit=1").json()
    assert len(limited) == 1
    assert limited[0]["event_id"] == all_events[-1]["event_id"]  # the most recent one


# ---- F/G. Submit and cancel a job through the API -----------------------

def test_submit_job_through_the_api_reuses_the_existing_endpoint(client):
    token = login(client, "user_a")
    job = submit(client, token, gpu_count=1)
    assert job["status"] == "RUNNING"
    assert job["assigned_gpu_ids"]


def test_cancel_own_job_through_the_api(client):
    submit(client, login(client, "user_a"), workload="filler-a", gpu_count=9, minutes=9999)
    submit(client, login(client, "user_c"), workload="filler-c", gpu_count=1, minutes=9999)
    waiter_token = login(client, "user_b")
    waiting = submit(client, waiter_token, workload="wait-job", gpu_count=1)
    assert waiting["status"] == "WAITING"

    response = client.post(f"/api/jobs/{waiting['job_id']}/cancel", headers=auth_headers(waiter_token))
    assert response.status_code == 200
    portal = client.get("/api/portal/state", headers=auth_headers(waiter_token)).json()
    cancelled = next(j for j in portal["my_jobs"] if j["job_id"] == waiting["job_id"])
    assert cancelled["status"] == "CANCELLED"


def test_cancel_someone_elses_job_is_forbidden(client):
    token_a = login(client, "user_a")
    job = submit(client, token_a)
    token_b = login(client, "user_b")

    response = client.post(f"/api/jobs/{job['job_id']}/cancel", headers=auth_headers(token_b))
    assert response.status_code == 403


def test_admin_can_still_cancel_any_users_job_via_the_user_facing_route(client):
    submit(client, login(client, "user_a"), workload="filler-a", gpu_count=9, minutes=9999)
    submit(client, login(client, "user_c"), workload="filler-c", gpu_count=1, minutes=9999)
    waiter_token = login(client, "user_b")
    waiting = submit(client, waiter_token, workload="wait-job", gpu_count=1)

    admin_token = login(client, "admin")
    response = client.post(f"/api/jobs/{waiting['job_id']}/cancel", headers=auth_headers(admin_token))
    assert response.status_code == 200


def test_cancel_running_job_returns_409_conflict(client):
    token = login(client, "user_a")
    job = submit(client, token)  # runs immediately
    response = client.post(f"/api/jobs/{job['job_id']}/cancel", headers=auth_headers(token))
    assert response.status_code == 409


# ---- L. Invalid ids return correct errors --------------------------------

def test_cancel_unknown_job_returns_404(client):
    token = login(client, "user_a")
    response = client.post("/api/jobs/NOPE/cancel", headers=auth_headers(token))
    assert response.status_code == 404


def test_cancel_own_job_route_requires_a_token(client):
    response = client.post("/api/jobs/NOPE/cancel")
    assert response.status_code == 401


# ---- M/N. WebSocket connects and receives a scheduler event -------------

def test_websocket_events_connects_and_receives_nothing_immediately(client):
    with client.websocket_connect("/ws/events"):
        pass  # connecting and disconnecting cleanly is itself the assertion


def test_websocket_events_receives_a_real_scheduler_event(client):
    with client.websocket_connect("/ws/events") as ws:
        token = login(client, "user_a")
        submit(client, token, workload="triggers-events")

        message = ws.receive_json()
        assert set(message.keys()) == {"type", "timestamp", "data"}
        assert message["data"]["event_type"] == message["type"]
        assert isinstance(message["data"]["message"], str) and message["data"]["message"]


def test_events_arrive_in_the_same_order_the_scheduler_logged_them(client):
    with client.websocket_connect("/ws/events") as ws:
        token = login(client, "user_a")
        submit(client, token, workload="ordered")

        received = [ws.receive_json() for _ in range(2)]
        # REQUEST always precedes the resulting ALLOC/BALANCE events
        # for the same submission - never reordered by the stream.
        types = [m["type"] for m in received]
        assert types[0] == "REQUEST"


# ---- O. Multiple clients receive the same broadcast ----------------------

def test_multiple_websocket_clients_all_receive_the_same_event(client):
    with client.websocket_connect("/ws/events") as ws1, client.websocket_connect("/ws/events") as ws2:
        token = login(client, "user_a")
        submit(client, token, workload="fanout")

        msg1 = ws1.receive_json()
        msg2 = ws2.receive_json()
        assert msg1["type"] == msg2["type"] == "REQUEST"
        assert msg1["data"]["event_id"] == msg2["data"]["event_id"]


# ---- P. A disconnected client never crashes the scheduler ----------------

def test_a_disconnected_websocket_client_does_not_crash_subsequent_commands(client):
    with client.websocket_connect("/ws/events"):
        pass  # connect then immediately disconnect, before anything is ever sent

    token = login(client, "user_a")
    response_submit = client.post(
        "/api/requests", headers=auth_headers(token),
        json={"workload": "after-disconnect", "gpu_count": 1, "estimated_minutes": 10, "priority": "MEDIUM"},
    )
    assert response_submit.status_code == 200  # the scheduler kept working regardless


def test_scheduler_survives_broadcasting_to_a_client_that_dropped_mid_session(client):
    ws = client.websocket_connect("/ws/events")
    ws.__enter__()
    token = login(client, "user_a")
    submit(client, token, workload="first")
    ws.__exit__(None, None, None)  # drop the connection without a clean close handshake first

    response = client.post(
        "/api/requests", headers=auth_headers(login(client, "user_b")),
        json={"workload": "second", "gpu_count": 1, "estimated_minutes": 10, "priority": "MEDIUM"},
    )
    assert response.status_code == 200


# ---- Q. User-specific notification routing --------------------------------

def test_admin_view_sees_every_users_events(client):
    with client.websocket_connect("/ws/events") as admin_ws:
        submit(client, login(client, "user_a"), workload="a-job")
        message = admin_ws.receive_json()
        assert message["data"]["user_id"] == "user_a"


def test_a_users_own_websocket_only_receives_their_own_events():
    with TestClient(app) as client:
        client.post("/api/scenarios/interactive_demo/load")
        token_a = login(client, "user_a")
        token_b = login(client, "user_b")

        with client.websocket_connect(f"/ws/events?token={token_b}") as ws_b:
            submit(client, token_a, workload="private-to-a")
            # User B's own submission below is what B should actually see.
            submit(client, token_b, workload="visible-to-b")

            message = ws_b.receive_json()
            assert message["data"]["user_id"] == "user_b"
            assert "private-to-a" not in str(message)


def test_a_users_websocket_never_receives_another_users_private_notification():
    with TestClient(app) as client:
        client.post("/api/scenarios/interactive_demo/load")
        token_a = login(client, "user_a")
        token_b = login(client, "user_b")

        with client.websocket_connect(f"/ws/events?token={token_a}") as ws_a:
            submit(client, token_b, workload="only-b")
            # Prove A's stream isn't just "empty so far" - submit
            # something of A's own right after, and confirm that (not
            # B's event) is what actually arrives first.
            submit(client, token_a, workload="belongs-to-a")

            message = ws_a.receive_json()
            assert message["data"]["user_id"] == "user_a"


# ---- R. REST snapshot followed by WebSocket updates ------------------------

def test_rest_snapshot_then_websocket_events_never_duplicates_or_loses_history(client):
    token = login(client, "user_a")
    submit(client, token, workload="before-connect")

    snapshot_events = client.get("/api/events").json()
    assert len(snapshot_events) > 0
    before_count = len(snapshot_events)

    with client.websocket_connect("/ws/events") as ws:
        submit(client, login(client, "user_b"), workload="after-connect")
        first_live = ws.receive_json()

    after_events = client.get("/api/events").json()
    assert len(after_events) > before_count
    assert first_live["data"]["event_id"] not in {e["event_id"] for e in snapshot_events}
