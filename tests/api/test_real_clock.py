"""Issue 5: the header clock/uptime track the actual system clock and
real elapsed session time - never the deterministic simulated clock,
never a value invented in React.
"""

import time
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from api.app import app
from api.session import SimulationSession
from engine.simulation import load_default_registry


# -- Test 1: real clock --------------------------------------------------

def test_1_session_real_time_tracks_the_actual_system_clock():
    session = SimulationSession(load_default_registry(), "interactive_demo")
    before = datetime.now(timezone.utc)
    reported = session.current_real_time()
    after = datetime.now(timezone.utc)

    assert before <= reported <= after


def test_1_api_clock_endpoint_reports_the_real_system_clock():
    with TestClient(app) as client:
        before = datetime.now(timezone.utc)
        body = client.get("/api/clock").json()
        after = datetime.now(timezone.utc)

        reported = datetime.fromisoformat(body["real_time"])
        assert before <= reported <= after


def test_1_state_real_time_is_independent_of_simulated_time():
    with TestClient(app) as client:
        client.post("/api/scenarios/gta5_excel/load")
        state = client.get("/api/state").json()
        # The scenario's simulated clock always starts at a fixed,
        # arbitrary date (2026-01-01T09:00:00) - the real clock must
        # never equal that (unless run at that literal instant).
        assert state["simulated_time"] != state["real_time"]
        assert state["simulated_time"].startswith("2026-01-01T09:00:00")


# -- Test 2: uptime --------------------------------------------------------

def test_2_uptime_increases_with_real_elapsed_time():
    session = SimulationSession(load_default_registry(), "interactive_demo")
    first = session.uptime_seconds
    time.sleep(0.05)
    second = session.uptime_seconds

    assert second > first
    assert first >= 0.0


def test_2_uptime_resets_on_session_reset():
    session = SimulationSession(load_default_registry(), "interactive_demo")
    time.sleep(0.05)
    assert session.uptime_seconds > 0.0

    session.reset()
    assert session.uptime_seconds < 0.05


def test_2_uptime_resets_on_scenario_reload():
    session = SimulationSession(load_default_registry(), "interactive_demo")
    time.sleep(0.05)
    session.load_scenario("gta5_excel")
    assert session.uptime_seconds < 0.05


def test_2_api_clock_uptime_increases_across_calls():
    with TestClient(app) as client:
        first = client.get("/api/clock").json()["uptime_seconds"]
        time.sleep(0.05)
        second = client.get("/api/clock").json()["uptime_seconds"]
        assert second > first
