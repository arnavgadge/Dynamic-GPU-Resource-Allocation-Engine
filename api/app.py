"""The web adapter: FastAPI routes over `SimulationSession`.

    Scheduler Core (engine/*)
          ^
    Backend/API Adapter (this file + api/session.py, api/serializers.py)
          ^
    HTTP / WebSocket
          ^
    React

Nothing in `engine/` imports FastAPI, and nothing here contains a
scheduling decision - every route either reads `SimulationSession`
(via `serializers.serialize_state`) or calls one of its thin,
validated methods, which themselves only call `Simulator`/`Scheduler`.
REST is used for one-shot commands and the initial snapshot; the
WebSocket carries the live, repeating state stream so the dashboard
never has to poll.
"""

import asyncio
import contextlib
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api import auth
from api.auth import Account
from api.config import TICK_INTERVAL_SECONDS
from api.schemas import (
    GPURequestBody,
    LoginRequest,
    ManualAssignRequest,
    PromptResponseRequest,
    SetSpeedRequest,
    StepRequest,
)
from api.serializers import serialize_portal_state, serialize_state
from api.session import SimulationSession
from engine.simulation import load_default_registry

DEFAULT_SCENARIO_ID = "gta5_excel"


class ConnectionManager:
    """Tracks connected dashboards and pushes each one the payload
    shaped for *its* caller: the full admin snapshot for an ADMIN (or
    an anonymous/legacy connection with no token - see `state_stream`),
    the privacy-scoped portal snapshot for a logged-in USER. No
    per-client state beyond "which account is this" - every client
    still reads from the one real session, which is the whole point
    of "backend is the source of truth"; only how much of it a given
    client is shown differs.
    """

    def __init__(self) -> None:
        self._connections: Dict[WebSocket, Optional[Account]] = {}

    async def connect(self, websocket: WebSocket, account: Optional[Account]) -> None:
        await websocket.accept()
        self._connections[websocket] = account

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.pop(websocket, None)

    async def broadcast(self) -> None:
        admin_payload = None
        portal_payloads: Dict[str, dict] = {}
        dead: List[WebSocket] = []

        for connection, account in list(self._connections.items()):
            try:
                if account is not None and account.role == "USER":
                    if account.username not in portal_payloads:
                        portal_payloads[account.username] = _portal_payload(account.username)
                    await connection.send_json(portal_payloads[account.username])
                else:
                    if admin_payload is None:
                        admin_payload = _state_payload()
                    await connection.send_json(admin_payload)
            except Exception:
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


registry = load_default_registry()
session = SimulationSession(registry, DEFAULT_SCENARIO_ID)
manager = ConnectionManager()


# -- authentication ----------------------------------------------------
# A demonstration login, not enterprise auth (see `api/auth.py`). The
# one rule that matters: a request never gets to claim an identity on
# its own - `require_account`/`require_admin` always resolve the
# caller from their opaque session token, looked up server-side.

def _extract_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def require_account(authorization: Optional[str] = Header(None)) -> Account:
    token = _extract_token(authorization)
    account = auth.resolve_token(token) if token else None
    if account is None:
        raise HTTPException(status_code=401, detail="missing or invalid session token")
    return account


def require_admin(account: Account = Depends(require_account)) -> Account:
    if account.role != "ADMIN":
        raise HTTPException(status_code=403, detail="admin role required")
    return account


def _optional_account(authorization: Optional[str] = Header(None)) -> Optional[Account]:
    """Like `require_account`, but never raises - used by the one
    endpoint (`/api/prompt/respond`) both the pre-existing, token-free
    Admin Console and the new, authenticated User Portal call. No
    token (or an ADMIN token) preserves the original, unrestricted
    behavior; a USER token additionally enforces GPU ownership.
    """
    token = _extract_token(authorization)
    return auth.resolve_token(token) if token else None


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_background_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="GPU Scheduler API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local course-project demo only - see README
    allow_methods=["*"],
    allow_headers=["*"],
)


def _state_payload() -> dict:
    return {"type": "state", "payload": serialize_state(
        session.simulator, session.running, session.speed, session.scenario_id, session.scenario_name,
        real_time=session.current_real_time(), uptime_seconds=session.uptime_seconds,
    )}


def _portal_payload(username: str) -> dict:
    return {"type": "portal_state", "payload": serialize_portal_state(
        session.simulator, username, session.running, session.speed,
    )}


async def _broadcast_state() -> None:
    await manager.broadcast()


async def _background_loop() -> None:
    """Advances the session's clock while ``running`` is true, once
    every `TICK_INTERVAL_SECONDS` of *real* time - the only place real
    wall-clock time enters this project, and only ever as a pacing
    signal, never as a value fed into a scheduling decision.

    100-scenario validation, Phase 16/H06: real-hardware failures are
    already caught and recorded inside `MonitorPoller` itself (never
    raised up to here) - this broad except is only the last line of
    defense against a genuinely unexpected error, so one bad tick
    can never permanently kill the loop that drives the entire live
    session (the exact failure mode a prior version of this loop had).
    """
    while True:
        await asyncio.sleep(TICK_INTERVAL_SECONDS)
        try:
            if session.background_tick():
                await _broadcast_state()
        except Exception:
            # Deliberately broad and deliberately silent-but-alive:
            # the loop must keep ticking regardless of what went
            # wrong in one iteration. session.hardware_health (when
            # real hardware is enabled) is still the right place to
            # look for *expected* hardware trouble; this only guards
            # against the unexpected.
            continue


# -- REST: read-only ---------------------------------------------------

@app.get("/api/scenarios")
def list_scenarios():
    return [
        {"scenario_id": info.scenario_id, "name": info.name, "description": info.description}
        for info in registry.list_scenarios()
    ]


@app.get("/api/state")
def get_state():
    return serialize_state(
        session.simulator, session.running, session.speed, session.scenario_id, session.scenario_name,
        real_time=session.current_real_time(), uptime_seconds=session.uptime_seconds,
    )


@app.get("/api/clock")
def get_clock():
    """A cheap, dedicated real-clock reading (Issue 5) - the frontend
    polls this every second for a continuously-updating header clock/
    uptime without waiting on (or interfering with) the main state
    WebSocket's own broadcast cadence. Always the real system clock
    and this session's real elapsed uptime - never the simulated
    clock, never a value invented in React.
    """
    return {
        "real_time": session.current_real_time().isoformat(),
        "uptime_seconds": session.uptime_seconds,
    }


# -- REST: authentication (Part 28/29) ---------------------------------
# A demonstration login - five seeded accounts, an opaque token, no
# password. `require_account`/`require_admin` are what every route
# below actually trusts for identity; nothing accepts a user id from
# a request body as proof of who is asking.

@app.post("/api/auth/login")
def login(body: LoginRequest):
    try:
        account = auth.login(body.username)
    except KeyError:
        raise HTTPException(status_code=401, detail=f"unknown demo account {body.username!r}")
    token = auth.create_token(account.username)
    return {"token": token, "username": account.username, "role": account.role, "display_name": account.display_name}


@app.post("/api/auth/logout")
def logout(account: Account = Depends(require_account), authorization: Optional[str] = Header(None)):
    auth.logout(_extract_token(authorization))
    return {"ok": True}


@app.get("/api/auth/me")
def whoami(account: Account = Depends(require_account)):
    return {"username": account.username, "role": account.role, "display_name": account.display_name}


# -- REST: User Portal (Phase 9) ---------------------------------------
# The literal missing piece a prior audit of this project flagged:
# `Scheduler.submit_job` always worked, but nothing let a live user
# reach it. `submit_gpu_request` is that bridge - see
# `SimulationSession.submit_user_request` for what it actually does
# (and does not fake: multi-GPU requests are rejected, not faked).

@app.get("/api/portal/state")
def get_portal_state(account: Account = Depends(require_account)):
    return serialize_portal_state(session.simulator, account.username, session.running, session.speed)


@app.post("/api/requests")
async def submit_gpu_request(body: GPURequestBody, account: Account = Depends(require_account)):
    try:
        session.submit_user_request(
            account.username, account.display_name, body.workload, body.gpu_count, body.estimated_minutes, body.priority,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await _broadcast_state()
    return serialize_portal_state(session.simulator, account.username, session.running, session.speed)


# -- REST: real hardware telemetry (Requirement 1) ----------------------
# Off by default (`SimulationSession.hardware_monitor is None`) - the
# interactive demo runs on the scenario's own simulated readings
# unless an admin explicitly enables real hardware, and even then only
# if `detect_gpu_monitor` genuinely finds NVML or `nvidia-smi`. This
# endpoint never claims real hardware is active when it isn't.

@app.get("/api/hardware/status")
def hardware_status():
    health = session.hardware_health
    return {
        "enabled": session.hardware_monitor is not None,
        "source": session.hardware_source,
        "error": session.hardware_error,
        "health": {
            "healthy": health.healthy,
            "consecutive_failures": health.consecutive_failures,
            "last_error": health.last_error,
            "last_success_at": health.last_success_at.isoformat() if health.last_success_at else None,
            "per_gpu_errors": health.per_gpu_errors,
        } if health is not None else None,
    }


@app.post("/api/hardware/enable")
def hardware_enable(account: Account = Depends(require_admin)):
    available = session.enable_real_hardware()
    return {
        "enabled": available,
        "source": session.hardware_source,
        "error": session.hardware_error,
    }


@app.post("/api/hardware/disable")
def hardware_disable(account: Account = Depends(require_admin)):
    session.disable_real_hardware()
    return {"enabled": False, "source": None, "error": None}


# -- REST: admin manual assignment (Phase 10) --------------------------
# The demo/test-initialization control described in Requirement 4's
# "manual GPU assignment" - an explicit admin action, still going
# through `Scheduler.manual_assign_gpu` (never a raw state mutation),
# and still broadcast to every connected dashboard exactly like every
# other command below.

@app.post("/api/admin/assign")
async def manual_assign(body: ManualAssignRequest, account: Account = Depends(require_admin)):
    try:
        session.manual_assign_gpu(body.gpu_id, body.user_id, body.display_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await _broadcast_state()
    return get_state()


# -- REST: commands ----------------------------------------------------
# Every command below returns the resulting state directly (so a
# plain REST caller never needs the WebSocket just to see the effect
# of its own request) *and* broadcasts the same state to every
# connected dashboard.

@app.post("/api/scenarios/{scenario_id}/load")
async def load_scenario(scenario_id: str):
    try:
        session.load_scenario(scenario_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no scenario registered with id {scenario_id!r}")
    await _broadcast_state()
    return get_state()


@app.post("/api/control/start")
async def start_simulation():
    session.start()
    await _broadcast_state()
    return get_state()


@app.post("/api/control/pause")
async def pause_simulation():
    session.pause()
    await _broadcast_state()
    return get_state()


@app.post("/api/control/reset")
async def reset_simulation():
    session.reset()
    await _broadcast_state()
    return get_state()


@app.post("/api/control/step")
async def step_simulation(body: Optional[StepRequest] = None):
    session.step(body.minutes if body else None)
    await _broadcast_state()
    return get_state()


@app.post("/api/control/speed")
async def set_speed(body: SetSpeedRequest):
    try:
        session.set_speed(body.speed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await _broadcast_state()
    return get_state()


@app.post("/api/prompt/respond")
async def respond_to_prompt(body: PromptResponseRequest, account: Optional[Account] = Depends(_optional_account)):
    """Answer a pending reclamation prompt.

    Called by both the Admin Console and the User Portal - but Issue
    6 draws a hard line the earlier admin-override design didn't: an
    ADMIN token or no token at all may still answer for a GPU that
    belongs to nobody with a real login (every pre-existing scenario's
    synthetic users - "frank", "user-a", etc. - have no portal to
    answer from, so an admin override there is the only way to
    demonstrate reclamation at all, exactly as before). But once a GPU
    is actually owned by a real logged-in demo account, the admin
    override no longer applies to it - only that account's own USER
    token may respond; the confirmation belongs on their User Portal,
    never decided on their behalf from the Admin Console. A USER token
    always enforces the narrower rule: only for a GPU that is
    genuinely theirs right now.
    """
    owner = session.gpu_owner(body.gpu_id)
    owner_account = auth.ACCOUNTS.get(owner) if owner else None

    if account is not None and account.role == "USER":
        if owner != account.username:
            raise HTTPException(status_code=403, detail="you may only respond for your own GPU")
    elif owner_account is not None and owner_account.role == "USER":
        raise HTTPException(
            status_code=403,
            detail=f"{body.gpu_id} belongs to a logged-in user - respond from their User Portal",
        )

    try:
        session.respond_to_prompt(body.gpu_id, body.response)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    await _broadcast_state()

    if account is not None and account.role == "USER":
        return serialize_portal_state(session.simulator, account.username, session.running, session.speed)
    return get_state()


# -- WebSocket: live state stream ---------------------------------------

@app.websocket("/ws/state")
async def state_stream(websocket: WebSocket, token: Optional[str] = Query(None)):
    """No ``token`` (or an ADMIN token) streams the full admin
    snapshot - unchanged from before Phase 9, so every existing
    Admin Console connection keeps working exactly as it did. A USER
    token streams the privacy-scoped portal snapshot instead
    (Part 24) - resolved *once*, here, from the session token; a
    connection can never claim a different identity than the one its
    token actually belongs to.
    """
    account = auth.resolve_token(token) if token else None
    await manager.connect(websocket, account)
    try:
        # Resynchronize immediately on connect (and reconnect) - a
        # client never has to guess at state it missed.
        if account is not None and account.role == "USER":
            await websocket.send_json(_portal_payload(account.username))
        else:
            await websocket.send_json(_state_payload())
        while True:
            # This endpoint is push-only; we still need to await
            # something so a client disconnect is detected promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
