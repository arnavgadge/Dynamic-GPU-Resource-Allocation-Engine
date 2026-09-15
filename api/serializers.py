"""Turn backend domain objects into plain, JSON-safe dicts.

This is the only place a `GPU`/`User`/`Job`/`Event`/dataclass from
`engine/` gets converted into something `json.dumps` (or FastAPI's
response encoder) can serialize. No scheduling decision is made here
- these functions only read fields that already exist on the real
objects and reshape them; the one piece of real computation
(`waiting_seconds`, `allocation score`) explicitly calls back into
the real `engine.allocation` code rather than recomputing anything.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from api.auth import ACCOUNTS
from api.config import ALLOWED_SPEEDS
from engine.allocation.config import JOB_SIZE_SIMILARITY_THRESHOLD, PRIORITY_WEIGHT, SIZE_WEIGHT
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState
from engine.models.user import User
from engine.models.event import Event
from engine.scheduler import Scheduler
from engine.simulation.simulator import Simulator

#: Which backend event types are worth surfacing to a USER as a
#: notification, and the plain-language label for each (Part 23) - the
#: *content* of every notification is still the real backend message,
#: this only picks which events are notification-worthy and what to
#: call them. `RESPONSE` is left out deliberately: it is the user's
#: own action echoed back, not new information for them.
_NOTIFICATION_LABELS = {
    "REQUEST": "GPU REQUEST SUBMITTED",
    "ALLOC": "GPU ALLOCATED",
    "PROMPT": "GPU RESOURCE WARNING",
    "RECLAIM": "GPU RECLAIMED",
    "STATUS": "GPU STATUS CHANGED",
    "JOB_COMPLETION": "JOB COMPLETED",
    "MANUAL_RELEASE": "GPU RELEASED",
    "PRIORITY_PREEMPTION": "GPU REASSIGNED (PRIORITY)",
    "HARDWARE_FAILURE": "GPU HARDWARE ISSUE",
    "JOB_CANCELLED": "REQUEST CANCELLED",
    "ADMIN_FORCE_RECLAIM": "GPU RECLAIMED BY ADMIN",
    "PRIORITY_CHANGED": "REQUEST PRIORITY CHANGED",
}


def _estimated_completion(job: Job, now: datetime) -> Optional[Dict[str, Any]]:
    """START/EST. COMPLETION/REMAINING for a RUNNING job, derived purely
    from `job.started_at` + `job.estimated_size_minutes` against the
    simulator's own clock (``now``) - never `Date.now()`, and never
    treated as proof a real workload is actually done (Part 35): this
    is what feeds `Scheduler.check_estimated_completions`'s *prompt*,
    not an automatic completion.
    """
    if job.started_at is None:
        return None
    completion = job.started_at + timedelta(minutes=job.estimated_size_minutes)
    return {
        "started_at": job.started_at.isoformat(),
        "estimated_completion": completion.isoformat(),
        "remaining_seconds": max((completion - now).total_seconds(), 0.0),
    }


def serialize_gpu(state: SchedulerState, gpu: GPU, reclamation_engine, now: datetime) -> Dict[str, Any]:
    user = state.get_user(gpu.assigned_user_id) if gpu.assigned_user_id else None
    job = state.get_job(gpu.assigned_job_id) if gpu.assigned_job_id else None
    timing = _estimated_completion(job, now) if job else None
    request_context = reclamation_engine.pending_request_context(gpu.gpu_id)
    return {
        "gpu_id": gpu.gpu_id,
        "utilization_percent": gpu.utilization_percent,
        "memory_used_mb": gpu.memory_used_mb,
        "total_memory_mb": gpu.total_memory_mb,
        "status": gpu.status.value,
        "assigned_user_id": gpu.assigned_user_id,
        "assigned_user_name": user.name if user else None,
        "assigned_job_id": gpu.assigned_job_id,
        "assigned_job_name": job.name if job else None,
        "job_gpu_count": job.gpu_count if job else None,
        "has_pending_prompt": reclamation_engine.has_pending_prompt(gpu.gpu_id),
        "requested_by_user_id": request_context.requesting_user_id if request_context else None,
        "requested_by_user_name": request_context.requesting_user_name if request_context else None,
        "requested_for_job_id": request_context.requesting_job_id if request_context else None,
        **(timing or {}),
    }


def serialize_user(state: SchedulerState, user: User) -> Dict[str, Any]:
    # A user's current "workload" for display: the name of whichever
    # job they currently have running, if any - purely informational,
    # exactly as Phase 1-6 always treated `Job.name`.
    running_job = next(
        (state.get_job(job_id) for job_id in user.running_job_ids if state.get_job(job_id) is not None),
        None,
    )
    return {
        "user_id": user.user_id,
        "name": user.name,
        "priority": user.priority.name,
        "assigned_gpu_ids": list(user.assigned_gpu_ids),
        "running_job_ids": list(user.running_job_ids),
        "workload": running_job.name if running_job else None,
        "gpu_id": running_job.assigned_gpu_id if running_job else None,
        "status": "ACTIVE" if running_job is not None else "IDLE",
    }


def _wait_seconds(job: Job, now: datetime) -> float:
    """How long ``job`` has been waiting, using the simulator's clock -
    never `datetime.now()`. For a job that has already started, this
    is its wait *before* it started (a fixed, historical number).
    """
    end = job.started_at or now
    return max((end - job.submitted_at).total_seconds(), 0.0)


def serialize_job(job: Job, now: datetime) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "user_id": job.user_id,
        "name": job.name,
        "priority": job.priority.name,
        "estimated_size_minutes": job.estimated_size_minutes,
        "status": job.status.value,
        "gpu_count": job.gpu_count,
        "assigned_gpu_id": job.assigned_gpu_id,
        "assigned_gpu_ids": list(job.assigned_gpu_ids),
        "gpus_still_needed": job.gpus_still_needed,
        "submitted_at": job.submitted_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "wait_seconds": _wait_seconds(job, now),
        **(_estimated_completion(job, now) or {}),
    }


def serialize_waiting_queue(scheduler: Scheduler, now: datetime) -> List[Dict[str, Any]]:
    """The waiting queue, each job annotated with the *real* allocation
    score from `AllocationEngine` - never recomputed in this file, let
    alone in JavaScript.

    The score shown is computed relative to every currently-waiting
    job (a display simplification - `AllocationEngine.select_next_job`
    internally narrows to only critical jobs first when any are
    waiting). The winner the backend actually picks is unaffected by
    this display choice; see the README's Phase 7 section.
    """
    waiting_jobs = scheduler.state.get_waiting_jobs()
    if not waiting_jobs:
        return []

    entries = []
    for job in waiting_jobs:
        score = scheduler.allocation_engine.calculate_score(job, waiting_jobs)
        entries.append({
            **serialize_job(job, now),
            "allocation_score": round(score, 3),
        })
    return entries


def serialize_event(event: Event) -> Dict[str, Any]:
    return {
        "event_id": event.event_id,
        "timestamp": event.timestamp.isoformat(),
        "event_type": event.event_type.value,
        "gpu_id": event.gpu_id,
        "user_id": event.user_id,
        "job_id": event.job_id,
        "message": event.message,
        "reason": event.reason,
        "metadata": dict(event.metadata),
    }


def serialize_pending_prompts(state: SchedulerState, reclamation_engine) -> List[Dict[str, Any]]:
    """One entry per GPU currently awaiting a YES/NO response.

    The prompt text comes from the most recent `PROMPT` event already
    logged for that GPU - never re-generated here.
    """
    prompts = []
    for gpu in state.gpus.values():
        if not reclamation_engine.has_pending_prompt(gpu.gpu_id):
            continue
        last_prompt = next(
            (e for e in reversed(state.events) if e.event_type.value == "PROMPT" and e.gpu_id == gpu.gpu_id),
            None,
        )
        request_context = reclamation_engine.pending_request_context(gpu.gpu_id)
        owner_account = ACCOUNTS.get(gpu.assigned_user_id) if gpu.assigned_user_id else None
        prompts.append({
            "gpu_id": gpu.gpu_id,
            "message": last_prompt.message if last_prompt else "Are you still using this GPU?",
            "reason": last_prompt.reason if last_prompt else None,
            "owner_user_id": gpu.assigned_user_id,
            "owner_user_name": owner_account.display_name if owner_account else None,
            # Issue 6: whether this GPU belongs to a real logged-in
            # demo account with an actual User Portal to answer from.
            # The Admin Console renders this prompt as read-only status
            # when true - the confirmation belongs on that user's own
            # portal, never decided for them from the Admin Console.
            # `False` for a classic scenario's synthetic user (e.g.
            # "frank"), which has no portal at all, so the admin
            # override remains the only way to demonstrate it.
            "owner_has_portal": owner_account is not None and owner_account.role == "USER",
            # Set only when this prompt was raised on behalf of another
            # user's waiting multi-GPU request (Phase 10, Requirement
            # 4) - `None` for an ordinary utilization-tier or
            # estimated-completion prompt, which are nobody else's ask.
            "requested_by_user_id": request_context.requesting_user_id if request_context else None,
            "requested_by_user_name": request_context.requesting_user_name if request_context else None,
            "requested_for_job_id": request_context.requesting_job_id if request_context else None,
        })
    return prompts


def serialize_config(scheduler: Scheduler) -> Dict[str, Any]:
    """The active policy configuration - read-only, for display only.
    Never sent back by the frontend to change scheduling behavior.
    """
    policy = scheduler.reclamation_engine.policy
    balancing_policy = scheduler.router.policy
    return {
        "allocation": {
            "priority_weight": PRIORITY_WEIGHT,
            "size_weight": SIZE_WEIGHT,
            "similarity_threshold": JOB_SIZE_SIMILARITY_THRESHOLD,
        },
        "reclamation": {
            "tier1_threshold_percent": policy.tier1.utilization_threshold_percent,
            "tier1_sustained_seconds": policy.tier1.sustained_duration.total_seconds(),
            "tier2_threshold_percent": policy.tier2.utilization_threshold_percent,
            "tier2_sustained_seconds": policy.tier2.sustained_duration.total_seconds(),
            "no_response_grace_seconds": policy.no_response_grace_period.total_seconds(),
        },
        "balancing": {
            "imbalance_threshold_percent": balancing_policy.imbalance_threshold_percent,
        },
    }


def serialize_decision_trace(scheduler: Scheduler) -> Optional[Dict[str, Any]]:
    """The most recent "which job / which GPU" decision, for the
    Admin Console's decision-trace panel (Part 17/44) - every field
    here is read straight off the real `AllocationDecision`/
    `RoutingDecision` objects `Scheduler.try_allocate_all` already
    produced; nothing is recalculated.

    Returns ``None`` until at least one allocation has actually
    happened - there is no decision to trace before that, and this
    never fabricates a placeholder one.
    """
    allocation = scheduler.last_allocation_decision
    routing = scheduler.last_routing_decision
    resource_request = scheduler.last_resource_request_decision
    resource_request_block = None
    if resource_request is not None:
        resource_request_block = {
            "gpu_id": resource_request.gpu_id,
            "timestamp": resource_request.timestamp.isoformat(),
            "action": resource_request.action.value,
            "reason": resource_request.reason,
            "requesting_user_id": resource_request.requesting_user_id,
            "requesting_job_id": resource_request.requesting_job_id,
        }

    if allocation is None or routing is None:
        # A resource request can exist before any allocation ever has
        # (an admin's manual assignment already occupies every GPU) -
        # report it on its own rather than hiding it behind an
        # allocation/routing pair that hasn't happened yet.
        return {"resource_request": resource_request_block} if resource_request_block else None

    return {
        "job_id": allocation.job_id,
        "user_id": allocation.user_id,
        "gpu_id": allocation.gpu_id,
        "timestamp": allocation.timestamp.isoformat(),
        "allocation": {
            "policy": allocation.policy.value,
            "reason": allocation.reason,
            "candidates": [
                {
                    "job_id": c.job_id,
                    "user_id": c.user_id,
                    "priority": c.priority.name,
                    "size_minutes": c.size_minutes,
                    "wait_seconds": c.waiting_time.total_seconds(),
                    "score": round(c.score, 3) if c.score is not None else None,
                }
                for c in allocation.candidates
            ],
        },
        "routing": {
            "outcome": routing.outcome.value,
            "reason": routing.reason,
            "imbalance_detected": routing.imbalance_detected,
            "candidates": [
                {"gpu_id": c.gpu_id, "utilization_percent": c.utilization_percent, "available": c.available}
                for c in routing.candidates
            ],
        },
        "resource_request": resource_request_block,
    }


def serialize_notifications(state: SchedulerState, user_id: str) -> List[Dict[str, Any]]:
    """Every backend event about ``user_id``, newest first, labeled
    for a user-facing notification list (Part 23) - never a fake
    notification synthesized in React; every one is a real logged
    `Event`, filtered to this user and given a plain-language label.
    """
    notifications = []
    for event in state.events:
        if event.user_id != user_id:
            continue
        label = _NOTIFICATION_LABELS.get(event.event_type.value)
        if label is None:
            continue
        notifications.append({
            "notification_id": event.event_id,
            "type": event.event_type.value,
            "label": label,
            "message": event.message,
            "timestamp": event.timestamp.isoformat(),
        })
    notifications.reverse()
    return notifications


def serialize_portal_state(simulator: Simulator, user_id: str, running: bool, speed: float) -> Dict[str, Any]:
    """The User Portal's scoped snapshot - Part 24's privacy rule
    enforced *here*, on the backend, not by the frontend politely
    declining to render fields it was handed anyway. A USER-role
    caller never receives other users' jobs, priorities, allocation
    scores, or the global decision trace - only their own.
    """
    scheduler = simulator.scheduler
    state = scheduler.state
    now = simulator.clock.now()

    waiting_jobs = state.get_waiting_jobs()
    scored: Dict[str, float] = {}
    rank: Dict[str, int] = {}
    if waiting_jobs:
        pairs = [(job, scheduler.allocation_engine.calculate_score(job, waiting_jobs)) for job in waiting_jobs]
        pairs.sort(key=lambda pair: -pair[1])
        for position, (job, score) in enumerate(pairs, start=1):
            scored[job.job_id] = score
            rank[job.job_id] = position

    my_jobs = []
    for job in state.jobs.values():
        if job.user_id != user_id:
            continue
        entry = serialize_job(job, now)
        if job.status.value == "WAITING":
            entry["allocation_score"] = round(scored.get(job.job_id, 0.0), 3)
            entry["queue_position"] = rank.get(job.job_id)
        if job.status.value == "RUNNING" and job.assigned_gpu_ids:
            # One entry per GPU this job actually holds - a 1-GPU job
            # (the default) gets a single-item list; a multi-GPU job
            # (Requirement 3) gets one per GPU, never a single
            # averaged/invented number. `gpu_utilization_percent`/
            # `gpu_status` stay as the first GPU's values, unchanged,
            # for a single-GPU job's existing display.
            gpus = []
            for gpu_id in job.assigned_gpu_ids:
                gpu = state.get_gpu(gpu_id)
                if gpu is not None:
                    gpus.append({
                        "gpu_id": gpu.gpu_id,
                        "utilization_percent": gpu.utilization_percent,
                        "status": gpu.status.value,
                    })
            entry["gpus"] = gpus
            if gpus:
                entry["gpu_utilization_percent"] = gpus[0]["utilization_percent"]
                entry["gpu_status"] = gpus[0]["status"]
        my_jobs.append(entry)
    my_jobs.sort(key=lambda entry: entry["submitted_at"])

    user = state.get_user(user_id)
    pending_prompt = None
    if user is not None:
        for gpu_id in user.assigned_gpu_ids:
            if scheduler.reclamation_engine.has_pending_prompt(gpu_id):
                prompts = serialize_pending_prompts(state, scheduler.reclamation_engine)
                pending_prompt = next((p for p in prompts if p["gpu_id"] == gpu_id), None)
                break

    return {
        "simulated_time": now.isoformat(),
        "user": {
            "user_id": user_id,
            "name": user.name if user is not None else user_id,
            "priority": user.priority.name if user is not None else None,
        },
        "my_jobs": my_jobs,
        "pending_prompt": pending_prompt,
        "notifications": serialize_notifications(state, user_id),
        "simulation": {"running": running, "speed": speed},
    }


def serialize_state(
    simulator: Simulator, running: bool, speed: float,
    scenario_id: Optional[str] = None, scenario_name: Optional[str] = None,
    real_time: Optional[datetime] = None, uptime_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """The full dashboard snapshot - what `GET /api/state` returns and
    what every WebSocket ``state`` message carries.

    ``real_time``/``uptime_seconds`` (Issue 5) are the actual system
    clock and real elapsed session time - purely for display (the
    header clock/uptime), never fed into any scheduling decision.
    ``simulated_time`` below remains the deterministic scenario clock
    the engines themselves actually run on; the two are deliberately
    never confused with each other.
    """
    scheduler = simulator.scheduler
    state = scheduler.state
    now = simulator.clock.now()

    return {
        "simulated_time": now.isoformat(),
        "real_time": (real_time or datetime.now(timezone.utc)).isoformat(),
        "uptime_seconds": uptime_seconds if uptime_seconds is not None else 0.0,
        "engine_status": state.engine_status.value,
        "gpus": [serialize_gpu(state, gpu, scheduler.reclamation_engine, now) for gpu in state.gpus.values()],
        "users": [serialize_user(state, user) for user in state.users.values()],
        "jobs": [serialize_job(job, now) for job in state.jobs.values()],
        "waiting_queue": serialize_waiting_queue(scheduler, now),
        "events": [serialize_event(event) for event in state.events],
        "pending_prompts": serialize_pending_prompts(state, scheduler.reclamation_engine),
        "decision_trace": serialize_decision_trace(scheduler),
        "config": serialize_config(scheduler),
        "simulation": {
            "running": running,
            "speed": speed,
            "allowed_speeds": list(ALLOWED_SPEEDS),
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "has_pending_actions": simulator.has_pending_actions(),
        },
    }
