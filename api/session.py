"""SimulationSession: the one live `Simulator` instance the API adapter
drives, plus the small amount of "is it running, how fast" state a
web control surface needs that the backend engines themselves have no
reason to know about.

This class makes no scheduling decision. Every mutating method here
is a thin, validated call straight into `Simulator`/`Scheduler` -
exactly the objects Phase 6 already built and tested on their own.
"""

import itertools
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from api.config import (
    ALLOWED_SPEEDS,
    BACKGROUND_TICK,
    BASE_TICK,
    DEFAULT_SPEED,
    MAX_USER_GPU_REQUEST,
    MIN_USER_GPU_REQUEST,
)
from engine.hardware.factory import detect_gpu_monitor
from engine.hardware.monitor import GPUMonitor, MonitorUnavailableError
from engine.hardware.poller import MonitorPoller
from engine.models.enums import Priority
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.policy import ConfirmationResponse
from engine.simulation.registry import ScenarioRegistry
from engine.simulation.simulator import Simulator


class SimulationSession:
    """Owns exactly one `Simulator` at a time and the control-surface
    state (``running``, ``speed``) layered on top of it."""

    def __init__(self, registry: ScenarioRegistry, initial_scenario_id: str) -> None:
        self.registry = registry
        self.running: bool = False
        self.speed: float = DEFAULT_SPEED
        self.simulator: Simulator
        self.scenario_id: str
        self.scenario_name: str
        self._request_seq = itertools.count(1)
        # Real wall-clock session start (Issue 5) - `time.monotonic()`
        # rather than `datetime.now()` because uptime must never jump
        # backward/forward if the system clock itself is adjusted; it
        # only ever measures *elapsed* real time. Reset alongside the
        # scenario/session itself (`load_scenario`/`reset` below) -
        # never touched by anything simulated-time-related.
        self._session_started_monotonic = time.monotonic()
        self.load_scenario(initial_scenario_id)

        # Real-hardware telemetry (Requirement 1) is entirely optional
        # and off by default - the interactive demo runs on the
        # scenario's own simulated readings unless `enable_real_hardware`
        # is called and a real monitor genuinely initializes. Never set
        # silently: `hardware_source` always tells the truth about
        # whether this session is reading real NVML/nvidia-smi data or
        # not, so nothing here can be mistaken for real-hardware
        # telemetry that was never actually confirmed available.
        self.hardware_monitor: Optional[GPUMonitor] = None
        self.hardware_source: Optional[str] = None
        self.hardware_error: Optional[str] = None
        #: One persistent `MonitorPoller` for the session's lifetime
        #: (Phase 16) - its `health` (consecutive failures, per-GPU
        #: errors, last success) needs to accumulate across ticks, not
        #: be discarded and recreated on every single one.
        self._hardware_poller: Optional[MonitorPoller] = None

    def enable_real_hardware(self) -> bool:
        """Try to switch this session onto real GPU telemetry (NVML,
        then ``nvidia-smi``) for every subsequent tick, through the
        exact same `engine.hardware` abstraction Phase 8 built and
        tested with mocks - this call is the only place that abstraction
        is ever pointed at *real* hardware instead of a test double.

        Returns ``True`` and records which source initialized
        (``self.hardware_source``) if real hardware is genuinely
        available; returns ``False`` and records why
        (``self.hardware_error``) otherwise - this method never
        pretends hardware was found when `detect_gpu_monitor` could
        not actually confirm one.
        """
        try:
            monitor = detect_gpu_monitor()
        except MonitorUnavailableError as exc:
            self.hardware_monitor = None
            self.hardware_source = None
            self.hardware_error = str(exc)
            return False
        self.hardware_monitor = monitor
        self.hardware_source = type(monitor).__name__
        self.hardware_error = None
        self._hardware_poller = MonitorPoller(self.simulator.scheduler, monitor)
        return True

    def disable_real_hardware(self) -> None:
        """Go back to the scenario's own simulated utilization readings."""
        self.hardware_monitor = None
        self.hardware_source = None
        self.hardware_error = None
        self._hardware_poller = None

    @property
    def hardware_health(self):
        """The persistent poller's own health snapshot (Phase 16) -
        `None` when real hardware isn't enabled at all. Exposed for
        the admin console: consecutive failures, per-GPU errors, and
        when the last successful poll actually happened."""
        return self._hardware_poller.health if self._hardware_poller is not None else None

    def _poll_real_hardware(self, now) -> None:
        """If real hardware is enabled, read it and feed whatever it
        reports straight into `Scheduler.record_utilization` - the
        identical entry point a scenario's own `UtilizationAction`
        uses, via the same persistent `MonitorPoller` Phase 8 built
        and Phase 16 hardened (feed the readings in, then let
        reclamation timeouts and allocation react), so the
        Allocation/Reclamation/Balancing engines behave exactly the
        same regardless of source. A metric for a logical GPU this
        session's pool doesn't have is skipped, never invented. A
        transient hardware failure (whole-poll or per-GPU) is
        recorded on `hardware_health` and never raised - the session
        keeps running on its last-known state.
        """
        if self._hardware_poller is None:
            return
        self._hardware_poller.poll_once(now)

    # -- scenario / lifecycle -------------------------------------------

    def load_scenario(self, scenario_id: str) -> None:
        """Load (and implicitly reset into) a scenario by id.

        Raises ``KeyError`` for an unknown id - the same error
        `ScenarioRegistry.load` already raises; the API layer maps
        that to an HTTP 404 rather than inventing a new error shape.
        """
        scenario = self.registry.load(scenario_id)
        self.simulator = Simulator(scenario, speed=self.speed)
        self.scenario_id = scenario_id
        self.scenario_name = scenario.name
        self.running = False
        self._session_started_monotonic = time.monotonic()

    def reset(self) -> None:
        """Reset the current scenario back to its initial state."""
        self.simulator.reset()
        self.running = False
        self._session_started_monotonic = time.monotonic()

    # -- real wall-clock (Issue 5) -----------------------------------
    # Purely a display concern - never fed into any scheduling
    # decision. `simulated_time` (the deterministic scenario clock)
    # remains the only clock the engines themselves ever see.

    @property
    def uptime_seconds(self) -> float:
        """Real elapsed seconds since this session (or its current
        scenario) last started/reset - resets exactly when
        `load_scenario`/`reset` do, never drifts with simulated time
        or speed."""
        return time.monotonic() - self._session_started_monotonic

    @staticmethod
    def current_real_time() -> datetime:
        """The actual system clock, right now - never the simulated
        clock, never a frontend-invented value."""
        return datetime.now(timezone.utc)

    def start(self) -> None:
        self.running = True

    def pause(self) -> None:
        self.running = False

    def set_speed(self, speed: float) -> None:
        if speed not in ALLOWED_SPEEDS:
            raise ValueError(f"speed must be one of {ALLOWED_SPEEDS}, got {speed!r}")
        self.speed = speed
        self.simulator.speed = speed
        self.simulator.clock.speed = speed

    # -- time progression --------------------------------------------

    def step(self, minutes: Optional[float] = None) -> None:
        """Manually advance by one step, regardless of ``running`` -
        the explicit "Step / Tick" control."""
        delta = timedelta(minutes=minutes) if minutes else BASE_TICK
        self.simulator.advance(delta)
        self._poll_real_hardware(self.simulator.clock.now())
        self._check_estimated_completions()

    def background_tick(self) -> bool:
        """Called by the server's background loop every real-time
        interval. Advances simulated time by ``speed`` background
        ticks only if the simulation is currently ``running``; a
        no-op (returns False) otherwise, so a paused session never
        drifts and a caller knows whether to bother broadcasting.

        Uses `BACKGROUND_TICK` (equal to `TICK_INTERVAL_SECONDS`, the
        real interval between calls to this method) rather than
        `BASE_TICK` - so "1x" genuinely advances simulated time at the
        same rate as real time, and a reclamation/estimated-completion
        threshold of N simulated minutes actually takes N real minutes
        at 1x, not N real seconds.
        """
        if not self.running:
            return False
        self.simulator.advance(BACKGROUND_TICK * self.speed)
        self._poll_real_hardware(self.simulator.clock.now())
        self._check_estimated_completions()
        return True

    def _check_estimated_completions(self) -> None:
        """After every time advance, ask (through the real confirmation
        flow - never by silently completing a job) whether any RUNNING
        job's estimated duration has elapsed. See
        `Scheduler.check_estimated_completions` for why this reuses
        `ReclamationEngine`'s prompt/response machinery rather than
        inventing a second one.
        """
        self.simulator.scheduler.check_estimated_completions(self.simulator.clock.now())

    # -- reclamation prompt -----------------------------------------------

    def respond_to_prompt(self, gpu_id: str, response: str) -> None:
        """Answer a pending reclamation prompt for ``gpu_id``.

        Mirrors exactly what `Simulator._execute` does for a scenario's
        own `UserResponseAction` - respond, then let the scheduler
        attempt to allocate anything the response just freed. Raises
        ``KeyError`` for an unknown GPU, ``ValueError`` if that GPU has
        no prompt currently pending.
        """
        state = self.simulator.scheduler.state
        if state.get_gpu(gpu_id) is None:
            raise KeyError(f"unknown GPU {gpu_id!r}")
        if not self.simulator.scheduler.reclamation_engine.has_pending_prompt(gpu_id):
            raise ValueError(f"GPU {gpu_id!r} has no pending confirmation prompt")

        now = self.simulator.clock.now()
        confirmation = ConfirmationResponse[response]
        self.simulator.scheduler.respond_to_prompt(gpu_id, confirmation, now=now)
        self.simulator.scheduler.try_allocate_all(now=now)

    def gpu_owner(self, gpu_id: str) -> Optional[str]:
        """The user id currently assigned to ``gpu_id``, or ``None`` if
        it's unknown or unassigned. Used by the API layer to check that
        a USER-role caller only answers a prompt for their *own* GPU -
        an ADMIN may answer for any GPU (Part 21's "admin override" is
        exactly this: acting on the real engine, never bypassing it).
        """
        gpu = self.simulator.scheduler.state.get_gpu(gpu_id)
        return gpu.assigned_user_id if gpu is not None else None

    # -- interactive GPU requests (Phase 9) --------------------------------

    def submit_user_request(
        self, user_id: str, display_name: str, workload: str,
        gpu_count: int, estimated_minutes: float, priority: str,
    ) -> Job:
        """Submit a real job on behalf of a logged-in demo user.

        This is the literal missing piece a prior audit of this
        project flagged: `Scheduler.submit_job` always worked, but
        nothing exposed it to a live user request. This method is
        that bridge - it validates the request, auto-registers the
        user with the engine if this is their first request in the
        currently-loaded scenario (a scenario's own users may use
        different ids, e.g. "alice"), and then calls the exact same
        `Scheduler.submit_job` + `try_allocate_all` sequence a
        scenario's own `AddJobAction` uses. It makes no allocation
        decision itself - a `gpu_count > 1` request that cannot be
        fully satisfied from the free pool is handled entirely by
        `Scheduler.try_allocate_all`'s own partial-allocation and
        resource-request logic, not by anything here.

        Raises ``ValueError`` for a ``gpu_count`` outside
        ``[MIN_USER_GPU_REQUEST, MAX_USER_GPU_REQUEST]`` (a user may
        never request the company's *entire* pool) or an unknown
        ``priority`` name.
        """
        if not (MIN_USER_GPU_REQUEST <= gpu_count <= MAX_USER_GPU_REQUEST):
            raise ValueError(
                f"gpu_count must be between {MIN_USER_GPU_REQUEST} and {MAX_USER_GPU_REQUEST} "
                f"(got gpu_count={gpu_count!r})"
            )
        try:
            priority_value = Priority[priority]
        except KeyError:
            raise ValueError(f"unknown priority {priority!r}") from None

        scheduler = self.simulator.scheduler
        if scheduler.state.get_user(user_id) is None:
            scheduler.add_user(User(user_id=user_id, name=display_name, priority=Priority.MEDIUM))

        now = self.simulator.clock.now()
        job = Job(
            job_id=f"REQ-{next(self._request_seq)}", user_id=user_id, name=workload,
            priority=priority_value, estimated_size_minutes=estimated_minutes, submitted_at=now,
            gpu_count=gpu_count,
        )
        scheduler.submit_job(job, now=now)
        scheduler.try_allocate_all(now=now)
        return job

    # -- manual admin assignment (Phase 10) -------------------------------

    def manual_assign_gpu(self, gpu_id: str, user_id: str, display_name: str) -> Job:
        """Admin-only demo/test control: place ``user_id`` directly onto
        a currently-free ``gpu_id``, establishing a starting
        arrangement before a demonstration begins. Thin, validated
        pass-through to `Scheduler.manual_assign_gpu` - auto-registers
        the user the same way `submit_user_request` does, so an admin
        can set up User A/B/C/D before any of them has ever logged in
        and submitted a request themselves.
        """
        scheduler = self.simulator.scheduler
        if scheduler.state.get_user(user_id) is None:
            scheduler.add_user(User(user_id=user_id, name=display_name, priority=Priority.MEDIUM))
        now = self.simulator.clock.now()
        return scheduler.manual_assign_gpu(gpu_id, user_id, now=now)
