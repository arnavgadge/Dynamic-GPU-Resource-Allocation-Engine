"""Simulator: runs a Scenario against the real Scheduler over simulated time.

    "The simulator generates inputs; the scheduler generates decisions."

`Simulator` owns a `SimulationClock` and a `Scheduler` (Phase 6's own
new orchestrator over Phase 3/4/5's real engines). Advancing simulated
time causes any scenario actions now due to fire - each one calls
straight into `Scheduler`, which calls straight into
`AllocationEngine`/`ReclamationEngine`/`LoadBalancingRouter`. Nothing
in this file computes a score, checks a threshold, or picks a GPU;
it only decides *when* to hand the engines their next piece of input,
and lets them react (`try_allocate_all`, `check_reclamation_timeouts`)
after every action.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from engine.models.assignment import GPUAssignment
from engine.models.enums import JobStatus
from engine.models.scheduler_state import SchedulerState
from engine.scheduler import Scheduler
from engine.simulation.actions import (
    AddJobAction,
    CompleteJobAction,
    ScenarioActionTypes,
    UserResponseAction,
    UtilizationAction,
)
from engine.simulation.clock import SimulationClock
from engine.simulation.scenario import Scenario

#: Default step size for `tick()`/`run_until()` when the caller
#: doesn't specify one - fine-grained enough that no scheduled action
#: is ever skipped over, coarse enough that reaching a multi-hour
#: Tier 2 reclamation window doesn't take thousands of steps.
DEFAULT_TICK = timedelta(minutes=1)


class Simulator:
    """Drives one `Scenario` against a fresh `Scheduler`, deterministically.

    Complexity
    ----------
    reset                  O(g + u + j + a log a) - rebuilds the scheduler from
                              g GPUs/u users/j initial jobs, and sorts a actions once
    advance(delta)            O(d + k) - d = actions due in this step (each executed
                              once), plus whatever `try_allocate_all`/
                              `check_reclamation_timeouts` cost for the resulting
                              state (k candidates - see `Scheduler`'s own complexity)
    tick / run_until          one or more `advance` calls - same per-call cost
    snapshot                  O(1) - returns the live `SchedulerState`, never a copy
    """

    def __init__(self, scenario: Scenario, speed: float = 1.0) -> None:
        self.scenario = scenario
        self.speed = speed
        self.scheduler: Scheduler
        self.clock: SimulationClock
        self._pending_actions: List = []
        self._next_action_index = 0
        self.reset()

    # -- lifecycle -----------------------------------------------------

    def reset(self) -> None:
        """Rebuild the scheduler and clock from the scenario's initial
        conditions, discarding everything a previous run did. O(g+u+j+a log a).
        """
        gpus, users, jobs, actions = self.scenario.fresh_copies()

        self.clock = SimulationClock(start=self.scenario.start_time, speed=self.speed)
        self.scheduler = Scheduler()

        for user in users:
            self.scheduler.add_user(user)
        for gpu in gpus:
            self.scheduler.add_gpu(gpu)
        for job in jobs:
            self._seed_initial_job(job)

        for action in actions:
            if not isinstance(action, ScenarioActionTypes):
                raise TypeError(f"unknown scenario action type: {type(action)!r}")
        self._pending_actions = sorted(actions, key=lambda action: action.offset)
        self._next_action_index = 0

        # Let the scheduler react to whatever initial state/jobs exist,
        # and fire any actions scheduled at or before T0, before the
        # caller ever calls `advance()`.
        self.scheduler.try_allocate_all(now=self.clock.now())
        self._process_due_actions()

    def _seed_initial_job(self, job) -> None:
        """Place one of the scenario's ``initial_jobs`` into the fresh state.

        A ``WAITING`` job is a real submission and goes through
        `Scheduler.submit_job` like any job the simulation adds later.
        A ``RUNNING`` job describes a pre-existing assignment the
        scenario starts *from* (its GPU was already constructed with
        `assigned_user_id`/`assigned_job_id` set, so `add_gpu` above
        already kept it out of the available pool) - this only wires
        up the matching `User`/`GPUAssignment` bookkeeping, the same
        fields `AllocationEngine._assign_gpu` would have set had that
        assignment actually happened during this simulation. Nothing
        here is a scheduling decision; it is scenario setup, exactly
        like `engine/sample_data.py` seeds Phase 1's sample state.
        """
        if job.status == JobStatus.WAITING:
            self.scheduler.submit_job(job, now=self.clock.now())
            return

        if job.status != JobStatus.RUNNING:
            raise ValueError(
                f"initial job {job.job_id!r} must be WAITING or RUNNING, got {job.status.value}"
            )

        state = self.scheduler.state
        state.add_job(job)
        if not job.assigned_gpu_ids:
            return

        user = state.get_user(job.user_id)
        if user is not None and job.job_id not in user.running_job_ids:
            user.running_job_ids.append(job.job_id)

        # A RUNNING initial job may already hold more than one GPU
        # (Phase 10's manual multi-GPU demo setup) - wire up every one
        # of them exactly as `AllocationEngine._assign_gpu` would have,
        # not just the first.
        for gpu_id in job.assigned_gpu_ids:
            if user is not None and gpu_id not in user.assigned_gpu_ids:
                user.assigned_gpu_ids.append(gpu_id)
            state.add_assignment(GPUAssignment(
                assignment_id=f"seed-{job.job_id}-{gpu_id}", gpu_id=gpu_id,
                user_id=job.user_id, job_id=job.job_id, created_at=self.scenario.start_time,
            ))

    # -- time progression --------------------------------------------

    def advance(self, delta: timedelta) -> None:
        """Move simulated time forward by exactly ``delta`` (never real
        wall-clock time), then process whatever that makes due.

        `_execute` already calls `Scheduler.try_allocate_all` after any
        action that could plausibly free a GPU or add a job (job
        arrival, a prompt response, a completion) - the only other way
        a GPU can turn newly available without an explicit action is a
        no-response reclaim firing purely because the grace period
        elapsed, which is why that case gets its own follow-up call
        here rather than trying on every single tick regardless of
        whether anything changed.
        """
        self.clock.advance(delta)
        self._process_due_actions()
        now = self.clock.now()
        timed_out_reclaims = self.scheduler.check_reclamation_timeouts(now)
        if timed_out_reclaims:
            self.scheduler.try_allocate_all(now=now)

    def tick(self, step: Optional[timedelta] = None) -> None:
        """Advance by one step (`DEFAULT_TICK` unless overridden)."""
        self.advance(step if step is not None else DEFAULT_TICK)

    def run_until(self, target_time: datetime, step: timedelta = DEFAULT_TICK) -> None:
        """Advance in ``step``-sized increments until ``target_time``.

        Never overshoots: the final increment is clamped so the clock
        lands exactly on ``target_time``, not past it.
        """
        while self.clock.now() < target_time:
            remaining = target_time - self.clock.now()
            self.advance(step if step < remaining else remaining)

    def run_to_completion(self, step: timedelta = DEFAULT_TICK, max_ticks: int = 100_000) -> None:
        """Advance until every scheduled action has fired.

        ``max_ticks`` is a safety bound against a scenario whose last
        action never actually gets reached (e.g. a typo'd offset) -
        this raises rather than looping forever.
        """
        ticks = 0
        while self._next_action_index < len(self._pending_actions):
            if ticks >= max_ticks:
                raise RuntimeError(
                    f"run_to_completion exceeded {max_ticks} ticks without exhausting "
                    f"the scenario's actions - check the scenario's action offsets"
                )
            self.advance(step)
            ticks += 1

    # -- internals -----------------------------------------------------

    def _process_due_actions(self) -> None:
        now = self.clock.now()
        while self._next_action_index < len(self._pending_actions):
            action = self._pending_actions[self._next_action_index]
            due_at = self.scenario.start_time + action.offset
            if due_at > now:
                break
            self._execute(action, due_at)
            self._next_action_index += 1

    def _execute(self, action, at: datetime) -> None:
        if isinstance(action, AddJobAction):
            self.scheduler.submit_job(action.job, now=at)
            self.scheduler.try_allocate_all(now=at)
        elif isinstance(action, UtilizationAction):
            if self.scheduler.state.get_gpu(action.gpu_id) is None:
                raise ValueError(f"scenario references unknown GPU {action.gpu_id!r}")
            self.scheduler.record_utilization(action.gpu_id, action.utilization_percent, at, action.memory_used_mb)
        elif isinstance(action, UserResponseAction):
            self.scheduler.respond_to_prompt(action.gpu_id, action.response, now=at)
            self.scheduler.try_allocate_all(now=at)
        elif isinstance(action, CompleteJobAction):
            self.scheduler.complete_job(action.job_id, now=at)
            self.scheduler.try_allocate_all(now=at)
        else:
            raise TypeError(f"unknown scenario action type: {type(action)!r}")

    # -- state -----------------------------------------------------------

    def snapshot(self) -> SchedulerState:
        """The live scheduler state - GPUs, users, jobs, assignments,
        event log, engine status. Deliberately not a copy: this
        project's convention (Phase 1 onward) is that `SchedulerState`
        already *is* the one source of truth a frontend would read;
        duplicating it here would just be a second thing that could
        drift out of sync with the first.
        """
        return self.scheduler.state

    def has_pending_actions(self) -> bool:
        return self._next_action_index < len(self._pending_actions)
