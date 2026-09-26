"""Reusable state-consistency invariants (Day 11).

A pure, read-only diagnostic - it never mutates `SchedulerState` or
any DSA structure, and it implements no scheduling policy of its own.
It exists so "did that admin operation leave the world consistent?"
is one shared, documented check instead of an ad-hoc assertion block
copy-pasted into every test file (Day 4's `_assert_fully_consistent`
was exactly that - a good check, but local to one test module and
easy to drift from). Every invariant here was already true by
construction across Days 1-10; this module only makes each one
checkable in one place, for tests and for an admin diagnostic to use
alike.
"""

from dataclasses import dataclass, field
from typing import List

from engine.scheduler import Scheduler


@dataclass
class ConsistencyReport:
    """The result of one `check_consistency` call. Falsy-friendly:
    ``if report:`` is true exactly when something is wrong."""

    violations: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return not self.violations

    def __str__(self) -> str:
        return "consistent" if self.violations == [] else "; ".join(self.violations)


def check_consistency(scheduler: Scheduler) -> ConsistencyReport:
    """Walk `SchedulerState` and every index derived from it, and
    report every invariant violation found - never raises itself (see
    `assert_consistent` for that). Safe to call after any sequence of
    operations, admin or ordinary.

    Checks:

    - A GPU never belongs to two jobs, and never belongs to a job that
      doesn't also list it back (`GPU.assigned_job_id` <->
      `Job.assigned_gpu_ids`) - checked in both directions.
    - `GPU.assigned_user_id` agrees with the holding job's `user_id`,
      and with that user's own `assigned_gpu_ids`.
    - `0 <= len(Job.assigned_gpu_ids) <= Job.gpu_count` (never more
      GPUs held than requested) and `gpus_still_needed ==
      gpu_count - len(assigned_gpu_ids)` (the model property, not a
      second computation - see `engine/models/job.py`).
    - A `RUNNING` job holds at least one GPU; a `WAITING` job holds
      none; every GPU a job claims to hold actually exists and points
      back at that job.
    - `AllocationEngine`'s `UserGPUIndex` (the HashMap-backed reverse
      index) agrees with `User.assigned_gpu_ids` for every user - the
      exact class of bug Day 4/9's fixes closed for `complete_job`/
      preemption specifically; this is the general-purpose version of
      that same check.
    - Every job in the waiting FIFO (`AllocationEngine.waiting_job_ids`)
      is genuinely `JobStatus.WAITING` and appears exactly once - no
      stale or duplicate entries.
    - A GPU marked `MAINTENANCE` or `UNAVAILABLE` is never assigned to
      anyone.
    """
    state = scheduler.state
    violations: List[str] = []

    # -- GPU <-> Job <-> User, both directions ---------------------------
    for gpu_id, gpu in state.gpus.items():
        if gpu.gpu_id != gpu_id:
            violations.append(f"{gpu_id}: GPU.gpu_id {gpu.gpu_id!r} does not match its own state key")

        if gpu.status.value in ("MAINTENANCE", "UNAVAILABLE") and gpu.is_assigned:
            violations.append(f"{gpu_id}: status={gpu.status.value} but still assigned to {gpu.assigned_user_id!r}")

        if gpu.assigned_job_id is not None:
            job = state.get_job(gpu.assigned_job_id)
            if job is None:
                violations.append(f"{gpu_id}: assigned_job_id {gpu.assigned_job_id!r} does not exist")
            elif gpu_id not in job.assigned_gpu_ids:
                violations.append(f"{gpu_id}: claims job {job.job_id!r}, but that job does not list {gpu_id} back")
            elif gpu.assigned_user_id != job.user_id:
                violations.append(
                    f"{gpu_id}: assigned_user_id {gpu.assigned_user_id!r} != holding job's user_id {job.user_id!r}"
                )

        if gpu.assigned_user_id is not None:
            user = state.get_user(gpu.assigned_user_id)
            if user is None:
                violations.append(f"{gpu_id}: assigned_user_id {gpu.assigned_user_id!r} does not exist")
            elif gpu_id not in user.assigned_gpu_ids:
                violations.append(f"{gpu_id}: assigned to {user.user_id!r}, but that user does not list {gpu_id} back")

    owners: dict = {}
    for job_id, job in state.jobs.items():
        if job.job_id != job_id:
            violations.append(f"{job_id}: Job.job_id {job.job_id!r} does not match its own state key")

        for gpu_id in job.assigned_gpu_ids:
            if gpu_id in owners:
                violations.append(f"{gpu_id}: held by both {owners[gpu_id]!r} and {job_id!r} simultaneously")
            owners[gpu_id] = job_id
            gpu = state.get_gpu(gpu_id)
            if gpu is None:
                violations.append(f"{job_id}: claims to hold {gpu_id}, which does not exist")
            elif gpu.assigned_job_id != job_id:
                violations.append(f"{job_id}: claims to hold {gpu_id}, but that GPU points at {gpu.assigned_job_id!r}")

        # requested/allocated/remaining
        if len(job.assigned_gpu_ids) > job.gpu_count:
            violations.append(
                f"{job_id}: holds {len(job.assigned_gpu_ids)} GPU(s), more than its own request of {job.gpu_count}"
            )
        if job.gpus_still_needed != max(job.gpu_count - len(job.assigned_gpu_ids), 0):
            violations.append(f"{job_id}: gpus_still_needed disagrees with gpu_count - len(assigned_gpu_ids)")

        if job.status.value == "RUNNING" and not job.assigned_gpu_ids:
            violations.append(f"{job_id}: status=RUNNING but holds no GPU")
        if job.status.value == "WAITING" and job.assigned_gpu_ids:
            violations.append(f"{job_id}: status=WAITING but still holds {job.assigned_gpu_ids}")

        if state.get_user(job.user_id) is None:
            violations.append(f"{job_id}: user_id {job.user_id!r} does not exist")

    # -- UserGPUIndex (HashMap) agrees with User.assigned_gpu_ids --------
    for user_id, user in state.users.items():
        if user.user_id != user_id:
            violations.append(f"{user_id}: User.user_id {user.user_id!r} does not match its own state key")
        indexed = sorted(scheduler.allocation_engine.get_gpus_for_user(user_id))
        actual = sorted(user.assigned_gpu_ids)
        if indexed != actual:
            violations.append(
                f"{user_id}: UserGPUIndex reports {indexed}, but User.assigned_gpu_ids is {actual}"
            )
        for gpu_id in user.assigned_gpu_ids:
            gpu = state.get_gpu(gpu_id)
            if gpu is None or gpu.assigned_user_id != user_id:
                violations.append(f"{user_id}: claims {gpu_id}, which does not point back at this user")

    # -- waiting FIFO: no stale or duplicate entries ---------------------
    waiting_ids = scheduler.allocation_engine.waiting_job_ids()
    if len(waiting_ids) != len(set(waiting_ids)):
        violations.append(f"waiting queue has duplicate job ids: {waiting_ids}")
    for job_id in waiting_ids:
        job = state.get_job(job_id)
        if job is None:
            violations.append(f"waiting queue references unknown job {job_id!r}")
        elif job.status.value != "WAITING":
            violations.append(f"waiting queue contains {job_id!r}, but its status is {job.status.value}, not WAITING")

    return ConsistencyReport(violations)


def assert_consistent(scheduler: Scheduler) -> None:
    """Like `check_consistency`, but raises ``AssertionError`` with
    every violation listed if the state is not consistent - the form
    a test wants."""
    report = check_consistency(scheduler)
    assert report, f"SchedulerState is inconsistent: {report}"
