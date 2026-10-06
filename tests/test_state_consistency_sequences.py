"""Long, seeded, mixed operation sequences checked for state consistency.

Every individual feature has its own focused tests. What those cannot
show is that the features still agree with each other after many
interleaved operations - a cancelled job's GPU reclaimed while a prompt
is pending, a failed GPU that was mid-prompt, a multi-GPU job partially
preempted, and so on. This drives the scheduler through a long,
deterministic (seeded) random mix of real public operations and runs
`assert_consistent` after every single step.

Only the expected, documented rejections (an operation that is simply
not valid in the current state) are tolerated; anything else - and any
inconsistency - fails the test.
"""

import random
from datetime import datetime, timedelta, timezone

import pytest

from engine.consistency import assert_consistent
from engine.models.enums import GPUStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.user import User
from engine.reclamation.policy import ConfirmationResponse
from engine.scheduler import Scheduler

T0 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
USERS = ["U1", "U2", "U3", "U4"]
GPU_IDS = [f"GPU-{i}" for i in range(1, 7)]
PRIORITIES = list(Priority)

# The only exceptions an operation may legitimately raise when it is
# simply invalid in the current state (e.g. completing a job that is
# already gone, answering a prompt that is not pending).
EXPECTED_REJECTIONS = (ValueError, KeyError)


def _build(seed: int, size_disparity_ratio=None):
    rng = random.Random(seed)
    scheduler = Scheduler(size_disparity_ratio=size_disparity_ratio)
    for user_id in USERS:
        scheduler.add_user(User(user_id=user_id, name=user_id, priority=rng.choice(PRIORITIES)))
    for gpu_id in GPU_IDS:
        scheduler.add_gpu(GPU(gpu_id=gpu_id, total_memory_mb=24_576, status=GPUStatus.IDLE))
    return scheduler, rng


def _random_op(scheduler: Scheduler, rng: random.Random, step: int, now: datetime) -> str:
    """Perform one randomly chosen public operation; return its name."""
    choice = rng.choice([
        "submit", "submit", "submit", "complete", "cancel", "reprioritise",
        "respond", "respond", "force_reclaim", "maintenance_on", "maintenance_off",
        "failure", "recover", "util", "timeouts", "manual_assign",
    ])
    jobs = list(scheduler.state.jobs.values())
    running = [j for j in jobs if j.status.name == "RUNNING"]
    gpu_id = rng.choice(GPU_IDS)

    if choice == "submit":
        user_id = rng.choice(USERS)
        job = Job(
            job_id=f"J{step}", user_id=user_id, name=f"J{step}",
            priority=rng.choice(PRIORITIES),
            estimated_size_minutes=float(rng.choice([2, 5, 10, 30, 60, 120])),
            gpu_count=rng.randint(1, 3), submitted_at=now,
        )
        scheduler.submit_job(job, now=now)
        scheduler.try_allocate_all(now=now)
    elif choice == "complete" and running:
        scheduler.complete_job(rng.choice(running).job_id, now=now)
        scheduler.try_allocate_all(now=now)
    elif choice == "cancel" and jobs:
        scheduler.cancel_job(rng.choice(jobs).job_id, now=now)
        scheduler.try_allocate_all(now=now)
    elif choice == "reprioritise" and jobs:
        scheduler.change_job_priority(rng.choice(jobs).job_id, rng.choice(PRIORITIES), now=now)
        scheduler.try_allocate_all(now=now)
    elif choice == "respond":
        pending = [g for g in GPU_IDS if scheduler.reclamation_engine.has_pending_prompt(g)]
        if pending:
            response = rng.choice([ConfirmationResponse.YES, ConfirmationResponse.NO])
            scheduler.respond_to_prompt(rng.choice(pending), response, now=now)
            scheduler.try_allocate_all(now=now)
    elif choice == "force_reclaim":
        scheduler.force_reclaim(gpu_id, now=now)
        scheduler.try_allocate_all(now=now)
    elif choice == "maintenance_on":
        scheduler.set_gpu_maintenance(gpu_id, now=now)
    elif choice == "maintenance_off":
        scheduler.clear_gpu_maintenance(gpu_id, now=now)
        scheduler.try_allocate_all(now=now)
    elif choice == "failure":
        scheduler.handle_gpu_failure(gpu_id, now=now)
        scheduler.try_allocate_all(now=now)
    elif choice == "recover":
        scheduler.recover_gpu_failure(gpu_id, now=now)
        scheduler.try_allocate_all(now=now)
    elif choice == "util":
        scheduler.record_utilization(gpu_id, float(rng.randint(0, 100)), now)
    elif choice == "timeouts":
        scheduler.check_reclamation_timeouts(now)
        scheduler.try_allocate_all(now=now)
    elif choice == "manual_assign":
        free = [g for g in GPU_IDS if scheduler.state.get_gpu(g).is_assigned is False
                and scheduler.state.get_gpu(g).status == GPUStatus.IDLE]
        if free:
            scheduler.manual_assign_gpu(rng.choice(free), rng.choice(USERS), now=now)
    return choice


@pytest.mark.parametrize("seed", [1, 7, 42, 2026])
def test_mixed_operation_sequence_stays_consistent_at_every_step(seed):
    scheduler, rng = _build(seed)
    now = T0
    for step in range(400):
        now = now + timedelta(minutes=rng.choice([0, 0.5, 1, 3, 6]))
        try:
            _random_op(scheduler, rng, step, now)
        except EXPECTED_REJECTIONS:
            pass
        assert_consistent(scheduler)


@pytest.mark.parametrize("seed", [3, 11])
def test_mixed_sequence_with_size_disparity_enabled_stays_consistent(seed):
    scheduler, rng = _build(seed, size_disparity_ratio=5.0)
    now = T0
    for step in range(400):
        now = now + timedelta(minutes=rng.choice([0, 0.5, 1, 3, 6]))
        try:
            _random_op(scheduler, rng, step, now)
        except EXPECTED_REJECTIONS:
            pass
        assert_consistent(scheduler)
