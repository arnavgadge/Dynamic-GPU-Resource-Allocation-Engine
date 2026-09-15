"""A small, hand-built SchedulerState used to exercise the domain model.

This is test/demo data only - it is NOT hardwired into the scheduler
itself. The real system must handle an arbitrary number of GPUs,
users and jobs; this module just gives Phase 1's models something
concrete to be built and inspected with.

Scenario (matches the project report's running example):

    4 GPUs, 3 users.

    GPU 0  92% util  Alice    ML Training       ACTIVE
    GPU 1   7% util  Bob      Excel             IDLE_WARNING
    GPU 2  68% util  Bob      Model Training    ACTIVE
    GPU 3  15% util  Charlie  Data Processing   ACTIVE

    Users:
        U1 Alice   HIGH
        U2 Bob     MEDIUM
        U3 Charlie LOW

    Charlie has also submitted a second job ("Video Editing") that is
    still WAITING for GPU capacity, to demonstrate the waiting-job
    path with no extra users.
"""

from datetime import datetime, timezone

from engine.models.assignment import GPUAssignment
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.event import Event
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState
from engine.models.user import User
from engine.models.utilization import UtilizationObservation


def build_sample_state() -> SchedulerState:
    """Build and return the sample 4-GPU / 3-user scheduler state."""
    state = SchedulerState()
    now = datetime.now(timezone.utc)

    # --- Users -----------------------------------------------------
    alice = User(user_id="U1", name="Alice", priority=Priority.HIGH)
    bob = User(user_id="U2", name="Bob", priority=Priority.MEDIUM)
    charlie = User(user_id="U3", name="Charlie", priority=Priority.LOW)
    for user in (alice, bob, charlie):
        state.add_user(user)

    # --- GPUs --------------------------------------------------------
    gpu0 = GPU(gpu_id="GPU0", total_memory_mb=24_576, status=GPUStatus.ACTIVE)
    gpu1 = GPU(gpu_id="GPU1", total_memory_mb=24_576, status=GPUStatus.IDLE_WARNING)
    gpu2 = GPU(gpu_id="GPU2", total_memory_mb=24_576, status=GPUStatus.ACTIVE)
    gpu3 = GPU(gpu_id="GPU3", total_memory_mb=24_576, status=GPUStatus.ACTIVE)
    for gpu in (gpu0, gpu1, gpu2, gpu3):
        state.add_gpu(gpu)

    # --- Jobs --------------------------------------------------------
    j1 = Job(job_id="J1", user_id=alice.user_id, name="ML Training",
              priority=alice.priority, estimated_size_minutes=180,
              status=JobStatus.RUNNING, started_at=now, assigned_gpu_ids=[gpu0.gpu_id])
    j2 = Job(job_id="J2", user_id=bob.user_id, name="Excel",
              priority=bob.priority, estimated_size_minutes=15,
              status=JobStatus.RUNNING, started_at=now, assigned_gpu_ids=[gpu1.gpu_id])
    j3 = Job(job_id="J3", user_id=bob.user_id, name="Model Training",
              priority=bob.priority, estimated_size_minutes=120,
              status=JobStatus.RUNNING, started_at=now, assigned_gpu_ids=[gpu2.gpu_id])
    j4 = Job(job_id="J4", user_id=charlie.user_id, name="Data Processing",
              priority=charlie.priority, estimated_size_minutes=45,
              status=JobStatus.RUNNING, started_at=now, assigned_gpu_ids=[gpu3.gpu_id])
    j5 = Job(job_id="J5", user_id=charlie.user_id, name="Video Editing",
              priority=charlie.priority, estimated_size_minutes=60,
              status=JobStatus.WAITING)
    for job in (j1, j2, j3, j4, j5):
        state.add_job(job)

    # --- Wire up current assignment pointers on GPU/User -------------
    gpu0.assigned_user_id, gpu0.assigned_job_id = alice.user_id, j1.job_id
    gpu1.assigned_user_id, gpu1.assigned_job_id = bob.user_id, j2.job_id
    gpu2.assigned_user_id, gpu2.assigned_job_id = bob.user_id, j3.job_id
    gpu3.assigned_user_id, gpu3.assigned_job_id = charlie.user_id, j4.job_id

    alice.assigned_gpu_ids = [gpu0.gpu_id]
    alice.running_job_ids = [j1.job_id]
    bob.assigned_gpu_ids = [gpu1.gpu_id, gpu2.gpu_id]
    bob.running_job_ids = [j2.job_id, j3.job_id]
    charlie.assigned_gpu_ids = [gpu3.gpu_id]
    charlie.running_job_ids = [j4.job_id]

    # --- Utilization observations -------------------------------------
    gpu0.record_observation(UtilizationObservation(utilization_percent=92.0, memory_used_mb=22_528, timestamp=now))
    gpu1.record_observation(UtilizationObservation(utilization_percent=7.0, memory_used_mb=2_048, timestamp=now))
    gpu2.record_observation(UtilizationObservation(utilization_percent=68.0, memory_used_mb=16_384, timestamp=now))
    gpu3.record_observation(UtilizationObservation(utilization_percent=15.0, memory_used_mb=6_144, timestamp=now))

    # --- Assignment records -------------------------------------------
    state.add_assignment(GPUAssignment(assignment_id="A1", gpu_id=gpu0.gpu_id, user_id=alice.user_id, job_id=j1.job_id, created_at=now))
    state.add_assignment(GPUAssignment(assignment_id="A2", gpu_id=gpu1.gpu_id, user_id=bob.user_id, job_id=j2.job_id, created_at=now))
    state.add_assignment(GPUAssignment(assignment_id="A3", gpu_id=gpu2.gpu_id, user_id=bob.user_id, job_id=j3.job_id, created_at=now))
    state.add_assignment(GPUAssignment(assignment_id="A4", gpu_id=gpu3.gpu_id, user_id=charlie.user_id, job_id=j4.job_id, created_at=now))

    # --- Event log -----------------------------------------------------
    state.log_event(Event(event_id="E1", event_type=EventType.ALLOC, gpu_id=gpu0.gpu_id,
                            user_id=alice.user_id, job_id=j1.job_id, timestamp=now,
                            message="Assigned GPU0 to Alice", reason="Initial allocation"))
    state.log_event(Event(event_id="E2", event_type=EventType.ALLOC, gpu_id=gpu1.gpu_id,
                            user_id=bob.user_id, job_id=j2.job_id, timestamp=now,
                            message="Assigned GPU1 to Bob", reason="Initial allocation"))
    state.log_event(Event(event_id="E3", event_type=EventType.ALLOC, gpu_id=gpu2.gpu_id,
                            user_id=bob.user_id, job_id=j3.job_id, timestamp=now,
                            message="Assigned GPU2 to Bob", reason="Initial allocation"))
    state.log_event(Event(event_id="E4", event_type=EventType.ALLOC, gpu_id=gpu3.gpu_id,
                            user_id=charlie.user_id, job_id=j4.job_id, timestamp=now,
                            message="Assigned GPU3 to Charlie", reason="Initial allocation"))
    state.log_event(Event(event_id="E5", event_type=EventType.MONITOR, gpu_id=gpu1.gpu_id,
                            user_id=bob.user_id, job_id=j2.job_id, timestamp=now,
                            message="GPU1 utilization at 7% - watching for sustained idle",
                            reason="Utilization below Tier 2 threshold"))
    state.log_event(Event(event_id="E6", event_type=EventType.REQUEST, user_id=charlie.user_id,
                            job_id=j5.job_id, timestamp=now,
                            message="Charlie requested GPU capacity for Video Editing",
                            reason="No free GPU in pool"))

    return state
