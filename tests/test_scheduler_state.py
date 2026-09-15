from engine.models.assignment import GPUAssignment
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.event import Event
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState
from engine.models.user import User


def build_minimal_state() -> SchedulerState:
    state = SchedulerState()

    alice = User(user_id="U1", name="Alice", priority=Priority.HIGH)
    bob = User(user_id="U2", name="Bob", priority=Priority.MEDIUM)
    state.add_user(alice)
    state.add_user(bob)

    gpu0 = GPU(gpu_id="GPU0", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
               assigned_user_id="U1", assigned_job_id="J1")
    gpu1 = GPU(gpu_id="GPU1", total_memory_mb=24_576, status=GPUStatus.IDLE)
    state.add_gpu(gpu0)
    state.add_gpu(gpu1)

    running_job = Job(job_id="J1", user_id="U1", name="ML Training", priority=Priority.HIGH,
                       estimated_size_minutes=60, status=JobStatus.RUNNING, assigned_gpu_ids=["GPU0"])
    waiting_job = Job(job_id="J2", user_id="U2", name="Excel", priority=Priority.MEDIUM,
                       estimated_size_minutes=10, status=JobStatus.WAITING)
    state.add_job(running_job)
    state.add_job(waiting_job)

    state.add_assignment(GPUAssignment(assignment_id="A1", gpu_id="GPU0", user_id="U1", job_id="J1"))
    state.log_event(Event(event_id="E1", event_type=EventType.ALLOC, gpu_id="GPU0",
                            user_id="U1", job_id="J1", message="Assigned GPU0 to Alice"))

    return state


def test_scheduler_state_contains_the_complete_system_state():
    state = build_minimal_state()

    assert set(state.gpus) == {"GPU0", "GPU1"}
    assert set(state.users) == {"U1", "U2"}
    assert set(state.jobs) == {"J1", "J2"}
    assert set(state.assignments) == {"A1"}
    assert len(state.events) == 1


def test_get_gpus_for_user_answers_which_gpus_a_user_holds():
    state = build_minimal_state()

    assert [g.gpu_id for g in state.get_gpus_for_user("U1")] == ["GPU0"]
    assert state.get_gpus_for_user("U2") == []


def test_get_job_on_gpu_answers_what_is_running_on_a_gpu():
    state = build_minimal_state()

    job = state.get_job_on_gpu("GPU0")
    assert job is not None
    assert job.job_id == "J1"
    assert state.get_job_on_gpu("GPU1") is None


def test_get_waiting_jobs_returns_only_waiting_jobs():
    state = build_minimal_state()

    waiting = state.get_waiting_jobs()
    assert [job.job_id for job in waiting] == ["J2"]


def test_get_active_assignment_for_gpu():
    state = build_minimal_state()

    assignment = state.get_active_assignment_for_gpu("GPU0")
    assert assignment is not None
    assert assignment.assignment_id == "A1"
    assert state.get_active_assignment_for_gpu("GPU1") is None


def test_get_events_for_gpu_filters_by_gpu():
    state = build_minimal_state()

    events = state.get_events_for_gpu("GPU0")
    assert len(events) == 1
    assert events[0].event_id == "E1"
    assert state.get_events_for_gpu("GPU1") == []
