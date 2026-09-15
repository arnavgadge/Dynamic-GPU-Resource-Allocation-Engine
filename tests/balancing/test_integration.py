"""End-to-end integration tests: Phase 3 (which job) + Phase 5 (which
GPU) + the real `AllocationEngine.finalize_assignment` commit path,
and Phase 4 (reclamation) -> pool -> Phase 5 -> Phase 3, decoupled.
"""

from datetime import datetime, timedelta, timezone

from engine.allocation.engine import AllocationEngine
from engine.balancing.decision import RoutingOutcome
from engine.balancing.router import LoadBalancingRouter
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState
from engine.models.user import User
from engine.reclamation.engine import ReclamationEngine
from engine.reclamation.policy import ConfirmationResponse

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def route_and_commit(allocation_engine: AllocationEngine, router: LoadBalancingRouter, now=NOW):
    """The Phase 3 -> Phase 5 -> commit sequence the brief describes:
    Phase 3 picks the job (pure decision), Phase 5 picks the GPU (pure
    decision), then the chosen pair is committed through
    `AllocationEngine.finalize_assignment` - the exact same state
    mutation Phase 3's own self-contained `allocate_next` uses.
    """
    selection = allocation_engine.select_next_job()
    if selection is None:
        return None, None
    job, policy, alloc_reason, candidates = selection

    routing = router.route_job(job, candidate_gpus=allocation_engine.gpu_pool.all_gpus(), now=now)
    if routing.outcome is RoutingOutcome.NO_GPU_AVAILABLE:
        return None, routing

    gpu = allocation_engine.state.get_gpu(routing.selected_gpu_id)
    combined_reason = f"{alloc_reason} | routed: {routing.reason}"
    decision = allocation_engine.finalize_assignment(job, gpu, policy, combined_reason, candidates)
    return decision, routing


# ------------------------------------------------------------------
# The brief's "IMPORTANT INTEGRATION TEST"
# ------------------------------------------------------------------

def test_critical_acceptance_scenario_available_gpu_is_chosen_over_busy_low_util_one():
    state = SchedulerState()
    for user_id, name in [("A", "Alice"), ("B", "Bob"), ("C", "Charlie")]:
        state.add_user(User(user_id=user_id, name=name, priority=Priority.MEDIUM))

    gpu1 = GPU(gpu_id="GPU-1", total_memory_mb=24_576, utilization_percent=90.0,
               status=GPUStatus.ACTIVE, assigned_user_id="A", assigned_job_id="J-A")
    gpu2 = GPU(gpu_id="GPU-2", total_memory_mb=24_576, utilization_percent=5.0,
               status=GPUStatus.ACTIVE, assigned_user_id="B", assigned_job_id="J-B")
    gpu3 = GPU(gpu_id="GPU-3", total_memory_mb=24_576, utilization_percent=75.0,
               status=GPUStatus.ACTIVE, assigned_user_id="C", assigned_job_id="J-C")
    gpu4 = GPU(gpu_id="GPU-4", total_memory_mb=24_576, utilization_percent=20.0, status=GPUStatus.IDLE)

    running_jobs = {
        "J-A": Job(job_id="J-A", user_id="A", name="existing", priority=Priority.MEDIUM,
                    estimated_size_minutes=120, status=JobStatus.RUNNING, assigned_gpu_ids=["GPU-1"]),
        "J-B": Job(job_id="J-B", user_id="B", name="existing", priority=Priority.MEDIUM,
                    estimated_size_minutes=120, status=JobStatus.RUNNING, assigned_gpu_ids=["GPU-2"]),
        "J-C": Job(job_id="J-C", user_id="C", name="existing", priority=Priority.MEDIUM,
                    estimated_size_minutes=120, status=JobStatus.RUNNING, assigned_gpu_ids=["GPU-3"]),
    }
    for job in running_jobs.values():
        state.add_job(job)

    allocation_engine = AllocationEngine(state)
    for gpu in (gpu1, gpu2, gpu3, gpu4):
        allocation_engine.add_gpu(gpu)
    router = LoadBalancingRouter(state)

    new_job = Job(job_id="JOB-42", user_id="A", name="new work", priority=Priority.HIGH,
                   estimated_size_minutes=30, submitted_at=NOW)
    allocation_engine.submit_job(new_job)

    decision, routing = route_and_commit(allocation_engine, router)

    # GPU-2 (5% util) was NOT selected - it is already assigned to Bob.
    assert routing.selected_gpu_id == "GPU-4"
    assert decision.gpu_id == "GPU-4"
    assert decision.job_id == "JOB-42"

    # GPU-4 is now active and correctly assigned.
    assert gpu4.status == GPUStatus.ACTIVE
    assert gpu4.assigned_user_id == "A"
    assert gpu4.assigned_job_id == "JOB-42"
    assert new_job.status == JobStatus.RUNNING

    # Bob's existing (low-utilization but legitimate) assignment is
    # completely untouched.
    assert gpu2.assigned_user_id == "B"
    assert gpu2.assigned_job_id == "J-B"
    assert gpu2.status == GPUStatus.ACTIVE
    assert gpu2.utilization_percent == 5.0
    assert running_jobs["J-B"].status == JobStatus.RUNNING

    # Events: BALANCE (evaluate) + BALANCE (select) + ALLOC + STATUS.
    types = [e.event_type for e in state.events]
    assert types.count(EventType.BALANCE) == 2
    assert EventType.ALLOC in types
    assert EventType.STATUS in types


# ------------------------------------------------------------------
# Reclamation -> pool -> balancing -> allocation, fully decoupled
# ------------------------------------------------------------------

def test_reclaimed_gpu_flows_through_the_pool_into_the_next_routing_decision():
    state = SchedulerState()
    state.add_user(User(user_id="B", name="Bob", priority=Priority.MEDIUM))
    state.add_user(User(user_id="E", name="Erin", priority=Priority.MEDIUM))

    gpu2 = GPU(gpu_id="GPU-2", total_memory_mb=24_576, utilization_percent=3.0,
               status=GPUStatus.ACTIVE, assigned_user_id="B", assigned_job_id="J-B")
    job_b = Job(job_id="J-B", user_id="B", name="stale", priority=Priority.MEDIUM,
                estimated_size_minutes=60, status=JobStatus.RUNNING, assigned_gpu_ids=["GPU-2"])
    state.add_job(job_b)
    state.get_user("B").assigned_gpu_ids.append("GPU-2")
    state.get_user("B").running_job_ids.append("J-B")

    allocation_engine = AllocationEngine(state)
    allocation_engine.add_gpu(gpu2)  # already assigned -> not yet in the available pool
    router = LoadBalancingRouter(state)
    reclamation_engine = ReclamationEngine(state, allocation_engine=allocation_engine)

    # Before reclamation: GPU-2 must NOT be routable, despite its low
    # utilization - Phase 5 never treats "underutilized" as "available".
    job_e_before = Job(job_id="JOB-E-early", user_id="E", name="probe",
                        priority=Priority.MEDIUM, estimated_size_minutes=15, submitted_at=NOW)
    early_routing = router.route_job(job_e_before, candidate_gpus=allocation_engine.gpu_pool.all_gpus(), now=NOW)
    assert early_routing.outcome is RoutingOutcome.NO_GPU_AVAILABLE

    # Phase 4 sustains the breach and the user does not respond.
    from engine.models.utilization import UtilizationObservation

    minutes = 25
    for minute in range(minutes + 1):
        observation = UtilizationObservation(utilization_percent=1.0, timestamp=NOW + timedelta(minutes=minute))
        reclamation_engine.record_utilization(gpu2, observation)
    breach_confirmed_at = NOW + timedelta(minutes=minutes)
    reclamation_engine.check_timeouts(now=breach_confirmed_at + timedelta(minutes=10))

    assert gpu2.status == GPUStatus.IDLE
    assert gpu2.assigned_user_id is None
    assert job_b.status == JobStatus.RECLAIMED

    # Now Job E arrives and Phase 5 should prefer the freed GPU-2.
    allocation_engine.submit_job(Job(job_id="JOB-E", user_id="E", name="new work",
                                       priority=Priority.MEDIUM, estimated_size_minutes=15,
                                       submitted_at=breach_confirmed_at + timedelta(minutes=11)))

    decision, routing = route_and_commit(allocation_engine, router, now=NOW + timedelta(hours=1))

    assert routing.selected_gpu_id == "GPU-2"
    assert decision.gpu_id == "GPU-2"
    assert gpu2.assigned_user_id == "E"
    assert gpu2.status == GPUStatus.ACTIVE
