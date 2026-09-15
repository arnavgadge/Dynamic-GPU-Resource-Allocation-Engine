"""Demo: build the sample scheduler state and print a summary.

Phase 1 section prints the domain model itself. Phase 2 section feeds
that same sample data into the DSA structures from `engine/dsa/` to
show they work together. Phase 3 section runs a fresh scenario
through the real `AllocationEngine` and prints its full decision
trace. Phase 4 section runs a GPU through a sustained idle breach and
the full confirm-then-reclaim flow. Phase 5 section shows the load
balancer routing a new job to a genuinely available GPU instead of
one that merely looks idle. Nothing before Phase 3 decides who gets a
GPU, nothing before Phase 4 ever reclaims one, and nothing before
Phase 5 ever prefers a GPU by utilization alone. Phase 6 section runs
the full end-to-end scenario through the deterministic simulator -
everything before it in this file is driven by explicit calls; from
here on, a `Simulator` drives the same real engines through simulated
time instead.
"""

from datetime import datetime, timedelta, timezone

from engine.allocation.engine import AllocationEngine
from engine.balancing.router import LoadBalancingRouter
from engine.dsa.gpu_pool import GPUPool
from engine.dsa.gpu_utilization_heap import GPUUtilizationHeap
from engine.dsa.priority_queue import PriorityQueue
from engine.dsa.reclaim_history import ReclaimHistory
from engine.dsa.sliding_window import UtilizationSlidingWindow
from engine.dsa.user_gpu_index import UserGPUIndex
from engine.dsa.waiting_job_queue import WaitingJobQueue
from engine.models.enums import EventType, GPUStatus, JobStatus, Priority
from engine.models.event import Event
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.scheduler_state import SchedulerState
from engine.models.user import User
from engine.models.utilization import UtilizationObservation
from engine.reclamation.engine import ReclamationEngine
from engine.reclamation.policy import ConfirmationResponse, DEFAULT_RECLAMATION_POLICY, ReclamationPolicy, ReclamationTierPolicy
from engine.sample_data import build_sample_state
from engine.simulation import load_default_registry


def main() -> None:
    state = build_sample_state()

    print(f"Engine status: {state.engine_status.value}\n")

    print("GPU Pool")
    print("-" * 60)
    for gpu in state.gpus.values():
        user = state.get_user(gpu.assigned_user_id) if gpu.assigned_user_id else None
        job = state.get_job_on_gpu(gpu.gpu_id)
        print(
            f"  {gpu.gpu_id:5s} util={gpu.utilization_percent:5.1f}%  "
            f"mem={gpu.memory_used_mb:.0f}/{gpu.total_memory_mb:.0f}MB  "
            f"status={gpu.status.value:13s}  "
            f"user={(user.name if user else '-'):8s}  "
            f"job={(job.name if job else '-')}"
        )

    print("\nUsers")
    print("-" * 60)
    for user in state.users.values():
        gpu_ids = ", ".join(g.gpu_id for g in state.get_gpus_for_user(user.user_id)) or "-"
        print(f"  {user.user_id} {user.name:10s} priority={user.priority.name:8s} gpus=[{gpu_ids}]")

    print("\nWaiting jobs")
    print("-" * 60)
    for job in state.get_waiting_jobs():
        owner = state.get_user(job.user_id)
        print(f"  {job.job_id} {job.name!r} owner={owner.name} waiting={job.waiting_time}")

    print("\nEvent log")
    print("-" * 60)
    for event in state.events:
        print(f"  [{event.timestamp:%H:%M:%S}] {event.event_type.value:8s} {event.message}")

    demo_dsa(state)
    demo_allocation()
    demo_reclamation()
    demo_load_balancing()
    demo_simulation()


def demo_dsa(state) -> None:
    """Phase 2 demo: feed the sample state through each DSA structure."""

    print("\n" + "=" * 60)
    print("Phase 2 - DSA structures")
    print("=" * 60)

    # Linked-list GPU pool + min-heap of utilization.
    pool = GPUPool()
    util_heap = GPUUtilizationHeap()
    for gpu in state.gpus.values():
        pool.add_gpu(gpu)
        util_heap.insert_gpu(gpu)

    print(f"\nGPU pool (linked list) size: {pool.size()}")
    least = util_heap.peek_least_utilized()
    print(f"Least-utilized GPU (min-heap): {least.gpu_id} at {least.utilization_percent:.1f}%")

    # HashMap-backed user -> GPU index, built from the assignment records.
    index = UserGPUIndex()
    for assignment in state.assignments.values():
        index.add_assignment(assignment)
    for user in state.users.values():
        print(f"HashMap lookup: {user.name:8s} -> {index.get_gpus_for_user(user.user_id)}")

    # Waiting jobs into a priority queue (comparator = existing Priority,
    # not yet the Phase 3 allocation-score formula) and an FCFS queue.
    priority_queue = PriorityQueue(key_fn=lambda job: job.priority)
    fcfs_queue = WaitingJobQueue()
    for job in state.get_waiting_jobs():
        priority_queue.insert(job)
        fcfs_queue.enqueue_job(job)
    print(f"\nPriority queue peek (waiting jobs): {priority_queue.peek_best().name!r}")
    print(f"FCFS queue peek (waiting jobs): {fcfs_queue.peek_job().name!r}")

    # Stack of reclaim history.
    history = ReclaimHistory()
    history.record_reclaim(Event(event_id="DEMO1", event_type=EventType.RECLAIM,
                                  gpu_id="GPU1", message="Reclaimed GPU1 (demo only)"))
    print(f"\nReclaim history top: {history.peek_last().message!r}")

    # Sliding window over one GPU's utilization history.
    window = UtilizationSlidingWindow(window_duration=timedelta(minutes=30))
    gpu1 = state.get_gpu("GPU1")
    for observation in gpu1.utilization_history:
        window.add_observation(observation)
    print(f"Sliding window size for {gpu1.gpu_id}: {window.size()} observation(s)")


def demo_allocation() -> None:
    """Phase 3 demo: one GPU becomes available, three jobs compete for it.

    Fresh scenario (not the Phase 1/2 sample state, whose GPUs are all
    already busy) so there is an actual decision to trace: three
    users of different priority and job size, one available GPU.
    """
    print("\n" + "=" * 60)
    print("Phase 3 - Allocation Engine")
    print("=" * 60)

    engine = AllocationEngine(SchedulerState())
    for user_id, name, priority in [("U1", "Alice", Priority.HIGH),
                                     ("U2", "Bob", Priority.MEDIUM),
                                     ("U3", "Charlie", Priority.LOW)]:
        engine.state.add_user(User(user_id=user_id, name=name, priority=priority))

    engine.add_gpu(GPU(gpu_id="GPU0", total_memory_mb=24_576))

    now = datetime.now(timezone.utc)
    engine.submit_job(Job(job_id="J1", user_id="U1", name="ML Training", priority=Priority.HIGH,
                            estimated_size_minutes=20, submitted_at=now))
    engine.submit_job(Job(job_id="J2", user_id="U2", name="Excel", priority=Priority.MEDIUM,
                            estimated_size_minutes=15, submitted_at=now + timedelta(minutes=1)))
    engine.submit_job(Job(job_id="J3", user_id="U3", name="Data Processing", priority=Priority.LOW,
                            estimated_size_minutes=200, submitted_at=now + timedelta(minutes=2)))

    decision = engine.allocate_next()

    print(f"\nGPU {decision.gpu_id} became available. Candidates:\n")
    for candidate in decision.candidates:
        score_text = f"{candidate.score:.3f}" if candidate.score is not None else "n/a"
        print(f"  {candidate.job_id:4s} priority={candidate.priority.name:8s} "
              f"size={candidate.size_minutes:.0f} min  score={score_text}")

    print(f"\nPolicy: {decision.policy.value}")
    print(f"Winner: {decision.job_id}")
    print(f"Reason: {decision.reason}")
    print(f"Event:  [{decision.event.event_type.value}] {decision.event.message}")


def demo_reclamation() -> None:
    """Phase 4 demo: a GPU sits idle long enough to trigger Tier 1, gets
    prompted, and - with no response - is auto-reclaimed back into the
    allocation pool."""
    print("\n" + "=" * 60)
    print("Phase 4 - Reclamation Engine")
    print("=" * 60)

    state = SchedulerState()
    state.add_user(User(user_id="U1", name="Alice", priority=Priority.HIGH))
    gpu = GPU(gpu_id="GPU0", total_memory_mb=24_576, status=GPUStatus.ACTIVE,
              assigned_user_id="U1", assigned_job_id="J1")
    job = Job(job_id="J1", user_id="U1", name="ML Training", priority=Priority.HIGH,
              estimated_size_minutes=180, status=JobStatus.RUNNING, assigned_gpu_ids=["GPU0"])
    state.add_gpu(gpu)
    state.add_job(job)
    state.get_user("U1").assigned_gpu_ids.append("GPU0")
    state.get_user("U1").running_job_ids.append("J1")

    allocation_engine = AllocationEngine(state)
    allocation_engine.add_gpu(gpu)  # already assigned -> stays out of the available pool for now

    reclaim_engine = ReclamationEngine(state, allocation_engine=allocation_engine)
    tier1 = DEFAULT_RECLAMATION_POLICY.tier1
    grace = DEFAULT_RECLAMATION_POLICY.no_response_grace_period
    print(f"\nPolicy: Tier 1 < {tier1.utilization_threshold_percent:.0f}% sustained for "
          f"{tier1.sustained_duration}; no-response grace period {grace}")

    now = datetime.now(timezone.utc)
    minutes = int(tier1.sustained_duration.total_seconds() // 60)
    decision = None
    for minute in range(minutes + 1):
        observation = UtilizationObservation(utilization_percent=1.0, timestamp=now + timedelta(minutes=minute))
        decision = reclaim_engine.record_utilization(gpu, observation)

    print(f"\nAfter {minutes} minutes at 1% utilization:")
    print(f"  GPU status: {gpu.status.value}")
    print(f"  Decision: {decision.action.value} ({decision.tier.value}) - {decision.reason}")
    print(f"  Event: [{decision.event.event_type.value}] {decision.event.message}")

    timeout_at = now + timedelta(minutes=minutes) + grace + timedelta(minutes=1)
    reclaimed = reclaim_engine.check_timeouts(now=timeout_at)

    print(f"\nNo response for {grace} + 1 minute:")
    for d in reclaimed:
        print(f"  {d.action.value}: {d.event.message} ({d.reason})")
    print(f"  GPU status: {gpu.status.value}")
    print(f"  Job status: {job.status.value}")
    print(f"  GPUs available for allocation again: {allocation_engine.available_gpu_count()}")


def demo_load_balancing() -> None:
    """Phase 5 demo: a new job arrives while one GPU is genuinely
    available and another merely *looks* idle (but is already
    legitimately assigned) - the brief's critical acceptance scenario.
    """
    print("\n" + "=" * 60)
    print("Phase 5 - Load Balancing & Task Routing")
    print("=" * 60)

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

    allocation_engine = AllocationEngine(state)
    for gpu in (gpu1, gpu2, gpu3, gpu4):
        allocation_engine.add_gpu(gpu)
    router = LoadBalancingRouter(state)

    new_job = Job(job_id="JOB-42", user_id="A", name="new work", priority=Priority.HIGH,
                  estimated_size_minutes=30, submitted_at=datetime.now(timezone.utc))
    allocation_engine.submit_job(new_job)

    print("\nPool: GPU-1=90% (assigned/A)  GPU-2=5% (assigned/B)  "
          "GPU-3=75% (assigned/C)  GPU-4=20% (IDLE/available)")

    selection = allocation_engine.select_next_job()
    job, policy, alloc_reason, candidates = selection
    routing = router.route_job(job, candidate_gpus=allocation_engine.gpu_pool.all_gpus())

    print(f"\nPhase 3 (which job?): {job.job_id} via {policy.value} - {alloc_reason}")
    print("Phase 5 (which GPU?) candidates:")
    for c in routing.candidates:
        tag = "available" if c.available else "assigned/unavailable"
        print(f"  {c.gpu_id:6s} util={c.utilization_percent:5.1f}%  {tag}")
    print(f"Selected: {routing.selected_gpu_id}  Reason: {routing.reason}")

    gpu = state.get_gpu(routing.selected_gpu_id)
    decision = allocation_engine.finalize_assignment(job, gpu, policy, alloc_reason, candidates)

    print(f"\nCommitted: {decision.job_id} -> {decision.gpu_id} (status={gpu.status.value})")
    print(f"Bob's GPU-2 untouched: assigned_user={gpu2.assigned_user_id}  "
          f"status={gpu2.status.value}  utilization={gpu2.utilization_percent:.1f}%")


def demo_simulation() -> None:
    """Phase 6 demo: run the built-in `full_lifecycle` scenario to
    completion through the deterministic `Simulator`, printing the
    resulting event log - the same 15-step demonstration as
    `demo_load_balancing`/`demo_reclamation` above, but now driven by
    a scenario's timeline instead of hand-written calls.
    """
    print("\n" + "=" * 60)
    print("Phase 6 - Scenario Simulator")
    print("=" * 60)

    registry = load_default_registry()
    print("\nAvailable scenarios:")
    for info in registry.list_scenarios():
        print(f"  {info.scenario_id:15s} {info.name}")

    from engine.simulation.simulator import Simulator

    scenario = registry.load("full_lifecycle")
    simulator = Simulator(scenario)
    print(f"\nRunning {scenario.scenario_id!r} - simulated start: {simulator.clock.now()}")

    simulator.run_to_completion()

    print(f"Simulated end: {simulator.clock.now()}  (real time elapsed: negligible - no sleeping)")
    print("\nEvent log:")
    for event in simulator.snapshot().events:
        print(f"  [{event.timestamp:%H:%M}] {event.event_type.value:8s} {event.message}")

    print("\nFinal job states:")
    for job in simulator.snapshot().jobs.values():
        print(f"  {job.job_id:6s} {job.status.value:10s} gpu={job.assigned_gpu_id}")


if __name__ == "__main__":
    main()
