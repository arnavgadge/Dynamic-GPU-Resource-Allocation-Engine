"""Phase 2 integration demo: the DSA structures working together.

This does NOT make a scheduling decision anywhere - it only shows
that data flows correctly through each structure: GPUs into the
linked-list pool and the utilization min-heap, assignments into the
hashmap-backed index, jobs into the priority queue and FCFS queue,
a reclaim event onto the stack, and observations into the sliding
window.
"""

from datetime import datetime, timedelta, timezone

from engine.dsa.gpu_pool import GPUPool
from engine.dsa.gpu_utilization_heap import GPUUtilizationHeap
from engine.dsa.priority_queue import PriorityQueue
from engine.dsa.reclaim_history import ReclaimHistory
from engine.dsa.sliding_window import UtilizationSlidingWindow
from engine.dsa.user_gpu_index import UserGPUIndex
from engine.dsa.waiting_job_queue import WaitingJobQueue
from engine.models.assignment import GPUAssignment
from engine.models.enums import EventType, Priority
from engine.models.event import Event
from engine.models.gpu import GPU
from engine.models.job import Job
from engine.models.utilization import UtilizationObservation

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_gpu(gpu_id: str, utilization: float) -> GPU:
    gpu = GPU(gpu_id=gpu_id, total_memory_mb=24_576)
    gpu.record_observation(UtilizationObservation(utilization_percent=utilization, timestamp=NOW))
    return gpu


def test_gpu_pool_and_utilization_heap_agree_on_the_same_gpus():
    pool = GPUPool()
    heap = GPUUtilizationHeap()

    readings = {"GPU0": 90.0, "GPU1": 5.0, "GPU2": 70.0, "GPU3": 15.0}
    for gpu_id, util in readings.items():
        gpu = make_gpu(gpu_id, util)
        pool.add_gpu(gpu)
        heap.insert_gpu(gpu)

    # The pool (linked list) knows about every GPU that exists.
    assert pool.size() == 4
    assert {g.gpu_id for g in pool.all_gpus()} == set(readings)

    # The heap can immediately identify the least-utilized GPU.
    least = heap.peek_least_utilized()
    assert least.gpu_id == "GPU1"
    assert least.utilization_percent == 5.0


def test_hashmap_backed_index_tracks_user_to_gpu_assignments():
    index = UserGPUIndex()

    index.add_assignment(GPUAssignment(assignment_id="A1", gpu_id="GPU0", user_id="U1", job_id="J1"))
    index.add_assignment(GPUAssignment(assignment_id="A2", gpu_id="GPU1", user_id="U2", job_id="J2"))
    index.add_assignment(GPUAssignment(assignment_id="A3", gpu_id="GPU2", user_id="U2", job_id="J3"))

    assert index.get_gpus_for_user("U1") == ["GPU0"]
    assert sorted(index.get_gpus_for_user("U2")) == ["GPU1", "GPU2"]


def test_waiting_jobs_enter_the_priority_queue_ordered_by_priority():
    # key_fn stands in for "the allocation score" that Phase 3 will
    # define - here it is just the job's existing Priority value, so
    # the queue itself hardcodes nothing about scoring.
    pq = PriorityQueue(key_fn=lambda job: job.priority)

    low = Job(job_id="J1", user_id="U3", name="Data Processing",
              priority=Priority.LOW, estimated_size_minutes=45)
    high = Job(job_id="J2", user_id="U1", name="ML Training",
               priority=Priority.HIGH, estimated_size_minutes=180)
    medium = Job(job_id="J3", user_id="U2", name="Excel",
                 priority=Priority.MEDIUM, estimated_size_minutes=15)

    for job in (low, high, medium):
        pq.insert(job)

    assert pq.pop_best() is high
    assert pq.pop_best() is medium
    assert pq.pop_best() is low
    assert pq.is_empty() is True


def test_queue_preserves_fcfs_arrival_order():
    fcfs = WaitingJobQueue()
    job_a = Job(job_id="J1", user_id="U1", name="Job A", priority=Priority.LOW, estimated_size_minutes=10)
    job_b = Job(job_id="J2", user_id="U1", name="Job B", priority=Priority.LOW, estimated_size_minutes=10)
    job_c = Job(job_id="J3", user_id="U1", name="Job C", priority=Priority.LOW, estimated_size_minutes=10)

    for job in (job_a, job_b, job_c):
        fcfs.enqueue_job(job)

    assert fcfs.dequeue_job() is job_a
    assert fcfs.dequeue_job() is job_b
    assert fcfs.dequeue_job() is job_c


def test_stack_demonstrates_reclaim_history_ordering():
    history = ReclaimHistory()
    reclaim_gpu0 = Event(event_id="E1", event_type=EventType.RECLAIM, gpu_id="GPU0", message="Reclaimed GPU0")
    reclaim_gpu1 = Event(event_id="E2", event_type=EventType.RECLAIM, gpu_id="GPU1", message="Reclaimed GPU1")

    history.record_reclaim(reclaim_gpu0)
    history.record_reclaim(reclaim_gpu1)

    # Most recent reclaim comes back first, as a rollback would need.
    assert history.undo_last() is reclaim_gpu1
    assert history.undo_last() is reclaim_gpu0
    assert history.is_empty() is True


def test_sliding_window_accumulates_utilization_observations():
    window = UtilizationSlidingWindow(window_duration=timedelta(minutes=20))

    for minute, percent in [(0, 2.0), (5, 1.0), (10, 1.5), (15, 2.5)]:
        window.add_observation(
            UtilizationObservation(utilization_percent=percent, timestamp=NOW + timedelta(minutes=minute))
        )

    observed = window.observations()
    assert [o.utilization_percent for o in observed] == [2.0, 1.0, 1.5, 2.5]
    assert window.window_span() == timedelta(minutes=15)


def test_full_walkthrough_touches_every_structure_without_deciding_anything():
    """One end-to-end pass through every Phase 2 structure.

    Mirrors the report's running example: 4 GPUs, one clearly idle,
    a waiting job, a reclaim, and a stream of utilization readings.
    No structure here picks a winner or reclaims anything - they only
    store and retrieve data.
    """
    pool = GPUPool()
    heap = GPUUtilizationHeap()
    index = UserGPUIndex()
    waiting_jobs = WaitingJobQueue()
    reclaim_history = ReclaimHistory()
    window = UtilizationSlidingWindow(window_duration=timedelta(minutes=30))

    gpus = {gid: make_gpu(gid, util) for gid, util in
            [("GPU0", 90.0), ("GPU1", 5.0), ("GPU2", 70.0), ("GPU3", 15.0)]}
    for gpu in gpus.values():
        pool.add_gpu(gpu)
        heap.insert_gpu(gpu)

    assert pool.size() == 4
    assert heap.peek_least_utilized().gpu_id == "GPU1"

    index.add_assignment(GPUAssignment(assignment_id="A1", gpu_id="GPU0", user_id="U1", job_id="J1"))
    assert index.get_gpus_for_user("U1") == ["GPU0"]

    waiting_job = Job(job_id="J5", user_id="U3", name="Video Editing",
                       priority=Priority.LOW, estimated_size_minutes=60)
    waiting_jobs.enqueue_job(waiting_job)
    assert waiting_jobs.peek_job() is waiting_job

    reclaim_history.record_reclaim(
        Event(event_id="E1", event_type=EventType.RECLAIM, gpu_id="GPU1", message="Reclaimed GPU1")
    )
    assert reclaim_history.peek_last().gpu_id == "GPU1"

    for minute in range(3):
        window.add_observation(
            UtilizationObservation(utilization_percent=1.0, timestamp=NOW + timedelta(minutes=minute))
        )
    assert window.size() == 3
