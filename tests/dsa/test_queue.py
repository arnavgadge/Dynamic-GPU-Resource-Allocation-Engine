from engine.dsa.queue import Queue
from engine.dsa.waiting_job_queue import WaitingJobQueue
from engine.models.enums import Priority
from engine.models.job import Job


def test_empty_queue():
    q = Queue()
    assert q.is_empty() is True
    assert len(q) == 0
    assert q.peek() is None
    assert q.dequeue() is None


def test_enqueue_and_front():
    q = Queue()
    q.enqueue("a")
    q.enqueue("b")
    assert q.peek() == "a"
    assert len(q) == 2


def test_fifo_ordering_with_multiple_items():
    q = Queue()
    for value in ("A", "B", "C"):
        q.enqueue(value)

    assert q.dequeue() == "A"
    assert q.dequeue() == "B"
    assert q.dequeue() == "C"
    assert q.is_empty() is True


def test_dequeue_on_empty_queue_returns_none():
    q = Queue()
    assert q.dequeue() is None


# -- WaitingJobQueue (domain wrapper) -----------------------------------

def make_job(job_id: str) -> Job:
    return Job(job_id=job_id, user_id="U1", name="Job", priority=Priority.LOW,
               estimated_size_minutes=10)


def test_waiting_job_queue_preserves_arrival_order():
    queue = WaitingJobQueue()
    job_a, job_b, job_c = make_job("J1"), make_job("J2"), make_job("J3")

    queue.enqueue_job(job_a)
    queue.enqueue_job(job_b)
    queue.enqueue_job(job_c)

    assert queue.size() == 3
    assert queue.peek_job() is job_a
    assert queue.dequeue_job() is job_a
    assert queue.dequeue_job() is job_b
    assert queue.dequeue_job() is job_c
    assert queue.is_empty() is True
