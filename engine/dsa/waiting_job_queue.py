"""WaitingJobQueue: FCFS ordering for waiting jobs.

Wraps the generic `Queue` so that when the scheduler (in a later
phase) decides two jobs are close enough in size to use "first come,
first served" instead of shortest-job-first, it has a structure that
already preserves arrival order to pull from. This module does not
implement the 20% similarity rule, or any other decision about *when*
FCFS should be used - it only preserves order.
"""

from typing import List, Optional

from engine.dsa.queue import Queue
from engine.models.job import Job


class WaitingJobQueue:
    """FIFO queue of `Job` objects, oldest arrival first."""

    def __init__(self) -> None:
        self._queue: Queue[Job] = Queue()

    def enqueue_job(self, job: Job) -> None:
        """Add a job to the back of the line. O(1)."""
        self._queue.enqueue(job)

    def dequeue_job(self) -> Optional[Job]:
        """Remove and return the longest-waiting job. O(1)."""
        return self._queue.dequeue()

    def peek_job(self) -> Optional[Job]:
        """The longest-waiting job, without removing it. O(1)."""
        return self._queue.peek()

    def to_list(self) -> List[Job]:
        return self._queue.to_list()

    def size(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return self._queue.is_empty()
