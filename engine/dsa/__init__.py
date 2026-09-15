"""Phase 2: the DSA foundation the scheduler engine will be built on.

Two kinds of things live here:

- Generic, reusable structures with no project knowledge at all:
  `LinkedList`, `MinHeap`, `MaxHeap`, `PriorityQueue`, `HashMap`,
  `Queue`, `Stack`.
- Domain-specific wrappers that give one of those structures a fixed
  project meaning: `GPUPool` (linked list of `GPU`),
  `GPUUtilizationHeap` (min-heap keyed on utilization),
  `UserGPUIndex` (hashmap of user id -> GPU ids),
  `WaitingJobQueue` (FIFO of `Job`), `ReclaimHistory` (stack of
  reclaim `Event`s), `UtilizationSlidingWindow` (recent-history view
  over `UtilizationObservation`).

Nothing in this package makes a scheduling decision - see each
module's docstring for exactly what it does and does not do.
"""

from engine.dsa.linked_list import LinkedList
from engine.dsa.gpu_pool import GPUPool
from engine.dsa.min_heap import MinHeap
from engine.dsa.max_heap import MaxHeap
from engine.dsa.gpu_utilization_heap import GPUUtilizationHeap
from engine.dsa.priority_queue import PriorityQueue
from engine.dsa.hashmap import HashMap
from engine.dsa.user_gpu_index import UserGPUIndex
from engine.dsa.queue import Queue
from engine.dsa.waiting_job_queue import WaitingJobQueue
from engine.dsa.stack import Stack
from engine.dsa.reclaim_history import ReclaimHistory
from engine.dsa.sliding_window import UtilizationSlidingWindow

__all__ = [
    "LinkedList",
    "GPUPool",
    "MinHeap",
    "MaxHeap",
    "GPUUtilizationHeap",
    "PriorityQueue",
    "HashMap",
    "UserGPUIndex",
    "Queue",
    "WaitingJobQueue",
    "Stack",
    "ReclaimHistory",
    "UtilizationSlidingWindow",
]
