"""GPUUtilizationHeap: find the least-utilized GPU in the pool.

Wraps the generic `MinHeap` with a fixed key of "current utilization
percentage" - that is simply reading a value the `GPU` model already
exposes, not a scheduling rule. This structure only answers "which
GPU is least busy right now?". It does NOT decide whether that GPU
should be reclaimed - that decision belongs to the reclamation engine
built in a later phase.
"""

from typing import Optional

from engine.dsa.min_heap import MinHeap
from engine.models.gpu import GPU


class GPUUtilizationHeap:
    """Min-heap of `GPU` objects ordered by `GPU.utilization_percent`."""

    def __init__(self) -> None:
        self._heap: MinHeap[GPU] = MinHeap(key_fn=lambda gpu: gpu.utilization_percent)

    def insert_gpu(self, gpu: GPU) -> None:
        """Add a GPU, ordered by its utilization at insertion time. O(log n)."""
        self._heap.insert(gpu)

    def peek_least_utilized(self) -> Optional[GPU]:
        """The currently least-utilized GPU, without removing it. O(1)."""
        return self._heap.peek_min()

    def extract_least_utilized(self) -> Optional[GPU]:
        """Remove and return the currently least-utilized GPU. O(log n)."""
        return self._heap.extract_min()

    def refresh(self) -> None:
        """Re-sort after GPUs already in the heap changed utilization.

        `GPU.record_observation` mutates a GPU's ``utilization_percent``
        in place; since the heap holds references, not copies, that
        change is invisible to the heap until it is told to
        re-establish its ordering. O(n).
        """
        self._heap.rebuild()

    def size(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return self._heap.is_empty()
