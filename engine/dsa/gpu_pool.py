"""GPUPool: the company's shared GPU pool, backed by a linked list.

The report specifically calls for a linked list here because the
pool is dynamic - GPUs can be added or removed from the company's
inventory at any time, and a linked list gives O(1) insertion without
the shifting a fixed-size array would need. This wrapper gives that
generic `LinkedList` project meaning: it only ever holds `GPU`
objects and speaks in terms of GPU ids.

This is a pool abstraction, not a scheduler: it can tell you what
GPUs exist and hand you one by id, but it never decides which GPU a
job should get.
"""

from typing import Iterator, List, Optional

from engine.dsa.linked_list import LinkedList
from engine.models.gpu import GPU


class GPUPool:
    """The company's GPU inventory as a dynamic linked list of `GPU`."""

    def __init__(self) -> None:
        self._gpus: LinkedList[GPU] = LinkedList()

    def add_gpu(self, gpu: GPU) -> None:
        """Add a GPU to the pool. O(1)."""
        self._gpus.append(gpu)

    def remove_gpu(self, gpu_id: str) -> bool:
        """Remove a GPU from the pool by id. Returns True if found. O(n)."""
        return self._gpus.remove(lambda gpu: gpu.gpu_id == gpu_id)

    def get_gpu(self, gpu_id: str) -> Optional[GPU]:
        """Find a GPU by id. O(n)."""
        return self._gpus.find(lambda gpu: gpu.gpu_id == gpu_id)

    def all_gpus(self) -> List[GPU]:
        """All GPUs currently in the pool, in insertion order. O(n)."""
        return self._gpus.to_list()

    def size(self) -> int:
        return len(self._gpus)

    def is_empty(self) -> bool:
        return self._gpus.is_empty()

    def __iter__(self) -> Iterator[GPU]:
        return iter(self._gpus)

    def __len__(self) -> int:
        return len(self._gpus)
