"""A generic binary min-heap, implemented on a plain array (list).

This is the reusable DSA component. `engine.dsa.gpu_utilization_heap`
is the domain-specific wrapper that gives it project meaning (finding
the least-utilized GPU). Nothing in this file knows what a GPU is.
"""

from typing import Callable, Generic, List, Optional, TypeVar

T = TypeVar("T")
K = TypeVar("K")


class MinHeap(Generic[T]):
    """Binary min-heap ordered by ``key_fn(item)``.

    Standard array-backed binary heap: for a node at index ``i`` its
    children live at ``2i+1`` and ``2i+2``, and its parent at
    ``(i-1)//2``. Smaller keys sit closer to the root.

    Complexity
    ----------
    insert            O(log n)  - one sift-up from the new leaf to its place
    extract_min       O(log n)  - swap root with last leaf, then sift down
    peek_min           O(1)      - the minimum is always at index 0
    rebuild             O(n)      - bottom-up heapify after external mutation
    size / is_empty       O(1)
    """

    def __init__(self, key_fn: Callable[[T], K]) -> None:
        self._key_fn = key_fn
        self._items: List[T] = []

    def __len__(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items

    def insert(self, item: T) -> None:
        """Add an item and restore the heap property. O(log n)."""
        self._items.append(item)
        self._sift_up(len(self._items) - 1)

    def peek_min(self) -> Optional[T]:
        """Return (without removing) the minimum item, or None if empty. O(1)."""
        return self._items[0] if self._items else None

    def extract_min(self) -> Optional[T]:
        """Remove and return the minimum item, or None if empty. O(log n)."""
        if not self._items:
            return None
        minimum = self._items[0]
        last = self._items.pop()
        if self._items:
            self._items[0] = last
            self._sift_down(0)
        return minimum

    def rebuild(self) -> None:
        """Re-establish the heap property from scratch.

        Needed when an already-inserted item's key changes out from
        under the heap (e.g. a GPU's utilization is updated via
        `GPU.record_observation` after it was already inserted) - the
        heap holds a reference to that item, but its position was
        only ever correct for the key it had *at insertion time*.
        Uses the standard bottom-up heapify, which is O(n) - not
        O(n log n) - because most sift-downs near the bottom of the
        tree terminate almost immediately.
        """
        for index in range(len(self._items) // 2 - 1, -1, -1):
            self._sift_down(index)

    def to_list(self) -> List[T]:
        """The heap's items in heap (not sorted) order."""
        return list(self._items)

    # -- internals ----------------------------------------------------

    def _key(self, index: int) -> K:
        return self._key_fn(self._items[index])

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if self._key(index) < self._key(parent):
                self._items[index], self._items[parent] = self._items[parent], self._items[index]
                index = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        size = len(self._items)
        while True:
            left, right = 2 * index + 1, 2 * index + 2
            smallest = index
            if left < size and self._key(left) < self._key(smallest):
                smallest = left
            if right < size and self._key(right) < self._key(smallest):
                smallest = right
            if smallest == index:
                break
            self._items[index], self._items[smallest] = self._items[smallest], self._items[index]
            index = smallest
