"""A generic binary max-heap, implemented on a plain array (list).

Mirror image of `engine.dsa.min_heap.MinHeap` - same array-backed
binary heap technique, with the comparison flipped so the largest key
sits at the root. `engine.dsa.priority_queue.PriorityQueue` builds on
this to give waiting jobs a comparator-driven ordering; this file
itself does not know what a "job" or a "priority" is.
"""

from typing import Callable, Generic, List, Optional, TypeVar

T = TypeVar("T")
K = TypeVar("K")


class MaxHeap(Generic[T]):
    """Binary max-heap ordered by ``key_fn(item)``.

    Complexity
    ----------
    insert            O(log n)
    extract_max       O(log n)
    peek_max           O(1)
    rebuild             O(n)      - see `MinHeap.rebuild` for why this is needed
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

    def peek_max(self) -> Optional[T]:
        """Return (without removing) the maximum item, or None if empty. O(1)."""
        return self._items[0] if self._items else None

    def extract_max(self) -> Optional[T]:
        """Remove and return the maximum item, or None if empty. O(log n)."""
        if not self._items:
            return None
        maximum = self._items[0]
        last = self._items.pop()
        if self._items:
            self._items[0] = last
            self._sift_down(0)
        return maximum

    def rebuild(self) -> None:
        """Re-establish the heap property from scratch. O(n).

        Same rationale as `MinHeap.rebuild`: call this if an item
        already in the heap has its key changed in place.
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
            if self._key(index) > self._key(parent):
                self._items[index], self._items[parent] = self._items[parent], self._items[index]
                index = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        size = len(self._items)
        while True:
            left, right = 2 * index + 1, 2 * index + 2
            largest = index
            if left < size and self._key(left) > self._key(largest):
                largest = left
            if right < size and self._key(right) > self._key(largest):
                largest = right
            if largest == index:
                break
            self._items[index], self._items[largest] = self._items[largest], self._items[index]
            index = largest
