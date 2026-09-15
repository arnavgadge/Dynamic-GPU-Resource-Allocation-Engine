"""A generic priority queue, built on top of `MaxHeap`.

This is the "central data structure" the project's design calls for:
waiting jobs go in, a caller-supplied comparator (``key_fn``) decides
priority, and the best candidate comes out first. Phase 2 only
builds the mechanism - the actual comparator (the
0.6 x priority + 0.4 x job-size-inverse allocation score) is supplied
by the scheduler in Phase 3. This class never hardcodes that formula
or any other priority rule; it just orders whatever `key_fn` tells it
to.
"""

import itertools
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar

from engine.dsa.max_heap import MaxHeap

T = TypeVar("T")


class PriorityQueue(Generic[T]):
    """Priority queue with a custom comparator, highest priority first.

    Internally a `MaxHeap` keyed on ``(key_fn(item), -insertion_seq)``.
    The second component is a tie-breaker: among items with equal
    priority it favours whichever was inserted first, so equal-priority
    items still come out in a predictable (FIFO) order instead of an
    arbitrary one.

    Complexity
    ----------
    insert     O(log n)  - one MaxHeap insert
    pop_best     O(log n)  - one MaxHeap extract_max
    peek_best      O(1)      - MaxHeap peek_max
    size / is_empty  O(1)
    """

    def __init__(self, key_fn: Callable[[T], Any]) -> None:
        self._key_fn = key_fn
        self._counter = itertools.count()
        self._heap: MaxHeap[Tuple[Any, int, T]] = MaxHeap(
            key_fn=lambda entry: (entry[0], entry[1])
        )

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return self._heap.is_empty()

    def insert(self, item: T) -> None:
        """Add an item, priority taken from ``key_fn(item)``. O(log n)."""
        seq = next(self._counter)
        # Negate the sequence number so that, for equal keys, the
        # entry with the *smaller* sequence number (inserted earlier)
        # compares as larger and is therefore preferred by the max-heap.
        self._heap.insert((self._key_fn(item), -seq, item))

    def peek_best(self) -> Optional[T]:
        """The current highest-priority item, without removing it. O(1)."""
        entry = self._heap.peek_max()
        return entry[2] if entry is not None else None

    def pop_best(self) -> Optional[T]:
        """Remove and return the current highest-priority item. O(log n)."""
        entry = self._heap.extract_max()
        return entry[2] if entry is not None else None
