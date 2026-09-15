"""A generic FIFO queue, implemented with head/tail node pointers.

Deliberately not backed by ``list.pop(0)`` - that shifts every
remaining element down and is O(n) per dequeue. Keeping explicit head
and tail pointers (the same technique as `engine.dsa.linked_list`)
gives true O(1) enqueue and dequeue.
"""

from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class _Node(Generic[T]):
    __slots__ = ("value", "next")

    def __init__(self, value: T):
        self.value = value
        self.next: Optional["_Node[T]"] = None


class Queue(Generic[T]):
    """FIFO queue: whatever is enqueued first is dequeued first.

    Complexity
    ----------
    enqueue      O(1)
    dequeue      O(1)
    peek/front       O(1)
    size / is_empty      O(1)
    """

    def __init__(self) -> None:
        self._head: Optional[_Node[T]] = None
        self._tail: Optional[_Node[T]] = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def enqueue(self, value: T) -> None:
        """Add ``value`` to the back of the queue. O(1)."""
        node = _Node(value)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1

    def dequeue(self) -> Optional[T]:
        """Remove and return the item at the front, or None if empty. O(1)."""
        if self._head is None:
            return None
        node = self._head
        self._head = node.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return node.value

    def peek(self) -> Optional[T]:
        """The item at the front, without removing it, or None if empty. O(1)."""
        return self._head.value if self._head is not None else None

    def to_list(self) -> list:
        """Contents from front to back, without consuming the queue."""
        result = []
        current = self._head
        while current is not None:
            result.append(current.value)
            current = current.next
        return result
