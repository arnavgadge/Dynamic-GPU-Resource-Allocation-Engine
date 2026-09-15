"""A real singly linked list - not a wrapper around Python's ``list``.

This is the generic structure. The GPU pool (`engine.dsa.gpu_pool`) is
the domain-specific wrapper that gives it project meaning.
"""

from typing import Callable, Generic, Iterator, Optional, TypeVar

T = TypeVar("T")


class _Node(Generic[T]):
    __slots__ = ("value", "next")

    def __init__(self, value: T):
        self.value = value
        self.next: Optional["_Node[T]"] = None


class LinkedList(Generic[T]):
    """Singly linked list with O(1) append/prepend and O(n) search/removal.

    A head and tail pointer are kept explicitly (rather than only a
    head pointer) so that appending - the common case for a pool that
    mostly grows - does not require walking the whole list first.

    Complexity
    ----------
    append(value)            O(1)   - tail pointer means no traversal
    prepend(value)            O(1)
    find(predicate)            O(n)   - must scan until a match is found
    remove(predicate)           O(n)   - same reason as find
    traversal (``for x in ll``)  O(n) total
    size                          O(1)   - maintained by a counter, not counted on demand
    """

    def __init__(self) -> None:
        self._head: Optional[_Node[T]] = None
        self._tail: Optional[_Node[T]] = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def append(self, value: T) -> None:
        """Insert ``value`` at the end of the list. O(1)."""
        node = _Node(value)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1

    def prepend(self, value: T) -> None:
        """Insert ``value`` at the front of the list. O(1)."""
        node = _Node(value)
        node.next = self._head
        self._head = node
        if self._tail is None:
            self._tail = node
        self._size += 1

    def find(self, predicate: Callable[[T], bool]) -> Optional[T]:
        """Return the first value for which ``predicate(value)`` is true. O(n)."""
        for value in self:
            if predicate(value):
                return value
        return None

    def remove(self, predicate: Callable[[T], bool]) -> bool:
        """Remove the first value for which ``predicate(value)`` is true.

        Returns True if something was removed, False otherwise. O(n).
        """
        previous: Optional[_Node[T]] = None
        current = self._head
        while current is not None:
            if predicate(current.value):
                if previous is None:
                    self._head = current.next
                else:
                    previous.next = current.next
                if current is self._tail:
                    self._tail = previous
                self._size -= 1
                return True
            previous, current = current, current.next
        return False

    def to_list(self) -> list:
        return list(self)

    def __iter__(self) -> Iterator[T]:
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def __repr__(self) -> str:
        return f"LinkedList({self.to_list()!r})"
