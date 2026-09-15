"""A generic LIFO stack.

Backed by a Python list used strictly as a stack (append/pop from the
end only, never indexed or popped from the front) - that access
pattern is exactly what a dynamic array is good at, giving true O(1)
push/pop, so no extra node-linking machinery is needed here the way
it was for `Queue` (which must operate on both ends).
"""

from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class Stack(Generic[T]):
    """LIFO stack: the last item pushed is the first item popped.

    Complexity
    ----------
    push      O(1)
    pop       O(1)
    peek        O(1)
    size / is_empty     O(1)
    """

    def __init__(self) -> None:
        self._items: List[T] = []

    def __len__(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items

    def push(self, value: T) -> None:
        """Add ``value`` on top of the stack. O(1)."""
        self._items.append(value)

    def pop(self) -> Optional[T]:
        """Remove and return the top item, or None if empty. O(1)."""
        if not self._items:
            return None
        return self._items.pop()

    def peek(self) -> Optional[T]:
        """The top item, without removing it, or None if empty. O(1)."""
        return self._items[-1] if self._items else None

    def to_list(self) -> List[T]:
        """Contents from bottom to top, without consuming the stack."""
        return list(self._items)
