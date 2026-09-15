"""ReclaimHistory: an undo-able log of reclaim actions.

Wraps the generic `Stack` with `Event` (Phase 1's model for "something
the engine did"), so the most recent reclaim is always what comes
back first if the future reclamation engine needs to roll one back.
This module does not decide when a reclaim should happen or what
"rolling back" actually does to a GPU/job - it only remembers order.
"""

from typing import Optional

from engine.dsa.stack import Stack
from engine.history_config import MAX_RECLAIM_HISTORY
from engine.models.event import Event


class ReclaimHistory:
    """Stack of reclaim `Event`s, most recent reclaim on top."""

    def __init__(self) -> None:
        self._stack: Stack[Event] = Stack()

    def record_reclaim(self, event: Event) -> None:
        """Record a reclaim action as having happened. O(1) amortized -
        occasionally O(n) when the retention cap (`MAX_RECLAIM_HISTORY`,
        Phase 15) is exceeded and the oldest entries are evicted; still
        never touches active scheduler state, and `undo_last`/
        `peek_last` (which only ever look at the most recent end) are
        unaffected either way.
        """
        self._stack.push(event)
        if len(self._stack) > MAX_RECLAIM_HISTORY:
            trimmed = self._stack.to_list()[-MAX_RECLAIM_HISTORY:]
            self._stack = Stack()
            for item in trimmed:
                self._stack.push(item)

    def undo_last(self) -> Optional[Event]:
        """Remove and return the most recent reclaim, or None if empty. O(1)."""
        return self._stack.pop()

    def peek_last(self) -> Optional[Event]:
        """The most recent reclaim, without removing it. O(1)."""
        return self._stack.peek()

    def size(self) -> int:
        return len(self._stack)

    def is_empty(self) -> bool:
        return self._stack.is_empty()
