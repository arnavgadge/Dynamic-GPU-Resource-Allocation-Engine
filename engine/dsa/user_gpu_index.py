"""UserGPUIndex: fast user -> currently-assigned-GPU(s) lookup.

Wraps the generic `HashMap` with the project's actual shape of that
relationship: a user may hold zero, one, or many GPUs at once (there
is no "one user, one GPU" assumption anywhere here), so each user id
maps to a *list* of GPU ids rather than a single one. Built from
Phase 1's `GPUAssignment` records rather than inventing a new
assignment model.
"""

from typing import List

from engine.dsa.hashmap import HashMap
from engine.models.assignment import GPUAssignment


class UserGPUIndex:
    """HashMap-backed index of user id -> list of assigned GPU ids."""

    def __init__(self) -> None:
        self._map: HashMap[str, List[str]] = HashMap()

    def add_assignment(self, assignment: GPUAssignment) -> None:
        """Record that ``assignment.gpu_id`` is now held by its user. O(1)."""
        gpu_ids = self._map.get(assignment.user_id)
        if gpu_ids is None:
            gpu_ids = []
            self._map.put(assignment.user_id, gpu_ids)
        if assignment.gpu_id not in gpu_ids:
            gpu_ids.append(assignment.gpu_id)

    def remove_assignment(self, assignment: GPUAssignment) -> bool:
        """Undo `add_assignment` for one ended assignment. O(1)."""
        gpu_ids = self._map.get(assignment.user_id)
        if gpu_ids is None or assignment.gpu_id not in gpu_ids:
            return False
        gpu_ids.remove(assignment.gpu_id)
        if not gpu_ids:
            self._map.remove(assignment.user_id)
        return True

    def get_gpus_for_user(self, user_id: str) -> List[str]:
        """Which GPU ids does this user currently hold? O(1)."""
        return list(self._map.get(user_id, []) or [])

    def has_user(self, user_id: str) -> bool:
        return self._map.contains(user_id)

    def size(self) -> int:
        """Number of users with at least one GPU tracked. O(1)."""
        return len(self._map)
