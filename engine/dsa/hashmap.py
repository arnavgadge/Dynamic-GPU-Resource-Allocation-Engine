"""A real HashMap - bucket array with chaining, not a wrapper around dict.

This is the generic component. `engine.dsa.user_gpu_index` is the
domain-specific wrapper that gives it project meaning (mapping a user
to the GPU(s) currently assigned to them).
"""

from typing import Any, Generic, Iterator, List, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")

_DEFAULT_CAPACITY = 8
_LOAD_FACTOR_THRESHOLD = 0.75


class HashMap(Generic[K, V]):
    """Hash map with separate-chaining collision handling.

    Storage is an array of ``capacity`` buckets; each bucket is a
    plain list of ``(key, value)`` pairs that hashed to it. ``hash(key)
    % capacity`` picks the bucket - collisions (different keys, same
    bucket) are resolved by scanning that bucket's short list. The
    table doubles and every entry is re-bucketed whenever the load
    factor (entries / capacity) would exceed 0.75, which is what keeps
    lookups close to O(1) on average as the map grows.

    Complexity (average case, assuming a reasonable hash spread)
    ----------
    put         O(1) amortized  - occasional O(n) resize is spread over many puts
    get           O(1)
    remove          O(1)
    contains          O(1)
    Worst case for all of the above is O(n) if every key collided into
    one bucket - unlikely in practice with Python's built-in `hash`.
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = max(capacity, 1)
        self._buckets: List[List[Tuple[K, V]]] = [[] for _ in range(self._capacity)]
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def _bucket_index(self, key: K) -> int:
        return hash(key) % self._capacity

    def put(self, key: K, value: V) -> None:
        """Insert or overwrite the value for ``key``. O(1) amortized."""
        bucket = self._buckets[self._bucket_index(key)]
        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self._size / self._capacity > _LOAD_FACTOR_THRESHOLD:
            self._resize(self._capacity * 2)

    # `update` is an alias kept for readability at call sites where
    # "update the mapping" reads better than "put".
    def update(self, key: K, value: V) -> None:
        self.put(key, value)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Look up ``key``, or return ``default`` if absent. O(1)."""
        bucket = self._buckets[self._bucket_index(key)]
        for existing_key, value in bucket:
            if existing_key == key:
                return value
        return default

    def contains(self, key: K) -> bool:
        bucket = self._buckets[self._bucket_index(key)]
        return any(existing_key == key for existing_key, _ in bucket)

    def remove(self, key: K) -> bool:
        """Remove ``key`` if present. Returns True if something was removed. O(1)."""
        bucket = self._buckets[self._bucket_index(key)]
        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                self._size -= 1
                return True
        return False

    def keys(self) -> List[K]:
        return [key for bucket in self._buckets for key, _ in bucket]

    def values(self) -> List[V]:
        return [value for bucket in self._buckets for _, value in bucket]

    def items(self) -> List[Tuple[K, V]]:
        return [pair for bucket in self._buckets for pair in bucket]

    def _resize(self, new_capacity: int) -> None:
        old_items = self.items()
        self._capacity = new_capacity
        self._buckets = [[] for _ in range(new_capacity)]
        self._size = 0
        for key, value in old_items:
            self.put(key, value)

    def __iter__(self) -> Iterator[K]:
        return iter(self.keys())

    def __contains__(self, key: Any) -> bool:
        return self.contains(key)
