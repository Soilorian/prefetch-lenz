from collections import deque

from prefetchlenz.cache.replacementpolicy.ReplacementPolicy import ReplacementPolicy


class FIFOReplacementPolicy(ReplacementPolicy):
    """
    FIFO (First-In-First-Out) replacement policy.
    Evicts the oldest inserted key regardless of access frequency.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = deque()  # maintains insertion order
        self.in_cache = set()  # quick membership lookup

    def touch(self, key: int):
        # FIFO ignores touches; order is defined only by insertions
        pass

    def prefetch_hit(self, key: int):
        # No effect for FIFO
        pass

    def evict(self) -> int:
        """Evict and return the oldest key."""
        if not self.queue:
            return None
        key = self.queue.popleft()
        self.in_cache.remove(key)
        return key

    def insert(self, key: int):
        """Insert a key if not already present. Evict the oldest if over capacity."""
        if key in self.in_cache:
            return None  # already present

        evicted = None
        if len(self.queue) >= self.capacity:
            evicted = self.evict()

        self.queue.append(key)
        self.in_cache.add(key)
        return evicted

    def remove(self, key: int):
        """Explicitly remove a key if present."""
        if key in self.in_cache:
            self.in_cache.remove(key)
            try:
                self.queue.remove(key)
            except ValueError:
                pass
