from abc import ABC, abstractmethod


class ReplacementPolicy(ABC):
    """
    Abstract interface for cache replacement policies.
    """

    @abstractmethod
    def touch(self, key: int):
        """Called on access to a key (read or insert)."""
        pass

    @abstractmethod
    def prefetch_hit(self, key: int):
        """Called on a successful prefetch"""
        pass

    @abstractmethod
    def evict(self) -> int:
        """Return the key to evict."""
        pass

    @abstractmethod
    def insert(self, key: int):
        """Called when inserting a new key."""
        pass

    @abstractmethod
    def remove(self, key: int):
        """Called when a key is removed from cache."""
        pass
