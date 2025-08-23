from abc import ABC, abstractmethod

from prefetchlenz.cache.replacementpolicy.ReplacementPolicy import ReplacementPolicy


class SingleReplacementPolicy(ReplacementPolicy):
    key: int | None

    def touch(self, key: int):
        pass

    def prefetch_hit(self, key: int):
        pass

    def evict(self) -> int:
        if self.key is None:
            raise RuntimeError("key should not be None")
        return self.key

    def insert(self, key: int):
        self.key = key

    def remove(self, key: int):
        self.key = None
