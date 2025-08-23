from collections import OrderedDict

from prefetchlenz.cache.replacementpolicy.ReplacementPolicy import ReplacementPolicy


class LruReplacementPolicy(ReplacementPolicy):
    def __init__(self):
        self.order = OrderedDict()

    def touch(self, key: int):
        if key in self.order:
            self.order.move_to_end(key)

    def evict(self) -> int:
        return next(iter(self.order))

    def insert(self, key: int):
        self.order[key] = None
        self.touch(key)

    def remove(self, key: int):
        self.order.pop(key, None)

    def prefetch_hit(self, key: int):
        pass
