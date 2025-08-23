from prefetchlenz.cache.replacementpolicy.ReplacementPolicy import ReplacementPolicy


class HawkeyeReplacementPolicy(ReplacementPolicy):
    def __init__(self):
        """
        score_func: callable(key) → int (e.g., metadata.score)
        """
        self.scores = {}

    def touch(self, key: int):
        pass

    def prefetch_hit(self, key: int):
        self.scores[key] += 1

    def evict(self) -> int:
        return min(self.scores, key=lambda x: self.scores[x])

    def insert(self, key: int):
        self.scores[key] = 0

    def remove(self, key: int):
        del self.scores[key]
