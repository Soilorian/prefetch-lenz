class Cache:
    """
    N-way set-associative cache with pluggable replacement policy.
    """

    def __init__(
        self, num_sets: int, num_ways: int, replacement_policy_cls, *args, **kwargs
    ):
        self.num_sets = num_sets
        self.num_ways = num_ways
        self.sets = [{} for _ in range(num_sets)]
        self.policies = [
            replacement_policy_cls(*args, **kwargs) for _ in range(num_sets)
        ]

    def _index(self, key: int) -> int:
        return key % self.num_sets

    def get(self, key: int):
        idx = self._index(key)
        set_ = self.sets[idx]
        if key in set_:
            self.policies[idx].touch(key)
            return set_[key]
        return None

    def put(self, key: int, value):
        idx = self._index(key)
        set_ = self.sets[idx]
        policy = self.policies[idx]

        if key in set_:
            set_[key] = value
            policy.touch(key)
        else:
            if len(set_) >= self.num_ways:
                victim = policy.evict()
                del set_[victim]
                policy.remove(victim)
            set_[key] = value
            policy.insert(key)

    def remove(self, key: int):
        idx = self._index(key)
        set_ = self.sets[idx]
        policy = self.policies[idx]
        if key in set_:
            del set_[key]
            policy.remove(key)

    def change_num_ways(self, num_ways: int):
        # todo update num ways, if increased, no need for further work
        # todo if decreased, we need to loop through the sets and evict from them until they reach the size
        pass

    def flush(self):
        # todo clears the cache
        pass

    def __contains__(self, key: int) -> bool:
        return self.get(key) is not None

    def __len__(self):
        return sum(len(s) for s in self.sets)
