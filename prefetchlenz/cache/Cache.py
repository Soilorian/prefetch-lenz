class Cache:
    """
    N-way set-associative cache with pluggable replacement policy.
    """

    def __init__(
        self, num_sets: int, num_ways: int, replacement_policy_cls, *args, **kwargs
    ):
        self.num_sets = num_sets
        self.num_ways = num_ways
        self.replacement_policy_cls = replacement_policy_cls
        self.policy_args = args
        self.policy_kwargs = kwargs
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
        """
        Update the number of ways per set. If increased, no eviction is needed.
        If decreased, evict from each set until it meets the new limit.
        """
        self.num_ways = num_ways
        for idx in range(self.num_sets):
            set_ = self.sets[idx]
            policy = self.policies[idx]

            while len(set_) > num_ways:
                victim = policy.evict()
                if victim in set_:
                    del set_[victim]
                policy.remove(victim)

    def flush(self):
        """
        Clears the entire cache and resets the replacement policies.
        """
        self.sets = [{} for _ in range(self.num_sets)]
        self.policies = [
            self.replacement_policy_cls(*self.policy_args, **self.policy_kwargs)
            for _ in range(self.num_sets)
        ]

    def prefetch_hit(self, key: int):
        idx = self._index(key)
        policy = self.policies[idx]
        policy.prefetch_hit(key)

    def __contains__(self, key: int) -> bool:
        return self.get(key) is not None

    def __len__(self):
        return sum(len(s) for s in self.sets)
