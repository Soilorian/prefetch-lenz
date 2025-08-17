from dataclasses import dataclass

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.lfu import LfuReplacementPolicy
from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm


@dataclass
class FrequencyEntry:
    address: int
    frequency: int


class HistoryUnit:
    order: int

    def __init__(self, order: int = 1):
        self.pc_histories = {}  # pc → list of last addresses
        self.order = order

    def access(self, access: MemoryAccess) -> tuple[list[int], int] | None:
        pc = access.pc
        addr = access.address
        hist = self.pc_histories.setdefault(pc, [])

        hist.append(addr)
        if len(hist) > self.order:
            # keep only last "order" addresses
            hist = hist[-self.order :]
            self.pc_histories[pc] = hist

            return hist[:-1], hist[-1]  # (previous addresses, current)
        return None

    def clear(self):
        self.pc_histories.clear()


@dataclass
class MarkovEntry:
    key: list[int]
    frequencies: Cache

    def __init__(
        self,
        key: list[int],
        num_cache_entry: int = 4,
        selection_tolerance: float = 0.05,
    ):
        self.key = key
        self.selection_tolerance = selection_tolerance
        self.frequencies = Cache(
            num_sets=1,
            num_ways=num_cache_entry,
            replacement_policy_cls=LfuReplacementPolicy,
        )

    def update(self, address: int):
        entry = self.frequencies.get(address)
        if entry is not None:
            # just increment frequency
            entry
        else:
            # insert with initial frequency = 1
            self.frequencies.put(address, 1)

    def predict(self) -> list[int]:
        # collect all candidates and sort by frequency
        candidates = list(self.frequencies.sets[0].items())
        if not candidates:
            return []

        candidates.sort(key=lambda kv: kv[1], reverse=True)
        top_freq = candidates[0][1]
        result = [
            addr
            for addr, freq in candidates
            if freq >= (1 - self.selection_tolerance) * top_freq
        ]
        return result


class MarkovHistoryTable:
    order: int
    cache: Cache

    def __init__(self, order: int = 1, num_entries: int = 1024):
        self.order = order
        self.cache = Cache(
            num_sets=1,
            num_ways=num_entries,
            replacement_policy_cls=LfuReplacementPolicy,
        )

    def clear(self):
        self.cache.flush()

    def get_entry(self, key: tuple[int]) -> MarkovEntry | None:
        return self.cache.get(hash(key))

    def put_entry(self, key: tuple[int], entry: MarkovEntry):
        self.cache.put(hash(key), entry)


class MarkovPredictor(PrefetchAlgorithm):
    def __init__(self):
        self.first_order_table = MarkovHistoryTable(order=1)
        self.second_order_table = MarkovHistoryTable(order=2)
        self.first_order_history = HistoryUnit(order=1)
        self.second_order_history = HistoryUnit(order=2)

    def init(self):
        self.first_order_table.clear()
        self.second_order_table.clear()

    def progress(self, access: MemoryAccess, prefetch_hit: bool):
        prefetches = []

        # 1) Try second-order prediction
        second_hist = self.second_order_history.access(access)
        if second_hist:
            prev, curr = second_hist
            key = tuple(prev)
            entry = self.second_order_table.get_entry(key)
            if entry:
                preds = entry.predict()
                prefetches.extend(preds)

        # fallback: 1st-order
        if not prefetches:
            first_hist = self.first_order_history.access(access)
            if first_hist:
                prev, curr = first_hist
                key = tuple(prev)
                entry = self.first_order_table.get_entry(key)
                if entry:
                    preds = entry.predict()
                    prefetches.extend(preds)

        # 2) Update Markov tables with the new access
        if second_hist:
            prev, curr = second_hist
            key = tuple(prev)
            entry = self.second_order_table.get_entry(key)
            if not entry:
                entry = MarkovEntry(list(prev))
                self.second_order_table.put_entry(key, entry)
            entry.update(curr)

        if first_hist:
            prev, curr = first_hist
            key = tuple(prev)
            entry = self.first_order_table.get_entry(key)
            if not entry:
                entry = MarkovEntry(list(prev))
                self.first_order_table.put_entry(key, entry)
            entry.update(curr)

        return prefetches

    def close(self):
        self.first_order_table.clear()
        self.second_order_table.clear()
