import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Optional

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm
from prefetchlenz.util.size import Size

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.triangel")


@dataclass
class TriangelPrefetcherMetaData:
    neighbor: int
    confidence: int = 1
    score: int = 0


class TriangelPrefetcher(PrefetchAlgorithm):
    """
    Triangel Prefetcher with:
      - PC‐localized training
      - History and Second‐Chance Samplers
      - Metadata Reuse Buffer
      - Set‐Dueller partitioning
      - LRU eviction
    """

    class HistorySampler:
        def __init__(self, size: int):
            # circular buffer of (pc, addr, timestamp)
            self.buf = {}
            self.maxdist = 0

        def sample(self, pc: int, addr: int, ts: int):
            """
            Return (reuse_conf_inc: bool, pattern_conf_inc: bool)
            based on history entries.
            """
            entry = self.buf.get(addr)
            if entry and entry[0] == pc:
                dist = ts - entry[2]
                reuse = dist <= self.maxdist
                # pattern only if target matches
                pattern = entry[1] == addr
            else:
                reuse = pattern = False
            # always overwrite
            self.buf[addr] = (pc, addr, ts)
            return reuse, pattern

    class SecondChanceSampler:
        def __init__(self, size: int, window: int = 512):
            self.buf = {}
            self.window = window

        def sample(self, pc: int, tgt: int, ts: int):
            """
            Return True if a second‐chance hit occurred within window.
            """
            last = self.buf.get(tgt)
            hit = last and (ts - last) <= self.window and last[0] == pc
            # update for future
            self.buf[tgt] = (pc, ts)
            return hit

    class MetadataReuseBuffer:
        def __init__(self, size: int = 256):
            # simple FIFO 2‐way by set
            self.sets = defaultdict(lambda: deque(maxlen=2))
            self.size = size

        def probe(self, addr: int) -> bool:
            s = addr % self.size
            return addr in self.sets[s]

        def insert(self, addr: int):
            s = addr % self.size
            buf = self.sets[s]
            if addr not in buf:
                if len(buf) >= buf.maxlen:
                    buf.popleft()
                buf.append(addr)

    class SetDueller:
        def __init__(self, max_ways: int):
            self.max_ways = max_ways
            # counters for partition sizes 0..max_ways
            self.counters = [0] * (max_ways + 1)

        def record_data_hit(self, way: int):
            for w in range(0, way + 1):
                self.counters[w] += 1

        def record_meta_hit(self, way: int):
            for w in range(way, self.max_ways + 1):
                self.counters[w] += 1

        def best_partition(self) -> int:
            # choose way with highest hits
            return max(range(len(self.counters)), key=lambda w: self.counters[w])

    def __init__(
        self,
        init_size: Size = Size.from_kb(512),
        min_size: Size = Size.from_kb(128),
        max_size: Size = Size.from_mb(2),
        resize_epoch: int = 50_000,
        grow_thresh: float = 0.1,
        shrink_thresh: float = 0.05,
    ):
        # metadata partition size (in entries)
        self.size = init_size
        self.min_size = min_size
        self.max_size = max_size

        self.table: dict[int, TriangelPrefetcherMetaData] = {}
        self.last_access_per_pc: dict[int, int] = {}

        # samplers & reuse buffer & dueller
        self.history = self.HistorySampler(size=1024)
        self.second_chance = self.SecondChanceSampler(size=64)
        self.mrb = self.MetadataReuseBuffer(size=256)
        self.dueller = self.SetDueller(max_ways=8)

        # dynamic resize stats
        self.resize_epoch = resize_epoch
        self.grow_thresh = grow_thresh
        self.shrink_thresh = shrink_thresh
        self.meta_accesses = 0
        self.useful_prefetches = 0

        # per‐PC timestamps
        self.timestamps: dict[int, int] = defaultdict(int)

    def init(self, size: Optional[Size] = None):
        if size:
            self.size = size
        self.table.clear()
        self.last_access_per_pc.clear()
        self.meta_accesses = 0
        self.useful_prefetches = 0
        logger.info(f"Triangel init: capacity={int(self.size)} entries")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        pc, addr = access.pc, access.address
        preds: List[int] = []

        # increment PC timestamp
        ts = self.timestamps[pc] + 1
        self.timestamps[pc] = ts

        # 1) TRAIN: form (prev→addr)
        prev = self.last_access_per_pc.get(pc)
        if prev is not None:
            md = self.table.get(prev)
            # history sampling for reuse & pattern
            reuse_ok, pattern_ok = self.history.sample(pc, prev, ts)
            sc_ok = self.second_chance.sample(pc, prev, ts)

            # only store if reuse_ok
            if reuse_ok:
                if md:
                    # update neighbor confidence
                    if md.neighbor == addr:
                        md.confidence = 1
                    else:
                        md.confidence -= 1
                        if md.confidence <= 0:
                            md.neighbor = addr
                            md.confidence = 1
                else:
                    # evict if full
                    if len(self.table) >= int(self.size):
                        self._evict_entry()
                    self.table[prev] = TriangelPrefetcherMetaData(neighbor=addr)

                # score based on pattern or second-chance
                if pattern_ok or sc_ok:
                    self.table[prev].score += 1

        # 2) PREDICT: only if PC trained before
        if prev is not None and addr in self.table:
            entry = self.table[addr]
            tgt = entry.neighbor

            # MRB: avoid redundant L3 lookup
            if not self.mrb.probe(addr):
                self.mrb.insert(addr)
                # record a metadata hit at current partition ways
                part = self.dueller.best_partition()
                self.dueller.record_meta_hit(part)

            preds.append(tgt)
            # if miss→useful
            if not prefetch_hit:
                self.useful_prefetches += 1

        # 3) DYNAMIC RESIZE
        self.meta_accesses += 1
        if self.meta_accesses >= self.resize_epoch:
            ratio = self.useful_prefetches / self.resize_epoch
            old = int(self.size)
            if ratio > self.grow_thresh and int(self.size) < int(self.max_size):
                self.size = Size(min(int(self.max_size), int(self.size) * 3 // 2))
                logger.info(f"GROW: {old}→{int(self.size)} (ratio={ratio:.3f})")
            elif ratio < self.shrink_thresh and int(self.size) > int(self.min_size):
                self.size = Size(max(int(self.min_size), int(self.size) * 3 // 4))
                logger.info(f"SHRINK: {old}→{int(self.size)} (ratio={ratio:.3f})")
            self.meta_accesses = 0
            self.useful_prefetches = 0

        # 4) SET DUELLER: record data hit for this PC access
        part = self.dueller.best_partition()
        self.dueller.record_data_hit(part)

        # 5) UPDATE last_access
        self.last_access_per_pc[pc] = addr
        return preds

    def _evict_entry(self):
        # LRU by lowest score, then arbitrary
        victim = min(self.table.items(), key=lambda kv: kv[1].score)[0]
        md = self.table.pop(victim)
        logger.info(f"EVICT {hex(victim)}→{hex(md.neighbor)} (score={md.score})")

    def close(self):
        logger.info(f"Triangel closed: final entries={len(self.table)}")
        self.table.clear()
        self.last_access_per_pc.clear()
