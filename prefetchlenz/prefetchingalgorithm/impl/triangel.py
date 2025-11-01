"""
Triangel: A High-Performance, Accurate, Timely On-Chip Temporal Prefetcher by Ainsworth et al.
This variant consistently uses the project's Cache class for Markov metadata storage.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.hawkeye import HawkeyeReplacementPolicy
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm
from prefetchlenz.util.size import Size

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.triangel")
logger.addHandler(logging.NullHandler())


@dataclass
class TriangelMeta:
    neighbor: int
    conf: int = 0


@dataclass
class TrainEntry:
    pc_tag: int
    last0: Optional[int] = None
    last1: Optional[int] = None
    time: int = 0
    reuse_conf: int = 0
    patt_base: int = 8
    patt_high: int = 8
    sample_rate: int = 8
    lookahead2: bool = False


class HistorySampler:
    """
    Small set-associative sampler used to record observed (index -> target, ts).
    Supports lookup by (key, pc_tag).
    """

    def __init__(self, sets: int = 64, ways: int = 2):
        assert sets > 0 and ways > 0
        self.sets = sets
        self.ways = ways
        self.table: List[Deque[Tuple[int, int, int, int]]] = [
            deque(maxlen=ways) for _ in range(sets)
        ]
        # entry: (key, pc_tag, target, ts)

    def _set(self, key: int) -> int:
        # distribute by line bits but be robust for non-power-of-two sets
        return (key >> 6) % self.sets

    def get(self, key: int, pc_tag: int) -> Optional[Tuple[int, int, int, int]]:
        s = self._set(key)
        for tup in self.table[s]:
            if tup[0] == key and tup[1] == pc_tag:
                return tup
        return None

    def insert(
        self, key: int, pc_tag: int, target: int, ts: int
    ) -> Optional[Tuple[int, int, int, int]]:
        s = self._set(key)
        evicted = None
        if len(self.table[s]) == self.table[s].maxlen:
            evicted = self.table[s].popleft()
        self.table[s].append((key, pc_tag, target, ts))
        return evicted


class SecondChanceSampler:
    def __init__(self, cap: int = 64):
        self.cap = cap
        self.q: Deque[Tuple[int, int, int]] = deque()
        self.index: Dict[int, Tuple[int, int]] = {}

    def put(self, target: int, pc_tag: int, deadline: int):
        if target in self.index:
            return
        if len(self.q) >= self.cap:
            old_t, _, _ = self.q.popleft()
            self.index.pop(old_t, None)
        self.q.append((target, pc_tag, deadline))
        self.index[target] = (pc_tag, deadline)

    def hit(self, addr: int, pc_tag: int, now: int) -> bool:
        rec = self.index.pop(addr, None)
        if not rec:
            return False
        rec_pc, deadline = rec
        return (rec_pc == pc_tag) and (now <= deadline)

    def age_out(self, now: int):
        while self.q and (self.q[0][2] < now or self.q[0][0] not in self.index):
            t, _, _ = self.q.popleft()
            self.index.pop(t, None)


class MetadataReuseBuffer:
    def __init__(self, sets: int = 128, ways: int = 2):
        assert sets > 0 and ways > 0
        self.sets = sets
        self.ways = ways
        self.lines: List[Deque[Tuple[int, TriangelMeta]]] = [
            deque(maxlen=ways) for _ in range(sets)
        ]

    def _set(self, key: int) -> int:
        return (key >> 6) % self.sets

    def get(self, key: int) -> Optional[TriangelMeta]:
        s = self._set(key)
        for k, meta in list(self.lines[s]):
            if k == key:
                # move-to-back MRU
                self.lines[s].remove((k, meta))
                self.lines[s].append((k, meta))
                return meta
        return None

    def put(self, key: int, meta: TriangelMeta):
        s = self._set(key)
        # remove existing
        for k, m in list(self.lines[s]):
            if k == key:
                self.lines[s].remove((k, m))
                break
        if len(self.lines[s]) == self.lines[s].maxlen:
            self.lines[s].popleft()
        self.lines[s].append((key, meta))


class TriangelPrefetcher(PrefetchAlgorithm):
    BASE_UP, BASE_DOWN = 1, 2
    HIGH_UP, HIGH_DOWN = 1, 5

    def __init__(
        self,
        num_ways: int = 1,
        init_size: Size = Size.from_kb(512),
        min_size: Size = Size(0),
        max_size: Size = Size.from_mb(1),
        resize_epoch: int = 50_000,
        grow_thresh: float = 0.06,
        shrink_thresh: float = 0.03,
        sampler_sets: int = 64,
        sampler_ways: int = 2,
        scs_cap: int = 64,
        mrb_sets: int = 128,
        mrb_ways: int = 2,
        max_degree: int = 4,
        l2_lines_hint: int = 512,
        replacement_policy_cls=HawkeyeReplacementPolicy,
    ):
        self.num_ways = max(1, int(num_ways))
        self.num_sets = max(1, int(init_size.bytes) // max(1, self.num_ways))
        self.min_size = (
            int(min_size) if isinstance(min_size, int) else int(min_size.bytes)
        )
        self.max_size = (
            int(max_size) if isinstance(max_size, int) else int(max_size.bytes)
        )

        # Use provided Cache class. Policy instances are per-set inside Cache.
        self.cache = Cache(
            num_sets=self.num_sets,
            num_ways=self.num_ways,
            replacement_policy_cls=replacement_policy_cls,
        )

        self.training: Dict[int, TrainEntry] = {}
        self.sampler = HistorySampler(sampler_sets, sampler_ways)
        self.scs = SecondChanceSampler(scs_cap)
        self.mrb = MetadataReuseBuffer(mrb_sets, mrb_ways)
        self.max_degree = max(1, int(max_degree))
        self.l2_lines_hint = int(l2_lines_hint)

        self.resize_epoch = int(resize_epoch)
        self.grow_thresh = float(grow_thresh)
        self.shrink_thresh = float(shrink_thresh)
        self.meta_accesses = 0
        self.useful_prefetches = 0
        self.issued_prefetches = 0
        self.prefetch_chain_hits = 0

        self.prev_access: Optional[MemoryAccess] = None

        logger.info(
            "Triangel init: ways=%d sets=%d (≈%d entries) MRB=%dx%d Sampler=%dx%d SCS=%d",
            self.num_ways,
            self.num_sets,
            self.num_sets * self.num_ways,
            mrb_sets,
            mrb_ways,
            sampler_sets,
            sampler_ways,
            scs_cap,
        )

    def init(self):
        # Use Cache.flush to keep consistent
        self.cache.flush()
        self.training.clear()
        self.meta_accesses = 0
        self.useful_prefetches = 0
        self.issued_prefetches = 0
        self.prefetch_chain_hits = 0
        self.prev_access = None
        logger.info("Triangel reset complete")

    def close(self):
        logger.info(
            "Triangel closed: meta_entries=%d ways=%d useful=%d issued=%d (%.1f%%) chain_hits=%d",
            len(self.cache),
            self.num_ways,
            self.useful_prefetches,
            self.issued_prefetches,
            (100.0 * self.useful_prefetches / max(1, self.issued_prefetches)),
            self.prefetch_chain_hits,
        )
        self.cache.flush()

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        addr, pc = int(access.address), int(access.pc)
        preds: List[int] = []

        if prefetch_hit:
            self.useful_prefetches += 1
            if self.prev_access is not None:
                try:
                    self.cache.prefetch_hit(int(self.prev_access.address))
                except Exception:
                    pass

        te = self._train_row(pc)

        meta = self._lookup_markov(addr)
        if meta is not None and self._allow_issue(te):
            preds = self._prefetch_chain(addr, meta, te)
            self.issued_prefetches += len(preds)

        te.time += 1
        prev_x = te.last0
        te.last1 = te.last0
        te.last0 = addr

        self._update_samplers_and_conf(te, prev_x, addr)

        if prev_x is not None and self._allow_train(te):
            index = te.last1 if te.lookahead2 and te.last1 is not None else prev_x
            self._train_markov(index, addr)

        self.meta_accesses += 1
        if self.meta_accesses >= self.resize_epoch:
            self._maybe_resize()
            self.meta_accesses = 0
            self.useful_prefetches = 0
            self.issued_prefetches = 0
            self.prefetch_chain_hits = 0

        self.prev_access = access
        return preds

    # Markov / MRB helpers using Cache API consistently
    def _lookup_markov(self, key: int) -> Optional[TriangelMeta]:
        # 1) MRB quick check
        m = self.mrb.get(key)
        if m is not None:
            self.prefetch_chain_hits += 1
            return m
        # 2) main cache
        m = self.cache.get(key)
        if isinstance(m, TriangelMeta):
            # populate MRB to speed up subsequent chained lookups
            self.mrb.put(key, m)
            return m
        return None

    def _train_markov(self, index: int, target: int):
        cur = self.cache.get(index)
        if not isinstance(cur, TriangelMeta):
            cur = TriangelMeta(neighbor=target, conf=0)
        else:
            if cur.neighbor == target:
                cur.conf = 1
            elif cur.conf == 0:
                cur.neighbor = target
            else:
                cur.conf = 0
        # use Cache.put to insert/update metadata
        self.cache.put(index, cur)
        # also update MRB so chained walk sees newest value
        self.mrb.put(index, cur)
        logger.debug("Markov train: [%#x] -> [%#x] conf=%d", index, target, cur.conf)

    def _prefetch_chain(
        self, start_key: int, meta: TriangelMeta, te: TrainEntry
    ) -> List[int]:
        preds: List[int] = []
        degree = self._select_degree(te)
        seen = set()
        cur_meta = meta
        steps = 0

        while steps < degree:
            tgt = cur_meta.neighbor
            if tgt in seen:
                break
            preds.append(tgt)
            seen.add(tgt)
            nxt = self._lookup_markov(tgt)
            if nxt is None:
                break
            cur_meta = nxt
            steps += 1

        if len(preds) == 1 and degree == 1:
            first_tgt = preds[0]
            nxt_meta = self._lookup_markov(first_tgt)
            if nxt_meta is not None:
                second = nxt_meta.neighbor
                if second not in seen:
                    preds.append(second)

        if preds:
            logger.debug(
                "Prefetch chain deg=%d look2=%s pattBase=%d pattHigh=%d -> %s",
                degree,
                te.lookahead2,
                te.patt_base,
                te.patt_high,
                [hex(p) for p in preds],
            )
        return preds

    # Sampling & update logic unchanged
    def _update_samplers_and_conf(
        self, te: TrainEntry, prev_x: Optional[int], curr: int
    ):
        self.scs.age_out(te.time)

        if prev_x is None:
            return

        hit = self.sampler.get(prev_x, te.pc_tag)
        if hit:
            _, pc_tag, target_y, ts = hit
            max_entries = max(1, (self.num_sets * self.num_ways) // 5)
            local_dist = te.time - ts
            if local_dist <= max_entries:
                te.reuse_conf = min(15, te.reuse_conf + 1)

            if curr == target_y:
                te.patt_base = min(15, te.patt_base + self.BASE_UP)
                te.patt_high = min(15, te.patt_high + self.HIGH_UP)
            else:
                self.scs.put(target_y, te.pc_tag, te.time + self.l2_lines_hint)

        if self.scs.hit(curr, te.pc_tag, te.time):
            te.patt_base = min(15, te.patt_base + self.BASE_UP)
            te.patt_high = min(15, te.patt_high + self.HIGH_UP)
        else:
            if hit and curr != hit[2]:
                te.patt_base = max(0, te.patt_base - self.BASE_DOWN)
                te.patt_high = max(0, te.patt_high - self.HIGH_DOWN)

        self._maybe_sample(prev_x, te, curr)

        if te.patt_high >= 15 and not te.lookahead2:
            te.lookahead2 = True
            logger.info("PC %#x entering lookahead-2", te.pc_tag)
        elif te.patt_base < 8 and te.lookahead2:
            te.lookahead2 = False
            logger.info("PC %#x reverting to lookahead-1", te.pc_tag)

    def _maybe_sample(self, key_x: int, te: TrainEntry, target_y: int):
        sampler_size = self.sampler.sets * self.sampler.ways
        max_entries = max(1, (self.num_sets * self.num_ways) // 5)
        scale = max(1.0, sampler_size / max_entries)
        prob = min(1.0, scale * (2 ** (te.sample_rate - 8)))
        if random.random() < prob:
            ev = self.sampler.insert(key_x, te.pc_tag, target_y, te.time)
            if ev:
                _, ev_pc, _, ev_ts = ev
                if (te.time - ev_ts) > max_entries and ev_pc in self.training:
                    self.training[ev_pc].reuse_conf = max(
                        0, self.training[ev_pc].reuse_conf - 1
                    )
                    te.sample_rate = min(15, te.sample_rate + 1)
                else:
                    te.sample_rate = max(0, te.sample_rate - 1)

    def _train_row(self, pc: int) -> TrainEntry:
        te = self.training.get(pc)
        if te is None:
            te = TrainEntry(pc_tag=(pc & ((1 << 10) - 1)))
            self.training[pc] = te
        return te

    def _allow_train(self, te: TrainEntry) -> bool:
        return te.reuse_conf >= 4 and te.patt_base >= 9

    def _allow_issue(self, te: TrainEntry) -> bool:
        return te.patt_base >= 8

    def _select_degree(self, te: TrainEntry) -> int:
        if te.patt_high >= 12:
            return self.max_degree
        if te.patt_high >= 10:
            return max(2, self.max_degree - 1)
        if te.patt_high >= 9:
            return 2
        return 1

    def _maybe_resize(self):
        ratio = self.useful_prefetches / max(1, self.issued_prefetches)
        old_ways = self.num_ways
        if (
            ratio >= self.grow_thresh
            and (self.num_sets * (self.num_ways + 1)) <= self.max_size
        ):
            self.num_ways += 1
            # delegate structural change to Cache.change_num_ways
            self.cache.change_num_ways(self.num_ways)
            logger.info("Triangel: GROW ways to %d", self.num_ways)
        elif ratio <= self.shrink_thresh and self.num_ways > 1:
            self.num_ways = max(1, self.num_ways - 1)
            self.cache.change_num_ways(self.num_ways)
            logger.info("Triangel: SHRINK ways to %d", self.num_ways)
        else:
            logger.debug("Triangel: KEEP ways=%d (ratio=%.3f)", self.num_ways, ratio)
