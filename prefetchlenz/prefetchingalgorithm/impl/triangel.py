import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.hawkeye import HawkeyeReplacementPolicy
from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm
from prefetchlenz.util.size import Size

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.neural")


# ----------------------------- Triangel Structures -----------------------------


@dataclass
class TriangelMeta:
    """Compressed Markov entry payload (neighbor + 1-bit confidence)."""

    neighbor: int
    conf: int = 0  # 0/1: replace only if 0, set to 1 on agreement


@dataclass
class TrainEntry:
    """Per-PC training-row (Triangel §4.2)."""

    pc_tag: int
    last0: Optional[int] = None
    last1: Optional[int] = None
    time: int = 0
    reuse_conf: int = 0  # 4-bit in paper; we'll clamp [0,15]
    patt_base: int = 8  # biased up/down: +1 / -2
    patt_high: int = 8  # biased up/down: +1 / -5
    sample_rate: int = 8  # 0..15, controls sampler insertion prob
    lookahead2: bool = False  # when True, use last1->curr for training


class HistorySampler:
    """
    Tiny set-assoc sampler storing (index=x) -> (pc_tag, target=y, timestamp).
    Used to estimate reuse distance & pattern accuracy (§4.4).
    """

    def __init__(self, sets: int = 64, ways: int = 2):
        self.sets = sets
        self.ways = ways
        self.table: List[Deque[Tuple[int, int, int, int]]] = [
            deque(maxlen=ways) for _ in range(sets)
        ]
        # each entry: (key_x, pc_tag, target_y, ts)

    def _set(self, key: int) -> int:
        return (key >> 6) & (self.sets - 1)  # spread by line granularity

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
            evicted = self.table[s][0]
            self.table[s].popleft()
        self.table[s].append((key, pc_tag, target, ts))
        return evicted


class SecondChanceSampler:
    """
    Small FIFO buffer of “missed” targets to credit accurate-but-nonsequential uses (§4.4.2).
    Stores (target_y -> (pc_tag, deadline_ts)).
    """

    def __init__(self, cap: int = 64):
        self.cap = cap
        self.q: Deque[Tuple[int, int, int]] = deque()  # (target, pc_tag, deadline)
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
        rec = self.index.get(addr)
        if not rec:
            return False
        rec_pc, deadline = rec
        ok = (rec_pc == pc_tag) and (now <= deadline)
        # consume entry either way
        self.index.pop(addr, None)
        # remove from deque lazily
        return ok

    def age_out(self, now: int):
        while self.q and (self.q[0][2] < now or self.q[0][0] not in self.index):
            t, _, _ = self.q.popleft()
            self.index.pop(t, None)


class MetadataReuseBuffer:
    """
    Tiny 2-way set-assoc buffer to avoid redundant L3 Markov lookups (§4.6).
    Keyed by lookup address; stores TriangelMeta.
    """

    def __init__(self, sets: int = 128, ways: int = 2):
        self.sets = sets
        self.ways = ways
        self.lines: List[Deque[Tuple[int, TriangelMeta]]] = [
            deque(maxlen=ways) for _ in range(sets)
        ]

    def _set(self, key: int) -> int:
        return (key >> 6) & (self.sets - 1)

    def get(self, key: int) -> Optional[TriangelMeta]:
        s = self._set(key)
        for i, (k, meta) in enumerate(self.lines[s]):
            if k == key:
                # move-to-back (MRU-ish)
                self.lines[s].remove((k, meta))
                self.lines[s].append((k, meta))
                return meta
        return None

    def put(self, key: int, meta: TriangelMeta):
        s = self._set(key)
        # dedup
        for i, (k, m) in enumerate(self.lines[s]):
            if k == key:
                self.lines[s].remove((k, m))
                break
        if len(self.lines[s]) == self.lines[s].maxlen:
            self.lines[s].popleft()
        self.lines[s].append((key, meta))


# --------------------------------- Prefetcher ----------------------------------


class TriangelPrefetcher(PrefetchAlgorithm):
    """
    Triangel prefetcher (PC-local temporal + sampling, lookahead/degree control).
    - PC-scoped training with History/Second-Chance sampling.
    - Markov table kept in your `Cache` via `TriangelMeta` payloads.
    - Aggression control: adaptive lookahead (1→2) and degree (1→4).
    - Tiny metadata reuse buffer to collapse redundant Markov lookups.
    - Lightweight “dueller-like” resizing: grows/shrinks Markov ways on usefulness.

    Notes:
      * We keep replacement Hawkeye by default to match your cache wiring.
      * All knobs are constructor params with sane defaults.
    """

    # bias factors like paper: Base +1/-2, High +1/-5
    BASE_UP, BASE_DOWN = 1, 2
    HIGH_UP, HIGH_DOWN = 1, 5

    def __init__(
        self,
        num_ways: int = 1,
        init_size: Size = Size.from_kb(512),
        min_size: Size = Size(0),
        max_size: Size = Size.from_mb(1),
        resize_epoch: int = 50_000,
        grow_thresh: float = 0.06,  # a hair stricter than triage
        shrink_thresh: float = 0.03,
        sampler_sets: int = 64,
        sampler_ways: int = 2,
        scs_cap: int = 64,
        mrb_sets: int = 128,
        mrb_ways: int = 2,
        max_degree: int = 4,
        l2_lines_hint: int = 512,  # second-chance deadline window
        replacement_policy_cls=HawkeyeReplacementPolicy,
    ):
        # Capacity bookkeeping (same style as your Triage)
        self.num_ways = num_ways
        self.num_sets = init_size.bytes // max(1, num_ways)
        self.min_size = int(min_size)
        self.max_size = int(max_size)

        self.cache = Cache(
            num_sets=self.num_sets,
            num_ways=self.num_ways,
            replacement_policy_cls=replacement_policy_cls,
        )

        # Triangel state
        self.training: Dict[int, TrainEntry] = {}
        self.sampler = HistorySampler(sampler_sets, sampler_ways)
        self.scs = SecondChanceSampler(scs_cap)
        self.mrb = MetadataReuseBuffer(mrb_sets, mrb_ways)
        self.max_degree = max_degree
        self.l2_lines_hint = l2_lines_hint

        # Stats & adaptation
        self.resize_epoch = resize_epoch
        self.grow_thresh = grow_thresh
        self.shrink_thresh = shrink_thresh
        self.meta_accesses = 0
        self.useful_prefetches = 0
        self.issued_prefetches = 0
        self.prefetch_chain_hits = 0

        self.prev_access: Optional[MemoryAccess] = None

        logger.info(
            "Triangel init: ways=%d sets=%d (≈%dB), MRB=%dx%d, Sampler=%dx%d, SCS=%d",
            self.num_ways,
            self.num_sets,
            self.num_ways * self.num_sets,
            mrb_sets,
            mrb_ways,
            sampler_sets,
            sampler_ways,
            scs_cap,
        )

    # ------------------------------- API hooks ---------------------------------

    def init(self):
        self.cache.flush()
        self.training.clear()
        # keep samplers across phases; they’re tiny. If you want, clear here.
        self.meta_accesses = 0
        self.useful_prefetches = 0
        self.issued_prefetches = 0
        self.prefetch_chain_hits = 0
        self.prev_access = None
        logger.info("Triangel reset complete")

    def close(self):
        logger.info(
            "Triangel closed: entries=%d, ways=%d, useful=%d / issued=%d (%.1f%%), chain_hits=%d",
            len(self.cache),
            self.num_ways,
            self.useful_prefetches,
            self.issued_prefetches,
            (100.0 * self.useful_prefetches / max(1, self.issued_prefetches)),
            self.prefetch_chain_hits,
        )
        self.cache.flush()

    # --------------------------------- Core ------------------------------------

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        addr, pc = access.address, access.pc
        preds: List[int] = []

        if prefetch_hit:
            self.useful_prefetches += 1
            if self.prev_access is not None:
                self.cache.prefetch_hit(self.prev_access.address)

        te = self._train_row(pc)

        # 1) Lookup & issue prefetch chain based on metadata and confidence
        meta = self._lookup_markov(addr)
        if meta is not None and self._allow_issue(te):
            preds = self._prefetch_chain(addr, meta, te)
            self.issued_prefetches += len(preds)

        # 2) Update training row timestamp and shift register
        te.time += 1
        prev_x = te.last0
        te.last1 = te.last0
        te.last0 = addr

        # 3) History sampling: reuse + pattern confidence updates
        self._update_samplers_and_conf(te, prev_x, addr)

        # 4) Train Markov if strong enough
        if prev_x is not None and self._allow_train(te):
            index = te.last1 if te.lookahead2 and te.last1 is not None else prev_x
            self._train_markov(index, addr)

        # 5) periodic resizing (dueller-lite using usefulness ratio)
        self.meta_accesses += 1
        if self.meta_accesses >= self.resize_epoch:
            self._maybe_resize()
            self.meta_accesses = 0
            self.useful_prefetches = 0
            self.issued_prefetches = 0
            self.prefetch_chain_hits = 0

        self.prev_access = access
        return preds

    # --------------------------- Markov / MRB helpers --------------------------

    def _lookup_markov(self, key: int) -> Optional[TriangelMeta]:
        # MRB first
        m = self.mrb.get(key)
        if m is not None:
            self.prefetch_chain_hits += 1
            return m
        # L3/Cache next
        m = self.cache.get(key)
        if isinstance(m, TriangelMeta):
            self.mrb.put(key, m)
            return m
        return None

    def _train_markov(self, index: int, target: int):
        cur: Optional[TriangelMeta] = self.cache.get(index)
        if cur is None:
            cur = TriangelMeta(neighbor=target, conf=0)
        else:
            if cur.neighbor == target:
                cur.conf = 1
            elif cur.conf == 0:
                cur.neighbor = target
            else:
                cur.conf = 0  # disagreement: drop confidence
        self.cache.put(index, cur)
        self.mrb.put(index, cur)  # keep MRB coherent
        logger.debug("Markov train: [%#x] -> [%#x], conf=%d", index, target, cur.conf)

    def _prefetch_chain(
        self, start_key: int, meta: TriangelMeta, te: TrainEntry
    ) -> List[int]:
        preds: List[int] = []
        degree = self._select_degree(te)
        seen = set()
        key = start_key
        cur = meta
        steps = 0

        while steps < degree:
            tgt = cur.neighbor
            if tgt in seen:
                break
            preds.append(tgt)
            seen.add(tgt)
            # walk chain
            nxt = self._lookup_markov(tgt)
            if nxt is None:
                break
            cur = nxt
            key = tgt
            steps += 1

        if preds:
            logger.debug(
                "Prefetch chain (deg=%d, look2=%s, pattBase=%d pattHigh=%d): %s",
                degree,
                te.lookahead2,
                te.patt_base,
                te.patt_high,
                [hex(p) for p in preds],
            )
        return preds

    # ------------------------------- Sampling ----------------------------------

    def _update_samplers_and_conf(
        self, te: TrainEntry, prev_x: Optional[int], curr: int
    ):
        # sampler aging & housekeeping
        self.scs.age_out(te.time)

        if prev_x is None:
            return

        # ReuseConf + PatternConf via HistorySampler
        hit = self.sampler.get(prev_x, te.pc_tag)
        if hit:
            _, pc_tag, target_y, ts = hit
            # local reuse distance check (approx MaxSize by cache bytes / entry size ~ 42b -> ~5B)
            max_entries = max(1, (self.num_sets * self.num_ways) // 5)
            local_dist = te.time - ts
            if local_dist <= max_entries:
                te.reuse_conf = min(15, te.reuse_conf + 1)

            if curr == target_y:
                te.patt_base = min(15, te.patt_base + self.BASE_UP)
                te.patt_high = min(15, te.patt_high + self.HIGH_UP)
            else:
                # put into SCS; deadline within L2-sized fill window
                self.scs.put(target_y, te.pc_tag, te.time + self.l2_lines_hint)
                # penalize only if SCS later misses
        else:
            # no sampler hit: no immediate update to confidence
            pass

        # Second-Chance credit (if current matches a queued target)
        if self.scs.hit(curr, te.pc_tag, te.time):
            te.patt_base = min(15, te.patt_base + self.BASE_UP)
            te.patt_high = min(15, te.patt_high + self.HIGH_UP)
        else:
            # If we *had* a history hit that disagreed and SCS didn't save it, penalize.
            if hit and curr != hit[2]:
                te.patt_base = max(0, te.patt_base - self.BASE_DOWN)
                te.patt_high = max(0, te.patt_high - self.HIGH_DOWN)

        # Insert a new sample probabilistically (§4.4.3)
        self._maybe_sample(prev_x, te, curr)

        # Lookahead toggle: enter lookahead-2 at high certainty; exit when base collapses
        if te.patt_high >= 15:
            if not te.lookahead2:
                te.lookahead2 = True
                logger.info("PC %#x entering lookahead-2", te.pc_tag)
        elif te.patt_base < 8 and te.lookahead2:
            te.lookahead2 = False
            logger.info("PC %#x reverting to lookahead-1", te.pc_tag)

    def _maybe_sample(self, key_x: int, te: TrainEntry, target_y: int):
        # probability ~ (sampler_size / max_entries) * 2^(rate-8)
        sampler_size = self.sampler.sets * self.sampler.ways
        max_entries = max(1, (self.num_sets * self.num_ways) // 5)
        scale = max(1.0, sampler_size / max_entries)
        prob = min(1.0, scale * (2 ** (te.sample_rate - 8)))
        if random.random() < prob:
            ev = self.sampler.insert(key_x, te.pc_tag, target_y, te.time)
            if ev:
                # adapt sampling rate depending on whether we evicted a “useful” sample
                _, ev_pc, _, ev_ts = ev
                if (te.time - ev_ts) > max_entries:
                    # evicted stale: victim PC had long reuse; increase our sample rate
                    if ev_pc in self.training:
                        self.training[ev_pc].reuse_conf = max(
                            0, self.training[ev_pc].reuse_conf - 1
                        )
                    te.sample_rate = min(15, te.sample_rate + 1)
                else:
                    # evicted potentially useful: back off a bit
                    te.sample_rate = max(0, te.sample_rate - 1)

    # ------------------------------ Policies -----------------------------------

    def _train_row(self, pc: int) -> TrainEntry:
        te = self.training.get(pc)
        if te is None:
            te = TrainEntry(pc_tag=pc & ((1 << 10) - 1))  # lightweight tag-ish
            self.training[pc] = te
        return te

    def _allow_train(self, te: TrainEntry) -> bool:
        # Train only when we’re at least moderately confident in both reuse & pattern (§4.5)
        return te.reuse_conf >= 4 and te.patt_base >= 9

    def _allow_issue(self, te: TrainEntry) -> bool:
        # Issue when base confidence is decent (≥9)
        return te.patt_base >= 9

    def _select_degree(self, te: TrainEntry) -> int:
        if te.patt_high >= 12:
            return self.max_degree
        if te.patt_high >= 10:
            return max(2, self.max_degree - 1)
        if te.patt_high >= 9:
            return 2
        return 1

    # ------------------------------ Resizing -----------------------------------

    def _maybe_resize(self):
        # Very lightweight “dueller-like” resizing: prefer more ways when prefetches are useful.
        ratio = self.useful_prefetches / max(1, self.issued_prefetches)
        old_bytes = int(self.num_ways * self.num_sets)
        grew = shrank = False
        if ratio >= self.grow_thresh and old_bytes < self.max_size:
            self.num_ways += 1
            self.cache.change_num_ways(self.num_ways)
            grew = True
        elif ratio <= self.shrink_thresh and old_bytes > self.min_size:
            self.num_ways = max(1, self.num_ways - 1)
            self.cache.change_num_ways(self.num_ways)
            shrank = True
        new_bytes = int(self.num_ways * self.num_sets)
        logger.info(
            "Resize window: useful=%d issued=%d ratio=%.3f %s -> %s (%d→%d bytes)",
            self.useful_prefetches,
            self.issued_prefetches,
            ratio,
            "GROW" if grew else ("SHRINK" if shrank else "KEEP"),
            f"ways={self.num_ways}",
            old_bytes,
            new_bytes,
        )
