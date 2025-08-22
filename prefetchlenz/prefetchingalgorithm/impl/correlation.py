import logging
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.correlation")


@dataclass
class _Succ:
    addr: int
    cnt: int


class _LRUMap:
    """
    Small, set-associative style map using an OrderedDict as a fully-assoc LRU.
    Holds: key(int) -> Counter(int->int) of successor counts.
    Evicts the least-recently used 'key' when capacity is exceeded.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.map: "OrderedDict[int, Counter]" = OrderedDict()

    def get(self, key: int) -> Optional[Counter]:
        c = self.map.get(key)
        if c is None:
            return None
        self.map.move_to_end(key)
        return c

    def touch(self, key: int):
        if key in self.map:
            self.map.move_to_end(key)

    def put_if_absent(self, key: int) -> Counter:
        c = self.map.get(key)
        if c is not None:
            self.map.move_to_end(key)
            return c
        # evict if full
        if len(self.map) >= self.capacity:
            ev_k, ev_v = self.map.popitem(last=False)
            logger.debug("corr: evict trigger 0x%x with %d succs", ev_k, len(ev_v))
        c = Counter()
        self.map[key] = c
        return c

    def items(self):
        return self.map.items()

    def clear(self):
        self.map.clear()


class CorrelationPrefetcher(PrefetchAlgorithm):
    """
    Correlation Prefetcher with 'user-level memory thread' behavior (software model).
    Key ideas from the paper:
      - Learn (trigger -> successor) relationships among blocks that recur within a small miss-distance window.
      - On each access to trigger T, issue top-K successors S1..Sk (prefetch degree).
      - When a prefetched successor becomes a demand hit later, *chain* by issuing its successors
        (emulating the ULT continuing along the correlation graph).
      - Periodic aging of counts to keep the table responsive.

    Interfaces:
      - init()
      - progress(access: MemoryAccess, prefetch_hit: bool) -> List[int]  # addresses to prefetch
      - close()
    """

    def __init__(
        self,
        window: int = 16,  # correlation window (in intervening demand accesses)
        table_capacity: int = 8192,  # max distinct trigger entries kept (LRU)
        max_succ_per_trigger: int = 16,  # bound stored successors per trigger (prune to top-N)
        degree: int = 4,  # prefetch degree per trigger
        min_count: int = 2,  # min occurrences before a successor is considered
        chain_on_prefetch_hit: bool = True,
        aging_interval: int = 50_000,  # accesses between count aging
        aging_factor: float = 0.5,  # multiply counts by this on aging
    ):
        self.window = max(1, window)
        self.table = _LRUMap(capacity=table_capacity)
        self.max_succ_per_trigger = max(1, max_succ_per_trigger)
        self.degree = max(1, degree)
        self.min_count = max(1, min_count)
        self.chain_on_prefetch_hit = chain_on_prefetch_hit
        self.aging_interval = max(1, aging_interval)
        self.aging_factor = max(0.1, min(0.95, aging_factor))

        # Sliding window of recent demand addresses (we correlate *demand* references).
        self.recent: Deque[int] = deque(maxlen=self.window)

        # Outstanding prefetches: tgt_addr -> trigger_addr that issued it
        self.outstanding: Dict[int, int] = {}

        # Stats/housekeeping
        self._accesses = 0
        self._prefetches_issued = 0
        self._chains = 0

    # ---------------- lifecycle ----------------
    def init(self):
        self.table.clear()
        self.recent.clear()
        self.outstanding.clear()
        self._accesses = 0
        self._prefetches_issued = 0
        self._chains = 0
        logger.info(
            "corr: init window=%d table_capacity=%d max_succ=%d degree=%d min_count=%d aging=(%d,%.2f) chain=%s",
            self.window,
            self.table.capacity,
            self.max_succ_per_trigger,
            self.degree,
            self.min_count,
            self.aging_interval,
            self.aging_factor,
            self.chain_on_prefetch_hit,
        )

    def close(self):
        logger.info(
            "corr: close issued=%d chains=%d triggers=%d",
            self._prefetches_issued,
            self._chains,
            len(list(self.table.items())),
        )
        self.table.clear()
        self.recent.clear()
        self.outstanding.clear()

    # ---------------- internals ----------------
    def _age(self):
        """
        Halve (by factor) all successor counts. This is coarse but effective.
        """
        aged = 0
        for trig, succs in self.table.items():
            # Multiply and floor to int, drop zeros
            new_c = Counter()
            for s, c in succs.items():
                nc = int(c * self.aging_factor)
                if nc > 0:
                    new_c[s] = nc
            self.table.map[trig] = new_c  # assign back
            aged += 1
        logger.debug("corr: aged %d triggers", aged)

    def _train_with_address(self, addr: int):
        """
        Update correlations: for each prior address in the window, bump count(prior -> addr).
        We also bound per-trigger successor set to top-N by count.
        """
        # Correlate with *distinct* prior addresses to avoid inflating self-correlation from bursts
        seen = set()
        for prev in reversed(self.recent):
            if prev == addr:  # avoid self-pairing (optional)
                continue
            if prev in seen:
                continue
            seen.add(prev)

            succs = self.table.put_if_absent(prev)
            succs[addr] += 1

            # prune if too many successors stored
            if len(succs) > self.max_succ_per_trigger:
                # keep top-N by count; deterministic tie-breaker by smaller addr
                top = sorted(succs.items(), key=lambda kv: (-kv[1], kv[0]))[
                    : self.max_succ_per_trigger
                ]
                self.table.map[prev] = Counter(dict(top))

        # push current address into the window after training
        self.recent.append(addr)

    def _best_successors(self, trigger: int, k: int) -> List[int]:
        cnts = self.table.get(trigger)
        if not cnts:
            return []
        # choose successors by highest count, tie-breaker by smaller address (stable)
        ranked = sorted(cnts.items(), key=lambda kv: (-kv[1], kv[0]))
        result: List[int] = []
        for s, c in ranked:
            if c < self.min_count:
                continue
            result.append(s)
            if len(result) >= k:
                break
        return result

    def _record_outstanding(self, origin: int, targets: List[int]):
        for t in targets:
            # only remember the first issuer if multiple would add the same target
            self.outstanding.setdefault(t, origin)

    # ---------------- main entry ----------------
    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Train on the demand 'access' and emit prefetch targets.
        If 'prefetch_hit' is True, and chaining is enabled, continue from that address as a trigger.
        """
        self._accesses += 1
        if self._accesses % self.aging_interval == 0:
            self._age()

        addr = int(access.address)

        # (A) If this access hits a previously issued prefetch, optionally chain
        chained_targets: List[int] = []
        if prefetch_hit and self.chain_on_prefetch_hit:
            origin = self.outstanding.pop(addr, None)
            if origin is not None:
                # treat current addr as a trigger; this emulates the ULT continuing along the graph
                chained_targets = self._best_successors(addr, self.degree)
                if chained_targets:
                    self._chains += 1
                    logger.debug(
                        "corr: chain from 0x%x -> %s",
                        addr,
                        [hex(x) for x in chained_targets],
                    )

        # (B) Normal trigger: current demand address
        direct_targets = self._best_successors(addr, self.degree)

        # Combine (avoid duplicates; chained first to encourage deeper walk)
        preds: List[int] = []
        seen = set()
        for t in chained_targets + direct_targets:
            if t != addr and t not in seen:
                preds.append(t)
                seen.add(t)

        # Record outstanding so we can credit chain on future hits
        if preds:
            self._record_outstanding(addr, preds)
            self._prefetches_issued += len(preds)
            logger.info("corr: trigger 0x%x -> preds=%s", addr, [hex(x) for x in preds])

        # (C) Train correlations using this demand address
        self._train_with_address(addr)

        return preds
