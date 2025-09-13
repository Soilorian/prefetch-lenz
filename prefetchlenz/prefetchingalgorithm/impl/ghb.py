import logging
import time
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.ghb")
logger.addHandler(logging.NullHandler())


# ----------------------------- Utilities / Small Pieces -----------------------


class UidGenerator:
    """Monotonic UID generator (resettable)."""

    def __init__(self, start: int = 1):
        self._next = int(start)

    def next(self) -> int:
        uid = self._next
        self._next += 1
        return uid

    def reset(self, start: int = 1) -> None:
        self._next = int(start)


# ------------------------------- Data Records --------------------------------


@dataclass
class GHBEntry:
    """
    One GHB entry.
      uid      : unique id
      addr     : aligned address stored
      pc       : instruction pc that caused miss
      prev_uid : previous uid for same pc (or None)
      ts       : insertion timestamp (float)
    """

    uid: int
    addr: int
    pc: int
    prev_uid: Optional[int]
    ts: float


# ------------------------------- Storage Blocks ------------------------------


class IndexTable:
    """
    Bounded index table mapping PC -> head UID (most recent).
    Implemented as an LRU (OrderedDict). When capacity exceeded the least-recently-used
    PC mapping is evicted.
    """

    def __init__(self, capacity: int = 4096):
        self.capacity = max(1, int(capacity))
        self._table: OrderedDict[int, int] = OrderedDict()

    def get_head(self, pc: int) -> Optional[int]:
        uid = self._table.get(pc)
        if uid is not None:
            # mark most-recently used
            self._table.move_to_end(pc, last=True)
        return uid

    def set_head(self, pc: int, uid: int) -> None:
        pc = int(pc)
        uid = int(uid)
        self._table[pc] = uid
        self._table.move_to_end(pc, last=True)
        # enforce capacity
        while len(self._table) > self.capacity:
            old_pc, old_uid = self._table.popitem(last=False)
            logger.debug(
                "IndexTable: evicted pc=0x%X uid=%d due to capacity", old_pc, old_uid
            )

    def clear(self) -> None:
        self._table.clear()
        logger.debug("IndexTable: cleared")

    def __len__(self) -> int:
        return len(self._table)


class GlobalHistoryBuffer:
    """
    Circular GHB with uid->index mapping.
    Stores aligned addresses and prev_uid links. On overwrite the mapping for the old uid
    is removed so subsequent lookups see that uid as evicted.
    """

    def __init__(self, capacity: int = 1024, line_size: int = 64):
        self.capacity = max(1, int(capacity))
        self.line_size = max(1, int(line_size))
        self._buf: List[Optional[GHBEntry]] = [None] * self.capacity
        self._uid_to_index: Dict[int, int] = {}
        self._ptr = 0
        self._uids = UidGenerator()
        logger.info(
            "GHB created capacity=%d line_size=%d", self.capacity, self.line_size
        )

    def _align(self, addr: int) -> int:
        mask = ~(self.line_size - 1)
        return int(addr) & mask

    def push(self, addr: int, pc: int, prev_uid: Optional[int]) -> int:
        """
        Insert a new aligned GHB entry. Returns the uid.
        """
        addr_a = self._align(addr)
        uid = self._uids.next()
        idx = self._ptr

        old = self._buf[idx]
        if old is not None and old.uid in self._uid_to_index:
            # evict mapping for overwritten entry
            del self._uid_to_index[old.uid]
            logger.debug("GHB: evicting old uid=%d at idx=%d", old.uid, idx)

        entry = GHBEntry(
            uid=uid, addr=addr_a, pc=int(pc), prev_uid=prev_uid, ts=time.time()
        )
        self._buf[idx] = entry
        self._uid_to_index[uid] = idx
        self._ptr = (self._ptr + 1) % self.capacity

        logger.debug(
            "GHB: push idx=%d uid=%d pc=0x%X addr=0x%X prev=%s",
            idx,
            uid,
            pc,
            addr_a,
            str(prev_uid),
        )
        return uid

    def get_entry(self, uid: int) -> Optional[GHBEntry]:
        idx = self._uid_to_index.get(int(uid))
        if idx is None:
            return None
        return self._buf[idx]

    def collect_chain_addresses(self, head_uid: int, max_items: int) -> List[int]:
        """
        Walk prev_uid links and return addresses (aligned) in order [head, prev1, prev2, ...].
        Stops if a referenced uid is not in current buffer (evicted).
        """
        addrs: List[int] = []
        uid = head_uid
        steps = 0
        while uid is not None and steps < max_items:
            entry = self.get_entry(uid)
            if entry is None:
                break
            addrs.append(entry.addr)
            uid = entry.prev_uid
            steps += 1
        logger.debug("GHB: collected chain for head=%s -> %s", head_uid, addrs)
        return addrs

    def clear(self) -> None:
        self._buf = [None] * self.capacity
        self._uid_to_index.clear()
        self._ptr = 0
        self._uids.reset()
        logger.info("GHB: cleared")


class CorrelationTable:
    """
    Address->successor Counter store and per-PC delta credits.
    Two maps:
      - addr_map: addr_A -> Counter({addr_B: count})
      - delta_credit: pc -> Counter({delta: count})
    Supports saturating increments, periodic decay, and top-k queries.
    """

    def __init__(self, saturating_max: int = 255, decay_factor: float = 0.5):
        self.addr_map: Dict[int, Counter] = defaultdict(Counter)
        self.delta_credit: Dict[int, Counter] = defaultdict(Counter)
        self.saturating_max = int(saturating_max)
        self.decay_factor = float(decay_factor)
        self._ops_since_decay = 0
        self.decay_interval_ops = 1024  # call decay every N credit ops

    def credit_addr_successor(self, a: int, b: int) -> None:
        a = int(a)
        b = int(b)
        cnt = self.addr_map[a][b] + 1
        if cnt > self.saturating_max:
            cnt = self.saturating_max
        self.addr_map[a][b] = cnt
        self._tick()

        logger.debug("CorrelationTable: credit addr %s -> %s = %d", hex(a), hex(b), cnt)

    def credit_delta(self, pc: int, delta: int) -> None:
        pc = int(pc)
        delta = int(delta)
        cnt = self.delta_credit[pc][delta] + 1
        if cnt > self.saturating_max:
            cnt = self.saturating_max
        self.delta_credit[pc][delta] = cnt
        self._tick()
        logger.debug(
            "CorrelationTable: credit pc=0x%X delta=%d cred=%d", pc, delta, cnt
        )

    def top_successors(self, a: int, topk: int = 4, min_count: int = 1) -> List[int]:
        cnt = self.addr_map.get(int(a))
        if not cnt:
            return []
        most = [x for x, c in cnt.most_common(topk) if c >= min_count]
        return [int(x) for x in most]

    def top_delta(self, pc: int, min_count: int = 1) -> Optional[int]:
        cnt = self.delta_credit.get(int(pc))
        if not cnt:
            return None
        most = cnt.most_common(1)
        if not most or most[0][1] < min_count:
            return None
        return int(most[0][0])

    def decay_all(self) -> None:
        """Decay all counters by factor (saturating to zero)."""

        def half_counter_map(m: Dict[int, Counter]):
            to_delete = []
            for k, cnt in m.items():
                for sk in list(cnt.keys()):
                    old = cnt[sk]
                    new = int(old * self.decay_factor)
                    if new <= 0:
                        del cnt[sk]
                    else:
                        cnt[sk] = new
                if not cnt:
                    to_delete.append(k)
            for k in to_delete:
                del m[k]

        half_counter_map(self.addr_map)
        half_counter_map(self.delta_credit)
        logger.info("CorrelationTable: decay applied")

    def _tick(self) -> None:
        self._ops_since_decay += 1
        if self._ops_since_decay >= self.decay_interval_ops:
            self.decay_all()
            self._ops_since_decay = 0

    def clear(self) -> None:
        self.addr_map.clear()
        self.delta_credit.clear()
        self._ops_since_decay = 0
        logger.debug("CorrelationTable: cleared")


class OutstandingPrefetchTracker:
    """
    Tracks outstanding prefetches with timestamps to detect late vs useful.
    """

    def __init__(self, limit: int = 8192, late_time_threshold: float = 0.01):
        self.limit = max(1, int(limit))
        self._map: Dict[int, Tuple[int, int, float]] = (
            {}
        )  # tgt -> (origin_pc, delta, ts)
        self._queue: Deque[int] = deque()
        self.late_time_threshold = float(late_time_threshold)

    def record(self, tgt: int, pc: int, delta: int) -> None:
        tgt = int(tgt)
        if tgt in self._map:
            return
        self._map[tgt] = (int(pc), int(delta), time.time())
        self._queue.append(tgt)
        if len(self._queue) > self.limit:
            old = self._queue.popleft()
            self._map.pop(old, None)
            logger.debug("Outstanding: evicted oldest tgt=%d", old)
        logger.debug("Outstanding: record tgt=0x%X pc=0x%X delta=%d", tgt, pc, delta)

    def credit_if_present(self, addr: int) -> Optional[Tuple[int, int, bool]]:
        """
        If addr was outstanding, remove it and return (origin_pc, delta, is_late)
        where is_late is True if wall-clock time since record >= late_time_threshold.
        """
        info = self._map.pop(int(addr), None)
        if info is None:
            return None
        try:
            self._queue.remove(int(addr))
        except ValueError:
            pass
        pc, delta, ts = info
        is_late = (time.time() - ts) >= self.late_time_threshold
        logger.debug(
            "Outstanding: credited addr=0x%X pc=0x%X delta=%d late=%s",
            addr,
            pc,
            delta,
            is_late,
        )
        return pc, delta, is_late

    def clear(self) -> None:
        self._map.clear()
        self._queue.clear()
        logger.debug("Outstanding: cleared")

    def __len__(self) -> int:
        return len(self._map)


# ----------------------------- Orchestrator Class ----------------------------


class GlobalHistoryBufferPrefetcher:
    """
    GHB-based correlation prefetcher.

    Features added vs simple implementation:
      - explicit alignment (line_size)
      - bounded IndexTable (LRU eviction)
      - Address->successor correlation table (addr_map)
      - per-PC delta credit counters (delta_credit)
      - saturating counters with periodic decay
      - outstanding tracker with late/useful detection and throttling
      - min_credit_threshold to prefer only sufficiently credited deltas
    """

    def __init__(
        self,
        ghb_size: int = 1024,
        line_size: int = 64,
        index_capacity: int = 4096,
        search_depth: int = 8,
        degree: int = 4,
        prefer_credited: bool = True,
        outstanding_limit: int = 8192,
        min_credit_threshold: int = 1,
        addr_successor_topk: int = 4,
        late_time_threshold: float = 0.01,
    ):
        self.ghb = GlobalHistoryBuffer(capacity=ghb_size, line_size=line_size)
        self.index = IndexTable(capacity=index_capacity)
        self.corr = CorrelationTable()
        self.outstanding = OutstandingPrefetchTracker(
            limit=outstanding_limit, late_time_threshold=late_time_threshold
        )

        self.search_depth = max(1, int(search_depth))
        self.degree = max(1, int(degree))
        self.prefer_credited = bool(prefer_credited)
        self.min_credit_threshold = max(1, int(min_credit_threshold))
        self.addr_successor_topk = max(1, int(addr_successor_topk))

        # simple throttling: do not issue if outstanding count exceeds limit (already enforced in tracker)
        self.outstanding_limit = int(outstanding_limit)

        logger.info(
            "GHBPrefetcher initialized ghb=%d line=%d index_cap=%d search_depth=%d degree=%d prefer_cred=%s",
            ghb_size,
            line_size,
            index_capacity,
            self.search_depth,
            self.degree,
            self.prefer_credited,
        )

    def init(self) -> None:
        """Reset all subcomponents."""
        self.ghb.clear()
        self.index.clear()
        self.corr.clear()
        self.outstanding.clear()
        logger.info("GHBPrefetcher: init")

    def close(self) -> None:
        """Clear state on shutdown."""
        self.ghb.clear()
        self.index.clear()
        self.corr.clear()
        self.outstanding.clear()
        logger.info("GHBPrefetcher: closed")

    def _choose_delta_from_chain(self, chain_addrs: List[int]) -> Optional[int]:
        """Select delta by most frequent then recency fallback."""
        if len(chain_addrs) < 2:
            return None
        deltas = [
            chain_addrs[i] - chain_addrs[i + 1] for i in range(len(chain_addrs) - 1)
        ]
        cnt = Counter(deltas)
        most = cnt.most_common(1)
        if most:
            chosen = int(most[0][0])
            logger.debug("choose_delta: chosen=%d from counts=%s", chosen, most)
            return chosen
        return int(deltas[0])

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Main entry. access must provide .pc and .key/.address.
        prefetch_hit indicates that this access was served by a prefetch.
        Returns list of prefetch target addresses (aligned).
        """
        pc = access.pc
        raw_addr = access.address
        addr = raw_addr  # GHB will align on push

        # 1) If this was a prefetch hit, credit origin (delta) and address→successor mapping if possible
        if prefetch_hit:
            info = self.outstanding.credit_if_present(addr)
            if info is not None:
                origin_pc, delta, was_late = info
                # credit delta for origin PC
                self.corr.credit_delta(origin_pc, delta)
                logger.debug(
                    "progress: credited delta for origin_pc=0x%X delta=%d late=%s",
                    origin_pc,
                    delta,
                    was_late,
                )
                # also, if we can infer an address->successor mapping, credit it:
                # The successor is 'addr' and we may inspect its predecessor in GHB (head for origin_pc)
                # Simple heuristic: if index head for origin_pc exists, link that head -> addr
                head_uid = self.index.get_head(origin_pc)
                if head_uid is not None:
                    head_entry = self.ghb.get_entry(head_uid)
                    if head_entry is not None:
                        self.corr.credit_addr_successor(head_entry.addr, addr)
            else:
                logger.debug(
                    "progress: prefetch_hit but no outstanding record for addr=0x%X",
                    addr,
                )

        # 2) Insert into GHB and update index head
        prev_uid = self.index.get_head(pc)
        new_uid = self.ghb.push(addr=addr, pc=pc, prev_uid=prev_uid)
        self.index.set_head(pc, new_uid)

        # 3) Decide prefetched targets
        chosen_delta: Optional[int] = None

        # (A) prefer per-PC credited delta when above threshold
        if self.prefer_credited:
            top_delta = self.corr.top_delta(pc, min_count=self.min_credit_threshold)
            if top_delta is not None:
                chosen_delta = int(top_delta)
                logger.debug(
                    "progress: using credited delta=%d for pc=0x%X", chosen_delta, pc
                )

        # (B) if no delta chosen, use address->successor correlations for head addr
        if chosen_delta is None:
            head_entry = self.ghb.get_entry(new_uid)
            head_addr = head_entry.addr if head_entry is not None else None
            if head_addr is not None:
                succs = self.corr.top_successors(
                    head_addr,
                    topk=self.addr_successor_topk,
                    min_count=self.min_credit_threshold,
                )
                if succs:
                    # pick first successor and form delta(s) (prefers direct successor)
                    chosen_delta = int(succs[0] - head_addr)
                    logger.debug(
                        "progress: using addr-successor mapping head=0x%X -> succ=0x%X delta=%d",
                        head_addr,
                        succs[0],
                        chosen_delta,
                    )

        # (C) if still None, analyze per-PC chain history
        if chosen_delta is None:
            head = self.index.get_head(pc)
            if head is not None:
                chain_addrs = self.ghb.collect_chain_addresses(
                    head, self.search_depth + 1
                )
                if chain_addrs:
                    chosen_delta = self._choose_delta_from_chain(chain_addrs)
                    logger.debug(
                        "progress: chain analysis pc=0x%X chain=%s chosen_delta=%s",
                        pc,
                        chain_addrs,
                        str(chosen_delta),
                    )

        # (D) fallback: immediate predecessor delta
        if chosen_delta is None:
            entry = self.ghb.get_entry(new_uid)
            if entry is not None and entry.prev_uid is not None:
                prev_entry = self.ghb.get_entry(entry.prev_uid)
                if prev_entry is not None:
                    chosen_delta = int(entry.addr - prev_entry.addr)
                    logger.debug(
                        "progress: fallback delta=%d for pc=0x%X", chosen_delta, pc
                    )

        # 4) If nothing chosen return empty
        if chosen_delta is None:
            logger.debug("progress: pc=0x%X addr=0x%X no delta chosen", pc, addr)
            return []

        # 5) Throttle: if outstanding count already high, do not issue
        if len(self.outstanding) >= self.outstanding_limit:
            logger.info(
                "progress: outstanding limit reached (%d) -> no new prefetches",
                self.outstanding_limit,
            )
            return []

        # 6) Issue degree prefetches (aligned by GHB line_size). Record outstanding and record addr->succ mapping
        prefetches: List[int] = []
        # align base as GHB did
        base = self.ghb._align(addr)
        for i in range(1, self.degree + 1):
            tgt = base + int(chosen_delta) * i
            self.outstanding.record(tgt, pc, int(chosen_delta))
            prefetches.append(tgt)
            # Also record addr->successor mapping (credit target mapping when these are later hit)
            # store mapping head_addr -> tgt for learning (increment heuristically now by 0? defer on hit)
        logger.info(
            "progress: pc=0x%X base=0x%X chosen_delta=%d degree=%d prefetches=%s",
            pc,
            base,
            chosen_delta,
            self.degree,
            [hex(x) for x in prefetches],
        )
        return prefetches
