"""
Graph Prefetching Using Data Structure Knowledge by Ainsworth
"""

from __future__ import annotations

import heapq
import logging
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

from prefetchlenz.prefetchingalgorithm.access.graphmemoryaccess import GraphMemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.graph_hw")
logger.addHandler(logging.NullHandler())


# ---------------------- AddrFilter ------------------------------------------


class AddrFilter:
    """
    Address / PC filter stage.
    Decides which events should become triggers.
    - You can add PC whitelist and/or address-range whitelist.
    - Default: accept all events.
    """

    def __init__(self):
        self.pcs: Set[int] = set()
        self.addr_ranges: List[Tuple[int, int]] = []

    def add_pc(self, pc: int) -> None:
        self.pcs.add(int(pc))

    def add_addr_range(self, lo: int, hi: int) -> None:
        self.addr_ranges.append((int(lo), int(hi)))

    def clear(self) -> None:
        self.pcs.clear()
        self.addr_ranges.clear()

    def pass_filter(self, access: GraphMemoryAccess) -> bool:
        pc = access.pc
        addr = access.address
        if self.pcs and pc not in self.pcs:
            logger.debug("AddrFilter: blocked by PC 0x%X", pc)
            return False
        if self.addr_ranges:
            ok = any(lo <= addr < hi for (lo, hi) in self.addr_ranges)
            if not ok:
                logger.debug("AddrFilter: blocked by addr range 0x%X", addr)
            return ok
        return True


# ---------------------- ObservationQueue ------------------------------------


@dataclass
class ObsRecord:
    ts: float
    pc: int
    addr: int
    stride: int
    inter_arrival: Optional[float]
    latency: Optional[float] = None


class ObservationQueue:
    """
    Keeps short per-PC history records used by the PPU and EWMA.
    - capacity_per_pc: number of recent records to keep per PC.
    """

    def __init__(self, capacity_per_pc: int = 8):
        self.capacity = max(1, int(capacity_per_pc))
        self._hist: Dict[int, Deque[ObsRecord]] = defaultdict(
            lambda: deque(maxlen=self.capacity)
        )
        self._last_addr: Dict[int, int] = {}
        self._last_ts: Dict[int, float] = {}

    def clear(self) -> None:
        self._hist.clear()
        self._last_addr.clear()
        self._last_ts.clear()

    def update(
        self, access: GraphMemoryAccess, latency: Optional[float] = None
    ) -> ObsRecord:
        pc = access.pc
        addr = access.address
        now = time.time()
        prev_addr = self._last_addr.get(pc)
        stride = (addr - prev_addr) if prev_addr is not None else 0
        prev_ts = self._last_ts.get(pc)
        inter_arrival = (now - prev_ts) if prev_ts is not None else None

        rec = ObsRecord(
            ts=now,
            pc=pc,
            addr=addr,
            stride=stride,
            inter_arrival=inter_arrival,
            latency=latency,
        )
        self._hist[pc].append(rec)
        self._last_addr[pc] = addr
        self._last_ts[pc] = now
        logger.debug(
            "ObsQ: pc=0x%X addr=0x%X stride=%d inter=%s",
            pc,
            addr,
            stride,
            str(inter_arrival),
        )
        return rec

    def history(self, access: GraphMemoryAccess, k: int = 4) -> List[ObsRecord]:
        pc = access.pc
        dq = self._hist.get(pc, deque())
        return list(dq)[-k:]


# ---------------------- EwmaCalculator --------------------------------------

from enum import Enum, auto


class Metric(Enum):
    STRIDE = auto()
    LATENCY = auto()
    INTER_ARRIVAL = auto()
    PRESSURE = auto()


@dataclass
class EwmaState:
    value: float = 0.0
    initialized: bool = False


class EwmaCalculator:
    """
    Lightweight per-(pc,metric) EWMA bank.
    - alpha controls smoothing.
    - get_parameters(access) returns dict of current metric values (or None).
    """

    def __init__(self, alpha: float = 0.25):
        assert 0.0 < alpha <= 1.0
        self.alpha = float(alpha)
        self._bank: Dict[Tuple[int, Metric], EwmaState] = {}

    def clear(self) -> None:
        self._bank.clear()

    def _key(self, pc: int, metric: Metric):
        return (int(pc), metric)

    def update(self, pc: int, metric: Metric, sample: float) -> None:
        k = self._key(pc, metric)
        st = self._bank.get(k)
        if st is None:
            self._bank[k] = EwmaState(value=float(sample), initialized=True)
        else:
            st.value = self.alpha * float(sample) + (1.0 - self.alpha) * st.value
            st.initialized = True
        logger.debug(
            "EWMA: pc=0x%X metric=%s -> %.3f", pc, metric.name, self._bank[k].value
        )

    def get(self, pc: int, metric: Metric) -> Optional[float]:
        st = self._bank.get(self._key(pc, metric))
        return st.value if (st and st.initialized) else None

    def get_parameters(self, pc: int) -> Dict[Metric, Optional[float]]:
        return {m: self.get(pc, m) for m in Metric}


# ---------------------- TransitionTable (Correlation) -----------------------


class TransitionTable:
    """
    Stores observed transitions A -> B with counters and provides top-K per A.
    Supports optional periodic decay (aging).
    """

    def __init__(
        self,
        max_successors_per_node: int = 8,
        min_edge_support: int = 2,
        aging_half_life_events: Optional[int] = None,
    ):
        self.transitions: Dict[int, Counter] = defaultdict(Counter)
        self.topk_cache: Dict[int, List[Tuple[int, int]]] = {}
        self._dirty: Set[int] = set()
        self.max_successors_per_node = int(max_successors_per_node)
        self.min_edge_support = int(min_edge_support)
        self.events = 0
        self.aging_half_life_events = aging_half_life_events

    def clear(self) -> None:
        self.transitions.clear()
        self.topk_cache.clear()
        self._dirty.clear()
        self.events = 0

    def observe_edge(self, a: int, b: int) -> None:
        if a == b:
            return
        counts = self.transitions[a]
        before = counts[b]
        counts[b] += 1
        if before == 0:
            pass
        self._dirty.add(a)
        self.events += 1
        if self.aging_half_life_events and (
            self.events % self.aging_half_life_events == 0
        ):
            self._decay_all()

    def _decay_all(self) -> None:
        to_clear = []
        for node, cnt in list(self.transitions.items()):
            for succ in list(cnt.keys()):
                newv = (cnt[succ] + 1) // 2
                if newv <= 0:
                    del cnt[succ]
                else:
                    cnt[succ] = newv
            if not cnt:
                to_clear.append(node)
        for n in to_clear:
            del self.transitions[n]
        self.topk_cache.clear()
        self._dirty.clear()
        logger.info("TransitionTable: decay applied")

    def _refresh_topk(self, a: int) -> None:
        if a not in self._dirty:
            return
        cnt = self.transitions.get(a)
        if not cnt:
            self.topk_cache[a] = []
            self._dirty.discard(a)
            return
        best = heapq.nlargest(
            self.max_successors_per_node, cnt.items(), key=lambda kv: kv[1]
        )
        # keep only edges with at least min_edge_support
        filtered = [(c, s) for (s, c) in best if c >= self.min_edge_support]
        self.topk_cache[a] = filtered
        self._dirty.discard(a)

    def topk_for(self, a: int) -> List[Tuple[int, int]]:
        self._refresh_topk(a)
        return self.topk_cache.get(a, [])


# ---------------------- AddressGenerator / PPU -------------------------------


class AddressGenerator:
    """
    Graph-aware address generator (PPU).
    Uses TransitionTable.topk_for(node) and performs best-first expansion up to hops.
    Returns ordered candidate list (may exceed budget; scheduler trims).
    """

    def __init__(self, transition_table: TransitionTable):
        self.tt = transition_table

    def select_successors(self, start: int, budget: int, hops: int) -> List[int]:
        if budget <= 0:
            return []
        results: List[int] = []
        seen: Set[int] = {start}
        pq: List[Tuple[int, int, int]] = []  # (-score, node, depth)

        # seed
        self.tt._refresh_topk(start)
        for cnt, succ in self.tt.topk_for(start):
            if succ in seen:
                continue
            seen.add(succ)
            heapq.heappush(pq, (-cnt, succ, 1))
            results.append(succ)
            if len(results) >= budget:
                return results

        while pq and hops > 1 and len(results) < budget:
            negscore, node, depth = heapq.heappop(pq)
            if depth >= hops:
                continue
            self.tt._refresh_topk(node)
            for cnt, succ in self.tt.topk_for(node):
                if succ in seen:
                    continue
                seen.add(succ)
                newscore = min(-negscore, cnt)
                heapq.heappush(pq, (-newscore, succ, depth + 1))
                results.append(succ)
                if len(results) >= budget:
                    break
        return results


# ---------------------- Scheduler -------------------------------------------


from ._shared import Scheduler as SharedScheduler

# Thin adapter to adapt the shared Scheduler API to GraphPrefetcher's expectations.


class Scheduler(SharedScheduler):
    """Adapter that exposes select(...) similar to the original implementation.

    Parameters are mapped: pressure_threshold -> unused (pressure gating is left to caller via EWMA),
    outstanding_limit -> mapped to shared max_outstanding.
    """

    def __init__(self, pressure_threshold: float = 8.0, outstanding_limit: int = 8192):
        # Use shared Scheduler to limit outstanding prefetches.
        super().__init__(max_outstanding=outstanding_limit, mrbsz=64, prefetch_degree=8)

    def select(
        self,
        candidates: List[int],
        ewmas: Dict[Metric, Optional[float]],
        outstanding_count: int,
        degree: int,
    ) -> List[int]:
        # We respect EWMA pressure gating here to keep original semantics.
        pr = ewmas.get(Metric.PRESSURE)
        if pr is not None and pr >= 8.0:
            logger.debug("Scheduler: blocked by pressure %.3f >= 8.0", pr)
            return []
        # Delegate to shared Scheduler.issue which enforces max_outstanding and MRB
        return self.issue(candidates, degree=degree)


# ---------------------- OutstandingTracker ----------------------------------


class OutstandingTracker:
    """
    Tracks outstanding prefetches and classifies hits as late/useful.
    Stores timestamp when recorded to compute lateness.
    """

    def __init__(self, limit: int = 8192, late_time_threshold: float = 0.01):
        self.limit = int(limit)
        self._map: Dict[int, float] = {}  # addr -> ts
        self._queue: Deque[int] = deque()
        self.late_time_threshold = float(late_time_threshold)

    def record(self, addr: int) -> None:
        a = int(addr)
        if a in self._map:
            return
        self._map[a] = time.time()
        self._queue.append(a)
        if len(self._queue) > self.limit:
            old = self._queue.popleft()
            self._map.pop(old, None)

    def credit_if_present(self, addr: int) -> Optional[bool]:
        a = int(addr)
        ts = self._map.pop(a, None)
        if ts is None:
            return None
        try:
            self._queue.remove(a)
        except ValueError:
            pass
        is_late = (time.time() - ts) >= self.late_time_threshold
        return is_late

    def discard(self, addr: int) -> None:
        self._map.pop(int(addr), None)
        try:
            self._queue.remove(int(addr))
        except ValueError:
            pass

    def __len__(self) -> int:
        return len(self._map)

    def clear(self) -> None:
        self._map.clear()
        self._queue.clear()


# ---------------------- Orchestrator: GraphPrefetcherHW ---------------------


@dataclass
class GraphPrefetcher(PrefetchAlgorithm):
    """
    Hardware-like graph prefetcher that wires components:
      AddrFilter -> ObservationQueue -> EwmaCalculator -> AddressGenerator -> Scheduler -> OutstandingTracker
    """

    # tunables
    trigger_on_miss_only: bool = True
    lookahead_hops: int = 2
    prefetch_degree: int = 8
    max_successors_per_node: int = 8
    min_edge_support: int = 2
    aging_half_life_events: Optional[int] = None
    line_size: int = 64

    def __post_init__(self):
        self.filter = AddrFilter()
        self.obsq = ObservationQueue(capacity_per_pc=8)
        self.ewma = EwmaCalculator(alpha=0.25)
        self.transitions = TransitionTable(
            max_successors_per_node=self.max_successors_per_node,
            min_edge_support=self.min_edge_support,
            aging_half_life_events=self.aging_half_life_events,
        )
        self.agg = AddressGenerator(self.transitions)
        self.scheduler = Scheduler(pressure_threshold=8.0, outstanding_limit=8192)
        self.outstanding = OutstandingTracker(limit=8192, late_time_threshold=0.01)

        # last address seen globally (for edge observation)
        self._last_addr: Optional[int] = None

    def init(self) -> None:
        self.filter.clear()
        self.obsq.clear()
        self.ewma.clear()
        self.transitions.clear()
        self.outstanding.clear()
        self._last_addr = None
        logger.info("GraphPrefetcherHW: init done")

    def close(self) -> None:
        logger.info("GraphPrefetcherHW: close")

    def _addr_of(self, access: GraphMemoryAccess) -> int:
        return access.address

    def _should_trigger(self, access, prefetch_hit: bool) -> bool:
        if not self.trigger_on_miss_only:
            return True
        return not bool(prefetch_hit)

    def progress(self, access: GraphMemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Handle one memory access event and possibly produce prefetch addresses.
        """
        if not self.filter.pass_filter(access):
            return []

        addr = access.address
        pc = access.pc

        # Update observation queue and EWMA
        rec = self.obsq.update(access, latency=access.accessLatency)
        if rec.stride is not None:
            self.ewma.update(pc, Metric.STRIDE, rec.stride)
        if rec.inter_arrival is not None:
            self.ewma.update(pc, Metric.INTER_ARRIVAL, rec.inter_arrival)
        if access.accessLatency is not None:
            self.ewma.update(pc, Metric.LATENCY, access.accessLatency)

        # Observe edge from last global address -> current
        if self._last_addr is not None and self._last_addr != addr:
            self.transitions.observe_edge(self._last_addr, addr)
        self._last_addr = addr

        self.transitions.events += (
            0  # keep signature (transitions maintains its own events)
        )

        # Credit outstanding prefetches if this was a prefetch-hit
        if prefetch_hit:
            late = self.outstanding.credit_if_present(addr)
            if late is not None:
                # credit transition or delta if desired; here credit address successor link
                # find head entry for predecessor: we use last observed predecessor heuristics:
                # best-effort: if previous obs exists for this PC, credit that mapping
                hist = self.obsq.history(access, k=2)
                if hist:
                    pred = hist[-1].addr if len(hist) >= 2 else None
                    if pred is not None:
                        self.transitions.observe_edge(pred, addr)
                logger.debug(
                    "GraphPrefetcherHW: prefetch_hit credited addr=0x%X late=%s",
                    addr,
                    str(late),
                )

        # Generate candidate successors using AddressGenerator / PPU
        candidates = self.agg.select_successors(
            start=addr, budget=self.prefetch_degree * 2, hops=self.lookahead_hops
        )
        # get ewma params for scheduler gating
        ewma_params = self.ewma.get_parameters(pc)
        # ask scheduler which to accept (respecting outstanding)
        accepted = self.scheduler.select(
            candidates, ewma_params, len(self.outstanding), degree=self.prefetch_degree
        )

        # record outstanding and return
        issued: List[int] = []
        for tgt in accepted:
            # align to line
            aligned = (tgt // self.line_size) * self.line_size
            if aligned not in self.outstanding._map and aligned != addr:
                self.outstanding.record(aligned)
                issued.append(aligned)
                if len(issued) >= self.prefetch_degree:
                    break

        if issued:
            logger.debug(
                "GraphPrefetcherHW: issued prefetches %s for trigger 0x%X",
                [hex(x) for x in issued],
                addr,
            )
        return issued

    def prefetch_completed(self, addr: int) -> None:
        self.outstanding.discard(addr)

    def notify_prefetch_hit(self, addr: int) -> None:
        self.outstanding.discard(addr)
