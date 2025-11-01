"""
Spatial Memory Streaming (SMS) by Somogyi et al.

This file implements:
  - AGT (filter + accumulation) with deterministic LRU eviction
  - PHT (set-associative) with per-block 2-bit saturating counters
  - Prediction registers and a simple Scheduler for issuing prefetches
  - SMSPrefetcher class implementing PrefetchAlgorithm

Configuration:
  All tunable numeric values are at the top in the CONFIG dictionary. Change them
  to match paper parameters or experiment locally.

Mapping notes:
  - AGT filter/accum emulate CAMs (Section 4.1)
  - PHT entries hold 2-bit counters; update rules saturate at COUNTER_MAX
  - Keying uses an XOR of region_base and PC mask as a rotation/index approximation
  - Because the provided Cache lacks eviction callbacks, generation closing is
    driven by an accumulation-size heuristic or explicit close() usage.
"""

import logging
from collections import deque
from typing import Deque, Dict, List, Optional

from prefetchlenz.prefetchingalgorithm.impl._shared import (
    align_to_region,
    get_region_offset,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.impl.sms")

# -------------------------
# Configuration variables
# -------------------------
CONFIG = {
    # Region / block sizing
    "REGION_SIZE_BYTES": 2 * 1024,  # 2KB regions (paper default)
    "BLOCK_SIZE_BYTES": 64,  # 64B cache blocks (common)
    # AGT sizes (filter CAM entries, accumulation CAM entries)
    "AGT_FILTER_ENTRIES": 32,
    "AGT_ACCUM_ENTRIES": 64,
    # PHT sizing (number of sets, associativity)
    "PHT_NUM_SETS": 256,  # set count; increase for larger experiments
    "PHT_ASSOC": 4,  # associativity
    # Prefetch behavior / scheduler
    "PREFETCH_DEGREE": 4,
    "MAX_OUTSTANDING_PREFETCH": 16,
    # Counter / hysteresis (2-bit counters)
    "COUNTER_BITS": 2,
    "INIT_OBSERVED": 2,  # initial counter for observed blocks
    "INIT_UNOBSERVED": 1,  # initial counter for unobserved blocks
    # Threshold for considering a counter "predict"
    "PREDICTION_THRESHOLD": 2,
    # Heuristic: commit accumulation when seen_blocks >= fraction of region
    "ACCUM_COMMIT_MIN_BLOCKS": 2,  # minimum
    "ACCUM_COMMIT_FRACTION": 0.125,  # fraction of blocks in region to trigger commit
    # Keying masks for rotating/indexing approximations
    "PC_KEY_MASK": 0xFFF,
    "TRIGGER_KEY_MASK": 0xFF,
}

# Derived constants
REGION_SIZE = CONFIG["REGION_SIZE_BYTES"]
BLOCK_SIZE = CONFIG["BLOCK_SIZE_BYTES"]
BLOCKS_PER_REGION = REGION_SIZE // BLOCK_SIZE
AGT_FILTER_ENTRIES = CONFIG["AGT_FILTER_ENTRIES"]
AGT_ACCUM_ENTRIES = CONFIG["AGT_ACCUM_ENTRIES"]
PHT_NUM_SETS = CONFIG["PHT_NUM_SETS"]
PHT_ASSOC = CONFIG["PHT_ASSOC"]
PREFETCH_DEGREE = CONFIG["PREFETCH_DEGREE"]
MAX_OUTSTANDING_PREFETCH = CONFIG["MAX_OUTSTANDING_PREFETCH"]
COUNTER_MAX = (1 << CONFIG["COUNTER_BITS"]) - 1
INIT_OBSERVED = CONFIG["INIT_OBSERVED"]
INIT_UNOBSERVED = CONFIG["INIT_UNOBSERVED"]
PREDICTION_THRESHOLD = CONFIG["PREDICTION_THRESHOLD"]
ACCUM_COMMIT_MIN_BLOCKS = CONFIG["ACCUM_COMMIT_MIN_BLOCKS"]
ACCUM_COMMIT_FRACTION = CONFIG["ACCUM_COMMIT_FRACTION"]
PC_KEY_MASK = CONFIG["PC_KEY_MASK"]
TRIGGER_KEY_MASK = CONFIG["TRIGGER_KEY_MASK"]


# -------------------------
# Utility helpers
# -------------------------


def region_base(addr: int) -> int:
    """Return base address of the region containing addr."""
    return align_to_region(addr, REGION_SIZE)


def block_index_in_region(addr: int) -> int:
    """Return block index (0..BLOCKS_PER_REGION-1) within the region for addr."""
    return get_region_offset(addr, REGION_SIZE, BLOCK_SIZE)


# -------------------------
# AGT: Active Generation Table
# -------------------------
class AGTFilterEntry:
    """Represents a small 'trigger-only' filter entry (paper: filter CAM)."""

    __slots__ = ("region", "trigger_block", "seen")

    def __init__(self, region: int, trigger_block: int):
        self.region = region
        self.trigger_block = trigger_block
        self.seen = False


class AGTAccumEntry:
    """Represents an accumulation entry recording observed blocks in a generation."""

    __slots__ = ("region", "trigger_block", "bitvec", "closed")

    def __init__(self, region: int, trigger_block: int):
        self.region = region
        self.trigger_block = trigger_block
        # bitvec stores 0/1 per block indicating observed accesses during generation
        self.bitvec: List[int] = [0] * BLOCKS_PER_REGION
        self.closed = False

    def observe(self, block_idx: int) -> None:
        """Mark a block as observed in this accumulation."""
        if 0 <= block_idx < BLOCKS_PER_REGION:
            self.bitvec[block_idx] = 1


class AGT:
    """Active Generation Table emulation.

    Emulates the two-CAM design (filter + accumulation) described in the paper.
    Uses deterministic FIFO/LRU-style eviction via deques to keep tests repeatable.

    Public methods:
      - trigger(region, trigger_block) -> AGTFilterEntry | AGTAccumEntry
      - observe(addr) -> AGTAccumEntry | AGTFilterEntry | None
      - promote_to_accum(region) -> AGTAccumEntry
      - close_generation(region) -> AGTAccumEntry | None
    """

    def __init__(
        self,
        filter_entries: int = AGT_FILTER_ENTRIES,
        accum_entries: int = AGT_ACCUM_ENTRIES,
    ):
        self.filter_cap = filter_entries
        self.accum_cap = accum_entries
        self.filter: Dict[int, AGTFilterEntry] = {}
        self.filter_order: Deque[int] = deque()
        self.accum: Dict[int, AGTAccumEntry] = {}
        self.accum_order: Deque[int] = deque()

    def trigger(self, region: int, trigger_block: int):
        """Called when a first (trigger) access is seen for a region.

        Returns an existing accumulation (if active), a filter entry, or allocates a new filter entry.
        """
        if region in self.accum:
            return self.accum[region]
        if region in self.filter:
            return self.filter[region]
        # allocate a filter entry, evict oldest if necessary
        if len(self.filter) >= self.filter_cap:
            victim = self.filter_order.popleft()
            del self.filter[victim]
            logger.debug("AGT.filter evict region=%x", victim)
        e = AGTFilterEntry(region, trigger_block)
        self.filter[region] = e
        self.filter_order.append(region)
        logger.debug("AGT.filter insert region=%x trigger=%d", region, trigger_block)
        return e

    def promote_to_accum(self, region: int) -> Optional[AGTAccumEntry]:
        """Promote a filter entry into an accumulation entry; used when additional blocks seen."""
        if region in self.accum:
            return self.accum[region]
        f = self.filter.pop(region, None)
        if f is None:
            return None
        try:
            self.filter_order.remove(region)
        except ValueError:
            pass
        # evict oldest accumulation if needed
        if len(self.accum) >= self.accum_cap:
            victim = self.accum_order.popleft()
            del self.accum[victim]
            logger.debug("AGT.accum evict region=%x", victim)
        a = AGTAccumEntry(region, f.trigger_block)
        self.accum[region] = a
        self.accum_order.append(region)
        logger.debug("AGT.promote region=%x", region)
        return a

    def observe(self, addr: int):
        """Record an observed block access either in accumulation or promote filter entry."""
        r = region_base(addr)
        b = block_index_in_region(addr)
        if r in self.accum:
            a = self.accum[r]
            a.observe(b)
            try:
                self.accum_order.remove(r)
            except ValueError:
                pass
            self.accum_order.append(r)
            logger.debug("AGT.observe accumulation region=%x block=%d", r, b)
            return a
        if r in self.filter:
            f = self.filter[r]
            logger.debug(
                "AGT.observe filter region=%x block=%d (trigger=%d)",
                r,
                b,
                f.trigger_block,
            )
            if f.trigger_block != b:
                a = self.promote_to_accum(r)
                if a:
                    a.observe(b)
                    return a
            return f
        return None

    def close_generation(self, region: int) -> Optional[AGTAccumEntry]:
        """Close (commit) an accumulation for the given region and return it.

        If the region had only a filter entry, remove it (no accumulation formed).
        """
        a = self.accum.pop(region, None)
        if a:
            try:
                self.accum_order.remove(region)
            except ValueError:
                pass
            a.closed = True
            logger.debug("AGT.close_generation region=%x", region)
            return a
        if region in self.filter:
            try:
                self.filter_order.remove(region)
            except ValueError:
                pass
            del self.filter[region]
            logger.debug("AGT.close_filter region=%x", region)
        return None


# -------------------------
# PHT: Pattern History Table
# -------------------------
class PHTEntry:
    """Entry in PHT storing per-block saturating counters and metadata."""

    __slots__ = ("region", "trigger_block", "counters", "age")

    def __init__(self, region: int, trigger_block: int, bitvec: List[int]):
        self.region = region
        self.trigger_block = trigger_block
        # initialize counters: observed -> higher initial value, else lower
        self.counters: List[int] = [
            INIT_OBSERVED if bit else INIT_UNOBSERVED for bit in bitvec
        ]
        self.age = 0

    def update_from_generation(self, bitvec: List[int]) -> None:
        """Update counters using saturating increment/decrement as described in paper.

        Observed blocks increment (up to COUNTER_MAX). Non-observed blocks decay by one.
        """
        for i, bit in enumerate(bitvec):
            if bit:
                if self.counters[i] < COUNTER_MAX:
                    self.counters[i] += 1
            else:
                if self.counters[i] > 0:
                    self.counters[i] -= 1


class SetAssocPHT:
    """Set-associative PHT implementation with per-set LRU replacement.

    Public API:
      - get(region) -> PHTEntry | None
      - insert_or_update(region, trigger_block, bitvec) -> PHTEntry
    """

    def __init__(self, num_sets: int = PHT_NUM_SETS, associativity: int = PHT_ASSOC):
        self.num_sets = num_sets
        self.assoc = associativity
        # each set is a dict: region -> PHTEntry
        self.sets: List[Dict[int, PHTEntry]] = [dict() for _ in range(num_sets)]
        # per-set order deque used for deterministic LRU
        self.orders: List[Deque[int]] = [deque() for _ in range(num_sets)]

    def _set_idx(self, region: int) -> int:
        # uses region index / division then modulo by num_sets (simple mapping)
        return (region // REGION_SIZE) % self.num_sets

    def get(self, region: int) -> Optional[PHTEntry]:
        idx = self._set_idx(region)
        s = self.sets[idx]
        if region in s:
            try:
                self.orders[idx].remove(region)
            except ValueError:
                pass
            self.orders[idx].append(region)
            return s[region]
        return None

    def insert_or_update(
        self, region: int, trigger_block: int, bitvec: List[int]
    ) -> PHTEntry:
        idx = self._set_idx(region)
        s = self.sets[idx]
        order = self.orders[idx]
        if region in s:
            e = s[region]
            e.update_from_generation(bitvec)
            try:
                order.remove(region)
            except ValueError:
                pass
            order.append(region)
            logger.debug("PHT.update region=%x set=%d", region, idx)
            return e
        # insertion
        if len(s) >= self.assoc:
            victim = order.popleft()
            del s[victim]
            logger.debug("PHT.evict region=%x from set=%d", victim, idx)
        e = PHTEntry(region, trigger_block, bitvec)
        s[region] = e
        order.append(region)
        logger.debug("PHT.insert region=%x set=%d", region, idx)
        return e


# -------------------------
# Prediction Register & Scheduler
# -------------------------
class PredictionRegister:
    """Small FIFO of block indices to stream for a prediction."""

    __slots__ = ("region", "to_stream", "trigger_block")

    def __init__(self, region: int, bitvec: List[int], trigger_block: int):
        self.region = region
        # To preserve ordering deterministically, iterate block indices ascending
        self.to_stream: Deque[int] = deque(i for i, b in enumerate(bitvec) if b)
        self.trigger_block = trigger_block

    def next(self) -> Optional[int]:
        if not self.to_stream:
            return None
        return self.to_stream.popleft()


from ._shared import Scheduler as SharedScheduler


class Scheduler(SharedScheduler):
    """Adapter mapping SMS simple counters to the shared Scheduler.

    SMS used a simple outstanding counter. Map max_outstanding -> shared max_outstanding
    and use issue/credit APIs to track credits. The shared MRB will also be used.
    """

    def __init__(self, max_outstanding: int = MAX_OUTSTANDING_PREFETCH):
        super().__init__(
            max_outstanding=max_outstanding, mrbsz=64, prefetch_degree=PREFETCH_DEGREE
        )

    def can_issue(self) -> bool:
        return super().can_issue()


# -------------------------
# SMS Prefetcher Implementation
# -------------------------
class SMSPrefetcher(PrefetchAlgorithm):
    """SMS-like spatial prefetcher mapped into the provided framework.

    Methods:
      - init() : initialize internal structures
      - progress(access, prefetch_hit) -> List[int] : handle an access and return prefetch addresses
      - close() : reset internal state

    Notes & approximations (documented in README + docstrings):
      - AGT relies on either explicit close_generation() (if caller triggers it) or an
        accumulation-size heuristic to commit patterns to the PHT. If your Cache exposes
        eviction callbacks, hook them to call agt.close_generation(region) for faithful behavior.
    """

    def __init__(self, cache=None, config: Dict[str, int] = None):
        # allow overrides via config param
        self.cache = cache
        if config:
            # merge user-supplied config into global CONFIG-derived values
            for k, v in config.items():
                if k in CONFIG:
                    CONFIG[k] = v
        # re-bind derived constants (modules using these should reference globals for runtime changes)
        self.agt = AGT(
            filter_entries=CONFIG["AGT_FILTER_ENTRIES"],
            accum_entries=CONFIG["AGT_ACCUM_ENTRIES"],
        )
        self.pht = SetAssocPHT(
            num_sets=CONFIG["PHT_NUM_SETS"], associativity=CONFIG["PHT_ASSOC"]
        )
        self.scheduler = Scheduler(max_outstanding=CONFIG["MAX_OUTSTANDING_PREFETCH"])
        self.pred_regs: Deque[PredictionRegister] = deque()
        self.initialized = False

    def init(self) -> None:
        logger.info("SMSPrefetcher.init")
        self.initialized = True

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """Handle one memory access event and return a list of prefetch addresses (absolute).

        Behavior implemented:
          - If prefetch_hit: credit scheduler and strengthen PHT counters for that block
          - On access: observe via AGT (trigger / filter / accumulation logic)
          - If a new filter-trigger occurs and PHT hit exists for the keyed region, create a
            PredictionRegister and begin issuing up to PREFETCH_DEGREE prefetches, subject to scheduler.
          - When an accumulation meets the commit heuristic, commit to PHT via commit_generation()
        """
        if not self.initialized:
            self.init()

        issued_addrs: List[int] = []
        addr = access.address
        pc = access.pc
        rbase = region_base(addr)
        bidx = block_index_in_region(addr)

        # Handle prefetch hit crediting: strengthen counter for that block if present in PHT
        if prefetch_hit:
            logger.debug("prefetch hit addr=%x", addr)
            # Use shared Scheduler API: credit the specific address so it is removed from outstanding
            try:
                self.scheduler.credit(addr)
            except Exception:
                # Fallback for adapters that ignore the addr parameter
                try:
                    self.scheduler.credit()
                except Exception:
                    pass
            p = self.pht.get(rbase)
            if p:
                if p.counters[bidx] < COUNTER_MAX:
                    p.counters[bidx] += 1
                    logger.debug(
                        "PHT.counter++ region=%x block=%d now=%d",
                        rbase,
                        bidx,
                        p.counters[bidx],
                    )
            return []

        # 1) Trigger/filter handling
        agt_entry = self.agt.trigger(rbase, bidx)

        # If the returned entry is a filter entry (fresh trigger), consult PHT for predictions.
        # Paper maps keys using PC+rotation; we approximate with XOR region ^ (pc & mask)
        if type(agt_entry).__name__ == "AGTFilterEntry":
            key = rbase ^ (pc & CONFIG["PC_KEY_MASK"])
            pht_entry = self.pht.get(key)
            if pht_entry:
                # convert counters -> bitvec via prediction threshold
                bitvec = [
                    1 if c >= CONFIG["PREDICTION_THRESHOLD"] else 0
                    for c in pht_entry.counters
                ]
                pr = PredictionRegister(
                    pht_entry.region, bitvec, pht_entry.trigger_block
                )
                self.pred_regs.append(pr)
                logger.info("PHT hit -> start streaming region=%x", pht_entry.region)

        # 2) Record observation (may promote filter->accum and record new bits)
        a = self.agt.observe(addr)

        # 3) Commit accumulation heuristics:
        if isinstance(a, AGTAccumEntry):
            seen_blocks = sum(a.bitvec)
            threshold_blocks = max(
                ACCUM_COMMIT_MIN_BLOCKS, int(BLOCKS_PER_REGION * ACCUM_COMMIT_FRACTION)
            )
            if a.closed or seen_blocks >= threshold_blocks:
                # commit to PHT
                self.commit_generation(a)
                # close AGT entry
                self.agt.close_generation(a.region)

        # 4) Issue prefetches from prediction registers, subject to scheduler and PREFETCH_DEGREE
        while (
            self.pred_regs
            and self.scheduler.can_issue()
            and len(issued_addrs) < CONFIG["PREFETCH_DEGREE"]
        ):
            pr = self.pred_regs[0]
            next_block_idx = pr.next()
            if next_block_idx is None:
                self.pred_regs.popleft()
                continue
            paddr = pr.region + next_block_idx * BLOCK_SIZE
            # Ask shared scheduler whether we can issue this address (degree=1)
            issued = self.scheduler.issue([paddr], degree=1)
            if issued:
                issued_addrs.append(paddr)
                logger.info("issue prefetch addr=%x", paddr)

        return issued_addrs

    def commit_generation(self, accum: AGTAccumEntry) -> None:
        """Commit an accumulation to the PHT.

        The paper uses rotation/PC-based indexing to improve generalization. We approximate that
        by XORing the region base with a small mask of the trigger_block (configurable).
        """
        key = accum.region ^ (accum.trigger_block & CONFIG["TRIGGER_KEY_MASK"])
        self.pht.insert_or_update(key, accum.trigger_block, accum.bitvec)
        logger.debug("commit_generation region=%x -> PHT key=%x", accum.region, key)

    def close(self) -> None:
        """Reset internal state."""
        logger.info("SMSPrefetcher.close")
        self.initialized = False
        # keep state structures intact but clear prediction registers and scheduler
        self.pred_regs.clear()
        self.scheduler.clear()
