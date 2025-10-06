"""
Domino: Temporal prefetcher implementation (defaults used where paper values are absent).

Place this file in: prefetchlenz/prefetchingalgorithm/impl/domino.py

Implements:
 - MHT1: 1-miss history table (maps last miss addr -> predicted next miss)
 - MHT2: 2-miss history table (maps last-two-misses signature -> predicted next miss)
 - MRB: Miss Resolution / suppression buffer to avoid immediate re-issue
 - Scheduler: outstanding prefetch cap + MRB suppression
 - DominoPrefetcher: top-level class implementing PrefetchAlgorithm interface

Notes:
 - This is a temporal prefetcher: it predicts addresses based on recent miss sequences.
 - Configurable parameters are in CONFIG at the top.
 - Storage is implemented using pure-Python dicts for determinism and easy testing.
   If you must use your project's Cache storage backend, see the README's "Integration"
   section — a short adapter is required and the code documents where to plug it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Import MemoryAccess from repo types. The repo must provide this dataclass.
# Example shape required: MemoryAccess(address: int, pc: int)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess  # user-supplied

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.impl.domino")

# -------------------------
# Configuration variables
# -------------------------
CONFIG = {
    # Table sizes (defaults chosen conservatively)
    "MHT1_SIZE": 1024,  # entries for 1-miss history table
    "MHT2_SIZE": 4096,  # entries for 2-miss history table
    "MHT_TAG_BITS": 20,  # tag bits for distinguishing addresses (for dictionary keys)
    "MRB_SIZE": 64,  # suppression buffer size (recent prefetches)
    "PREFETCH_DEGREE": 2,  # how many temporal predictions to issue per trigger
    "COUNTER_BITS": 2,  # saturating counter bits for confirmation / confidence
    "MAX_OUTSTANDING": 32,  # cap outstanding prefetches issued by scheduler
    "DEGREE_MHT1": 1,  # number of predictions from MHT1 per hit
    "DEGREE_MHT2": 2,  # number of predictions from MHT2 per hit
    # Aging / decay parameters (conservative defaults)
    "AGE_DECAY_INTERVAL": 10000,  # not used in tests, placeholder for real sim
}

# Derived constants
COUNTER_MAX = (1 << CONFIG["COUNTER_BITS"]) - 1
PREFETCH_DEGREE = CONFIG["PREFETCH_DEGREE"]

# -------------------------
# Small dataclasses
# -------------------------


@dataclass
class MHT1Entry:
    """Entry for MHT1: stores a predicted address and a confidence counter."""

    predicted: int
    conf: int  # saturating counter 0..COUNTER_MAX


@dataclass
class MHT2Entry:
    """Entry for MHT2: stores up to N predicted addresses (list) and a confidence counter."""

    predicted: List[int]  # ordered list of predicted addresses (up to DEGREE_MHT2)
    conf: int


# -------------------------
# Helper functions
# -------------------------
def trunc_tag(addr: int, bits: int = CONFIG["MHT_TAG_BITS"]) -> int:
    """Return a truncated tag of an address for use in keys if desired."""
    mask = (1 << bits) - 1
    return addr & mask


def key_two_misses(m1: int, m2: int) -> Tuple[int, int]:
    """Key function for MHT2 mapping from (prev2, prev1) -> predicted list."""
    # Use truncated tags to limit key size
    return (trunc_tag(m1), trunc_tag(m2))


# -------------------------
# Components
# -------------------------


class MHT1:
    """1-miss history table.

    Maps last_miss_addr (truncated) -> MHT1Entry.
    Implements saturating counters on updates.
    """

    def __init__(self, size: int = CONFIG["MHT1_SIZE"]):
        self.size = size
        self.table: Dict[int, MHT1Entry] = {}
        self.order: List[int] = []  # simple FIFO eviction for deterministic behavior

    def get(self, last_miss: int) -> Optional[MHT1Entry]:
        k = trunc_tag(last_miss)
        return self.table.get(k)

    def update(self, last_miss: int, next_miss: int, strengthen: bool = True) -> None:
        k = trunc_tag(last_miss)
        ent = self.table.get(k)
        if ent:
            # if predicted matches, strengthen; else overwrite and reset
            if ent.predicted == next_miss:
                if strengthen and ent.conf < COUNTER_MAX:
                    ent.conf += 1
                    logger.debug("MHT1: strengthen conf for key %x -> %d", k, ent.conf)
            else:
                # overwrite with new prediction, conservative reset conf
                ent.predicted = next_miss
                ent.conf = min(1, COUNTER_MAX)
                logger.debug(
                    "MHT1: overwrite key %x with new pred %x conf %d",
                    k,
                    next_miss,
                    ent.conf,
                )
        else:
            # insert, evict if needed
            if len(self.table) >= self.size:
                victim = self.order.pop(0)
                del self.table[victim]
                logger.debug("MHT1: evicted key %x", victim)
            self.table[k] = MHT1Entry(predicted=next_miss, conf=1)
            self.order.append(k)
            logger.debug("MHT1: inserted key %x -> pred %x", k, next_miss)

    def decay_all(self) -> None:
        """Optional aging: decrease confidence counters (not used in tests)."""
        for ent in self.table.values():
            if ent.conf > 0:
                ent.conf -= 1


class MHT2:
    """2-miss history table.

    Maps (prev2, prev1) -> MHT2Entry containing an ordered list of predictions.
    """

    def __init__(
        self, size: int = CONFIG["MHT2_SIZE"], degree: int = CONFIG["DEGREE_MHT2"]
    ):
        self.size = size
        self.degree = degree
        self.table: Dict[Tuple[int, int], MHT2Entry] = {}
        self.order: List[Tuple[int, int]] = []

    def get(self, prev2: int, prev1: int) -> Optional[MHT2Entry]:
        k = key_two_misses(prev2, prev1)
        return self.table.get(k)

    def update(
        self, prev2: int, prev1: int, next_miss: int, strengthen: bool = True
    ) -> None:
        k = key_two_misses(prev2, prev1)
        ent = self.table.get(k)
        if ent:
            # if next_miss is already in predicted list, optionally strengthen via rotating it forward
            if next_miss in ent.predicted:
                if strengthen and ent.conf < COUNTER_MAX:
                    ent.conf += 1
                    logger.debug("MHT2: strengthen conf for key %s -> %d", k, ent.conf)
                # optionally move to front to indicate recency
                ent.predicted.remove(next_miss)
                ent.predicted.insert(0, next_miss)
            else:
                # insert at front
                ent.predicted.insert(0, next_miss)
                # trim
                ent.predicted = ent.predicted[: self.degree]
                ent.conf = min(1, COUNTER_MAX)
                logger.debug("MHT2: updated key %s with new pred %x", k, next_miss)
        else:
            # insert (evict if needed)
            if len(self.table) >= self.size:
                victim = self.order.pop(0)
                del self.table[victim]
                logger.debug("MHT2: evicted key %s", victim)
            self.table[k] = MHT2Entry(predicted=[next_miss], conf=1)
            self.order.append(k)
            logger.debug("MHT2: inserted key %s -> pred %x", k, next_miss)

    def decay_all(self) -> None:
        for ent in self.table.values():
            if ent.conf > 0:
                ent.conf -= 1


class MRB:
    """Miss Resolution Buffer / short-term suppression.

    Keeps a small recent set of addresses (capped) to avoid re-issuing them immediately.
    Implemented as an ordered list for deterministic eviction.
    """

    def __init__(self, size: int = CONFIG["MRB_SIZE"]):
        self.size = size
        self.buf: List[int] = []

    def insert(self, addr: int) -> None:
        if addr in self.buf:
            # move to back (most recent)
            self.buf.remove(addr)
            self.buf.append(addr)
            return
        if len(self.buf) >= self.size:
            ev = self.buf.pop(0)
            logger.debug("MRB: evict %x", ev)
        self.buf.append(addr)
        logger.debug("MRB: insert %x", addr)

    def contains(self, addr: int) -> bool:
        return addr in self.buf

    def remove(self, addr: int) -> None:
        if addr in self.buf:
            self.buf.remove(addr)

    def clear(self) -> None:
        self.buf.clear()


class Scheduler:
    """Scheduler managing outstanding prefetches, MRB suppression, and issuance.

    - outstanding: set of addresses currently outstanding
    - issue(candidates) -> list of addresses actually issued (subject to caps and MRB)
    - credit(addr): called when a prefetched address is used, to remove from outstanding
    """

    def __init__(
        self,
        max_outstanding: int = CONFIG["MAX_OUTSTANDING"],
        mrbsz: int = CONFIG["MRB_SIZE"],
    ):
        self.max_outstanding = max_outstanding
        self.outstanding: List[int] = []  # maintain order for deterministic behavior
        self.mrb = MRB(size=mrbsz)

    def can_issue_more(self) -> bool:
        return len(self.outstanding) < self.max_outstanding

    def issue(self, candidates: List[int]) -> List[int]:
        """Issue prefetches from the candidate list. Respect MRB, outstanding cap, and avoid duplicates."""
        issued = []
        for a in candidates:
            if len(issued) >= PREFETCH_DEGREE:
                break
            if a in self.outstanding:
                continue
            if self.mrb.contains(a):
                continue
            if not self.can_issue_more():
                break
            # issue
            self.outstanding.append(a)
            issued.append(a)
            # also insert into MRB to prevent immediate reissue on same cycle
            self.mrb.insert(a)
            logger.info("Scheduler: issue prefetch %x", a)
        return issued

    def credit(self, addr: int) -> None:
        """Called when a prefetched address is used; remove from outstanding and keep in MRB to suppress re-issue."""
        if addr in self.outstanding:
            self.outstanding.remove(addr)
            logger.debug("Scheduler: credit outstanding for %x", addr)
        # keep addr in MRB so it won't be re-issued immediately (MRB insertion will be idempotent)
        self.mrb.insert(addr)

    def remove_from_mrb(self, addr: int) -> None:
        self.mrb.remove(addr)

    def clear(self) -> None:
        self.outstanding.clear()
        self.mrb.clear()


# -------------------------
# DominoPrefetcher: top-level
# -------------------------


class DominoPrefetcher:
    """Temporal prefetcher implementing Domino-like behavior.

    Public methods expected by the framework:
      - init()
      - progress(access: MemoryAccess, prefetch_hit: bool) -> List[int]
      - close()
    """

    def __init__(self, config: Optional[Dict] = None):
        # merge user config if provided
        if config:
            for k, v in config.items():
                if k in CONFIG:
                    CONFIG[k] = v
        # create components
        self.mht1 = MHT1(size=CONFIG["MHT1_SIZE"])
        self.mht2 = MHT2(size=CONFIG["MHT2_SIZE"], degree=CONFIG["DEGREE_MHT2"])
        self.scheduler = Scheduler(
            max_outstanding=CONFIG["MAX_OUTSTANDING"], mrbsz=CONFIG["MRB_SIZE"]
        )
        # track recent misses: prev1 is-most-recent miss, prev2 is previous to that
        self.prev1: Optional[int] = None
        self.prev2: Optional[int] = None
        self.initialized = False

    # -------------------------
    # PrefetchAlgorithm API
    # -------------------------
    def init(self) -> None:
        logger.info("DominoPrefetcher.init")
        self.initialized = True

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Handle an access event:
         - If prefetch_hit: credit scheduler with that address (feedback)
         - If this access is a miss (the framework will indicate misses by calling with prefetch_hit maybe False
           but we need an explicit 'is_miss' signal to be fully faithful. Here we treat each access as potential miss
           trigger when progress is invoked — tests will call it accordingly.)
         - Use prev1/prev2 history to consult MHT2 and MHT1 to produce candidate prefetch addresses.
         - Issue via scheduler (which suppresses via MRB and caps outstanding).
        Returns list of addresses actually issued as prefetches.
        """
        if not self.initialized:
            self.init()

        addr = access.address
        issued: List[int] = []

        # If this access is the result of a prefetch (prefetch_hit True), credit scheduler
        if prefetch_hit:
            logger.debug("Domino: prefetch_hit for %x", addr)
            self.scheduler.credit(addr)
            # update MHTs: a successful prefetch confirms the last mapping (strengthen)
            if self.prev1 is not None:
                # strengthen MHT1 mapping prev1 -> addr
                self.mht1.update(self.prev1, addr, strengthen=True)
            if self.prev2 is not None and self.prev1 is not None:
                self.mht2.update(self.prev2, self.prev1, addr, strengthen=True)
            # We do not produce new prefetches on a prefetch_hit event beyond crediting.
            return []

        # Otherwise, assume this is a demand access; in Domino the miss stream is the trigger.
        # For unit tests we will treat calls to progress(...) as miss-triggered training events.

        # 1) Query MHT2 using (prev2, prev1) if available
        candidates: List[int] = []
        if self.prev2 is not None and self.prev1 is not None:
            ent2 = self.mht2.get(self.prev2, self.prev1)
            if ent2 and ent2.predicted:
                # Use predictions in order, but respect conf threshold (we use conf >=1 conservatively)
                for p in ent2.predicted:
                    if len(candidates) >= CONFIG["DEGREE_MHT2"]:
                        break
                    if ent2.conf > 0:
                        candidates.append(p)
                logger.debug(
                    "Domino: MHT2 lookup produced %d candidates", len(candidates)
                )

        # 2) Query MHT1 using prev1 (if MHT2 did not yield enough)
        if len(candidates) < CONFIG["DEGREE_MHT2"] and self.prev1 is not None:
            ent1 = self.mht1.get(self.prev1)
            if ent1 and ent1.conf > 0:
                candidates.append(ent1.predicted)
                logger.debug("Domino: MHT1 lookup added pred %x", ent1.predicted)

        # Limit total candidates and remove duplicates (preserve order)
        seen = set()
        filtered = []
        for c in candidates:
            if c not in seen:
                filtered.append(c)
                seen.add(c)
        candidates = filtered[:PREFETCH_DEGREE]

        # 3) Ask scheduler to issue them (scheduler does MRB and outstanding checks)
        issued = self.scheduler.issue(candidates)

        # 4) Update MHTs with the observed transition: prev1 -> addr and (prev2, prev1) -> addr
        if self.prev1 is not None:
            self.mht1.update(self.prev1, addr, strengthen=False)
        if self.prev2 is not None and self.prev1 is not None:
            self.mht2.update(self.prev2, self.prev1, addr, strengthen=False)

        # 5) Advance history: shift prevs
        self.prev2 = self.prev1
        self.prev1 = addr

        return issued

    def close(self) -> None:
        logger.info("DominoPrefetcher.close")
        # reset
        self.mht1 = MHT1(size=CONFIG["MHT1_SIZE"])
        self.mht2 = MHT2(size=CONFIG["MHT2_SIZE"], degree=CONFIG["DEGREE_MHT2"])
        self.scheduler = Scheduler(
            max_outstanding=CONFIG["MAX_OUTSTANDING"], mrbsz=CONFIG["MRB_SIZE"]
        )
        self.prev1 = None
        self.prev2 = None
        self.initialized = False
