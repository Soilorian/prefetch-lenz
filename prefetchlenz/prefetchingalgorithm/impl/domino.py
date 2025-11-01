"""
Domino Temporal Prefetcher

Algorithm: Domino prefetcher implementation (defaults used where paper values are absent).

This is a temporal prefetcher that predicts future memory addresses based on recent
miss sequences. It maintains two miss history tables (MHT1 and MHT2) that track
patterns in miss address sequences and use them to predict subsequent misses.

Key Components:
- MHT1: 1-miss history table mapping last miss address to predicted next miss
- MHT2: 2-miss history table mapping (prev2, prev1) miss signature to predicted next miss
- MRB: Miss Resolution Buffer for suppressing immediate re-issue of prefetched addresses
- Scheduler: Caps outstanding prefetches and manages MRB suppression
- DominoPrefetcher: Top-level prefetcher coordinating all components

How it works:
1. Track recent miss sequence (prev2, prev1, current)
2. Query MHT2 first for predictions based on (prev2, prev1) pattern
3. If MHT2 doesn't yield enough, query MHT1 for predictions based on prev1
4. Update MHT tables with observed transitions, strengthening on prefetch hits
5. Use scheduler to issue prefetches while respecting outstanding limits and MRB
6. Confidence counters track prediction quality and decay over time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from prefetchlenz.prefetchingalgorithm.impl._shared import MRB, Scheduler
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger(__name__)

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


# -------------------------
# DominoPrefetcher: top-level
# -------------------------


class DominoPrefetcher(PrefetchAlgorithm):
    """Temporal prefetcher implementing Domino-like behavior.

    Uses MHT1 and MHT2 to track miss history and generate temporal predictions.
    """

    def __init__(self, config: Optional[Dict] = None):
        if config:
            for k, v in config.items():
                if k in CONFIG:
                    CONFIG[k] = v

        self.mht1 = MHT1(size=CONFIG["MHT1_SIZE"])
        self.mht2 = MHT2(size=CONFIG["MHT2_SIZE"], degree=CONFIG["DEGREE_MHT2"])
        self.scheduler = Scheduler(
            max_outstanding=CONFIG["MAX_OUTSTANDING"],
            mrbsz=CONFIG["MRB_SIZE"],
        )
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
        """Handle an access event.

        If prefetch_hit is True, credit scheduler with that address (feedback).
        If this access is a miss, use prev1/prev2 history to consult MHT2 and MHT1
        to produce candidate prefetch addresses. Issue via scheduler which suppresses
        via MRB and caps outstanding prefetches.

        Note: The framework indicates misses by calling with prefetch_hit=False.
        We treat each access as a potential miss trigger when progress is invoked.
        Tests will call it accordingly.

        Args:
            access: The current memory access.
            prefetch_hit: Whether this access was a prefetch hit.

        Returns:
            List of addresses actually issued as prefetches.
        """
        if not self.initialized:
            self.init()

        current_address = access.address

        if prefetch_hit:
            self._handle_prefetch_hit(current_address)
            return []

        candidates = self._generate_prefetch_candidates()
        issued = self.scheduler.issue(candidates)
        self._update_history_and_tables(current_address)
        return issued

    def _handle_prefetch_hit(self, address: int) -> None:
        """Handle prefetch hit by crediting scheduler and strengthening MHTs."""
        logger.debug("Domino: prefetch_hit for %x", address)
        self.scheduler.credit(address)
        if self.prev1 is not None:
            self.mht1.update(self.prev1, address, strengthen=True)
        if self.prev2 is not None and self.prev1 is not None:
            self.mht2.update(self.prev2, self.prev1, address, strengthen=True)

    def _generate_prefetch_candidates(self) -> List[int]:
        """Generate prefetch candidates by querying MHT2 and MHT1."""
        candidates: List[int] = []

        if self.prev2 is not None and self.prev1 is not None:
            mht2_entry = self.mht2.get(self.prev2, self.prev1)
            if mht2_entry and mht2_entry.predicted:
                for pred in mht2_entry.predicted:
                    if len(candidates) >= CONFIG["DEGREE_MHT2"]:
                        break
                    if mht2_entry.conf > 0:
                        candidates.append(pred)
                logger.debug(
                    "Domino: MHT2 lookup produced %d candidates", len(candidates)
                )

        if len(candidates) < CONFIG["DEGREE_MHT2"] and self.prev1 is not None:
            mht1_entry = self.mht1.get(self.prev1)
            if mht1_entry and mht1_entry.conf > 0:
                candidates.append(mht1_entry.predicted)
                logger.debug("Domino: MHT1 lookup added pred %x", mht1_entry.predicted)

        return self._deduplicate_candidates(candidates)

    def _deduplicate_candidates(self, candidates: List[int]) -> List[int]:
        """Remove duplicates while preserving order."""
        seen = set()
        filtered = []
        for candidate in candidates:
            if candidate not in seen:
                filtered.append(candidate)
                seen.add(candidate)
        return filtered[:PREFETCH_DEGREE]

    def _update_history_and_tables(self, current_address: int) -> None:
        """Update MHT tables and advance miss history."""
        if self.prev1 is not None:
            self.mht1.update(self.prev1, current_address, strengthen=False)
        if self.prev2 is not None and self.prev1 is not None:
            self.mht2.update(self.prev2, self.prev1, current_address, strengthen=False)

        self.prev2 = self.prev1
        self.prev1 = current_address

    def close(self) -> None:
        """Reset prefetcher state."""
        logger.info("DominoPrefetcher.close")
        self.mht1 = MHT1(size=CONFIG["MHT1_SIZE"])
        self.mht2 = MHT2(size=CONFIG["MHT2_SIZE"], degree=CONFIG["DEGREE_MHT2"])
        self.scheduler = Scheduler(
            max_outstanding=CONFIG["MAX_OUTSTANDING"],
            mrbsz=CONFIG["MRB_SIZE"],
        )
        self.prev1 = None
        self.prev2 = None
        self.initialized = False
