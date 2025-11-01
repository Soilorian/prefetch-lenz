"""
B-Fetch: Branch-Prediction Directed Prefetching by Kadjo et al.

Overview:
- Uses a Branch Stream Table (BST) to map branch PC -> stream descriptor (target base,
  block count, lookahead degree) plus a small confidence counter.
- Because framework lacks branch-predictor signals, we approximate:
  - A non-sequential instruction fetch (addr != last_pc + INSTR_WIDTH) is treated as a
    predicted-branch-target event for branch PC == last_pc.
  - When a later demand access matches the predicted target, treat as a branch resolution/hit.
  - Otherwise treat as misprediction and weaken BST entries.

Integration:
- Uses supplied Cache class for storing BST entries (configurable as small set-assoc via Cache wrapper).
- Implements PrefetchAlgorithm API: init(), progress(access, prefetch_hit) -> List[int], close().

All numeric limits are in CONFIG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

# Import Cache class from your repo; we will instantiate a small Cache for BST storage.
# The framework's Cache must be available at runtime; tests may use a small in-memory wrapper.
from typing import Any, Dict, List, Optional, Tuple

from prefetchlenz.prefetchingalgorithm.impl._shared import (
    MRB,
    Scheduler,
    align_to_block,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.impl.bfetch")

# ----------------------
# Configuration
# ----------------------
CONFIG: Dict[str, int] = {
    # Basic sizes
    "INSTR_WIDTH": 4,  # instruction size (bytes) for sequential PC detection
    "BST_NUM_SETS": 64,  # sets in BST Cache (small by default)
    "BST_ASSOC": 2,  # associativity
    "BST_NUM_ENTRIES": 128,  # total logical entries (for convenience)
    "BST_CONF_BITS": 2,  # confidence counter bits (0..3)
    "BST_INIT_CONF": 1,  # initial confidence
    "BST_CONF_MAX": (1 << 2) - 1,
    # Prefetch behavior
    "BLOCK_SIZE": 64,  # bytes (instruction cache line)
    "BLOCKS_PER_BB": 4,  # default basic-block length in blocks (configurable)
    "LOOKAHEAD_BLOCKS": 2,  # how many extra blocks after target to prefetch
    "PREFETCH_DEGREE": 4,
    "MAX_OUTSTANDING": 16,
    # MRB / suppression
    "MRB_SIZE": 64,
}

# derived
BST_CONF_MAX = CONFIG["BST_CONF_MAX"]
PREFETCH_DEGREE = CONFIG["PREFETCH_DEGREE"]


# ----------------------
# Data classes
# ----------------------


@dataclass
class StreamDescriptor:
    """
    Stream descriptor stored in BST: maps branch PC -> predicted target base and metadata.
    Fields:
      - target_base: address of the predicted basic-block start
      - blocks: number of blocks in the basic block (in cache blocks)
      - conf: saturating confidence counter (0..BST_CONF_MAX)
    """

    target_base: int
    blocks: int
    conf: int


# ----------------------
# Small helpers
# ----------------------


def block_base(addr: int) -> int:
    """Return block-aligned base address for an instruction fetch (block size)."""
    return align_to_block(addr, CONFIG["BLOCK_SIZE"])


def pc_sequential_next(pc: int) -> int:
    """Sequential next PC assuming fixed INSTR_WIDTH."""
    return pc + CONFIG["INSTR_WIDTH"]


# ----------------------
# MRB: miss-resolution / suppression buffer
# ----------------------
# ----------------------
# Simple scheduler: outstanding cap + MRB suppression
# ----------------------
from prefetchlenz.prefetchingalgorithm.impl._shared import MRB, Scheduler


# ----------------------
# BST: Branch Stream Table wrapper using supplied Cache
# ----------------------
class BST:
    """Branch Stream Table storing PC -> StreamDescriptor mappings."""

    def __init__(self, cache_factory: Optional[Any] = None):
        """
        Initialize BST.

        Args:
            cache_factory: Optional cache factory. If None, uses dict fallback.
        """
        self.use_cache = cache_factory is not None
        if self.use_cache:
            self.cache = cache_factory(CONFIG["BST_NUM_SETS"], CONFIG["BST_ASSOC"])
        else:
            # Dict fallback with LRU ordering
            self.cache: Dict[int, StreamDescriptor] = {}
            self._order: List[int] = []

    def _key(self, branch_pc: int) -> int:
        """Convert branch PC to key."""
        return branch_pc

    def insert(self, branch_pc: int, descriptor: StreamDescriptor) -> None:
        """Insert or update entry for branch_pc."""
        k = self._key(branch_pc)
        if self.use_cache:
            self.cache.put(k, descriptor)
        else:
            # Dict fallback: maintain LRU order, evict oldest if at capacity
            if k not in self.cache and len(self.cache) >= CONFIG["BST_NUM_ENTRIES"]:
                # Evict oldest
                if self._order:
                    oldest = self._order.pop(0)
                    if oldest in self.cache:
                        del self.cache[oldest]
            if k in self._order:
                self._order.remove(k)
            self.cache[k] = descriptor
            self._order.append(k)

    def lookup(self, branch_pc: int) -> Optional[StreamDescriptor]:
        """Lookup entry for branch_pc."""
        k = self._key(branch_pc)
        if self.use_cache:
            return self.cache.get(k)
        else:
            result = self.cache.get(k)
            if result is not None and k in self._order:
                # Move to end (MRU)
                self._order.remove(k)
                self._order.append(k)
            return result

    def update_conf(self, branch_pc: int, delta: int) -> None:
        """Update confidence for branch_pc entry."""
        ent = self.lookup(branch_pc)
        if ent is not None:
            new_conf = max(0, min(BST_CONF_MAX, ent.conf + delta))
            ent.conf = new_conf

    def remove(self, branch_pc: int) -> None:
        """Remove entry for branch_pc."""
        k = self._key(branch_pc)
        if self.use_cache:
            self.cache.remove(k)
        else:
            if k in self.cache:
                del self.cache[k]
                try:
                    self._order.remove(k)
                except ValueError:
                    pass


# ----------------------
# BFetchPrefetcher (top-level)
# ----------------------
class BFetchPrefetcher(PrefetchAlgorithm):
    """
    B-Fetch prefetcher implementation adapted to frameworks without branch predictor signals.

    Approximations:
    - A non-sequential fetch (addr != last_pc + INSTR_WIDTH) is treated as a branch-prediction event
      where branch_pc == last_pc and predicted_target == addr.
    - A later demand access equal to predicted_target is treated as branch resolution/hit.
    - Use BST to remember stream descriptors keyed by branch_pc.
    """

    def close(self):
        """Reset internal state."""
        self.initialized = False

    def __init__(self, cache_factory: Optional[Any] = None):
        """
        cache_factory: optional callable to create the project's Cache instances. It should accept
        (num_sets, num_ways) or be adapted as needed. If None, BST uses dict fallback.
        """
        self.bst = BST(cache_factory=cache_factory)
        self.scheduler = Scheduler(
            max_outstanding=CONFIG["MAX_OUTSTANDING"], mrbsz=CONFIG["MRB_SIZE"]
        )
        self.last_pc: Optional[int] = None
        self.last_addr: Optional[int] = None
        self.initialized = False

    # ---- PrefetchAlgorithm API ----
    def init(self) -> None:
        logger.info("BFetchPrefetcher.init")
        self.initialized = True

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Handle an instruction fetch access.

        - If prefetch_hit True: the address corresponds to a prefetched line being used; treat as feedback/credit.
        - Else: detect non-sequential control-flow (approximated branch-prediction event),
          consult BST for stream descriptor, issue prefetches of target basic block and lookahead blocks.
        """
        if not self.initialized:
            self.init()

        addr = access.address
        pc = access.pc
        issued: List[int] = []

        # If prefetch_hit: credit scheduler and strengthen BST mapping if exists
        if prefetch_hit:
            logger.debug("BFetch: prefetch_hit for %x", addr)
            self.scheduler.credit(addr)
            # If this address was predicted target for previous branch PC (last_pc), strengthen
            if self.last_pc is not None:
                ent = self.bst.lookup(self.last_pc)
                if ent and block_base(ent.target_base) == block_base(addr):
                    self.bst.update_conf(self.last_pc, +1)
            # Update state for next access
            self.last_pc = pc
            self.last_addr = addr
            return []

        # Detect non-sequential transfer (candidate predicted branch event)
        is_nonseq = self.last_pc is not None and addr != pc_sequential_next(
            self.last_pc
        )
        # Note: last_pc tracks previous instruction fetch PC (not branch PC); heuristic mapping:
        # treat branch_pc == last_pc when non-seq observed.
        if is_nonseq and self.last_pc is not None:
            branch_pc = self.last_pc
            pred_target = addr  # predicted target observed
            logger.debug(
                "BFetch: detected predicted branch event branch_pc=%x target=%x",
                branch_pc,
                pred_target,
            )
            # MRB and Scheduler are provided by the shared utilities module
            # Look up BST entry for this branch PC
            ent = self.bst.lookup(branch_pc)
            if ent is not None:
                # Entry exists: issue prefetches for target block + lookahead blocks
                target_block_base = block_base(pred_target)
                candidates: List[int] = []
                # Target block + lookahead blocks
                for i in range(ent.blocks + CONFIG["LOOKAHEAD_BLOCKS"]):
                    candidate_addr = target_block_base + i * CONFIG["BLOCK_SIZE"]
                    candidates.append(candidate_addr)
                # Issue prefetches through scheduler
                issued = self.scheduler.issue(
                    candidates, degree=CONFIG["PREFETCH_DEGREE"]
                )
            else:
                # No entry exists: create new descriptor
                target_block_base = block_base(pred_target)
                desc = StreamDescriptor(
                    target_base=target_block_base,
                    blocks=CONFIG["BLOCKS_PER_BB"],
                    conf=CONFIG["BST_INIT_CONF"],
                )
                self.bst.insert(branch_pc, desc)
                # No prefetches issued on first observation
                issued = []

        # Update state for next access
        self.last_pc = pc
        self.last_addr = addr

        # Return issued prefetches
        return issued
