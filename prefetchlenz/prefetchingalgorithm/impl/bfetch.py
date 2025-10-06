"""
B-Fetch: Branch-Prediction Directed Prefetching (software approximation)

Place in: prefetchlenz/prefetchingalgorithm/impl/bfetch.py

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

# Import MemoryAccess from your repo (must exist)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

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
    bs = CONFIG["BLOCK_SIZE"]
    return (addr // bs) * bs


def pc_sequential_next(pc: int) -> int:
    """Sequential next PC assuming fixed INSTR_WIDTH."""
    return pc + CONFIG["INSTR_WIDTH"]


# ----------------------
# MRB: miss-resolution / suppression buffer
# ----------------------
class MRB:
    """Short-term suppression buffer to avoid immediate re-issue of same blocks."""

    def __init__(self, size: int = CONFIG["MRB_SIZE"]):
        self.size = size
        self.buf: List[int] = []

    def insert(self, addr: int) -> None:
        if addr in self.buf:
            self.buf.remove(addr)
            self.buf.append(addr)
            return
        if len(self.buf) >= self.size:
            self.buf.pop(0)
        self.buf.append(addr)
        logger.debug("MRB insert %x", addr)

    def contains(self, addr: int) -> bool:
        return addr in self.buf

    def remove(self, addr: int) -> None:
        if addr in self.buf:
            self.buf.remove(addr)


# ----------------------
# Simple scheduler: outstanding cap + MRB suppression
# ----------------------
class Scheduler:
    """Issues prefetches subject to outstanding cap and MRB."""

    def __init__(
        self,
        max_outstanding: int = CONFIG["MAX_OUTSTANDING"],
        mrbsz: int = CONFIG["MRB_SIZE"],
    ):
        self.max_outstanding = max_outstanding
        self.outstanding: List[int] = []  # maintain list for deterministic behavior
        self.mrb = MRB(size=mrbsz)

    def can_issue(self) -> bool:
        return len(self.outstanding) < self.max_outstanding

    def issue(self, candidates: List[int]) -> List[int]:
        """Issue candidates respecting MRB and outstanding cap."""
        issued = []
        for a in candidates:
            if len(issued) >= PREFETCH_DEGREE:
                break
            if a in self.outstanding:
                continue
            if self.mrb.contains(a):
                continue
            if not self.can_issue():
                break
            self.outstanding.append(a)
            self.mrb.insert(a)
            issued.append(a)
            logger.info("Scheduler issued prefetch %x", a)
        return issued

    def credit(self, addr: int) -> None:
        """Called when a prefetched address is used (prefetch_hit)."""
        if addr in self.outstanding:
            self.outstanding.remove(addr)
            logger.debug("Scheduler credited outstanding for %x", addr)
        # keep it in MRB to avoid immediate reissue
        self.mrb.insert(addr)

    def clear(self) -> None:
        self.outstanding.clear()
        self.mrb = MRB(size=self.mrb.size)


# ----------------------
# BST: Branch Stream Table wrapper using supplied Cache
# ----------------------
class BST:
    """
    Branch Stream Table wrapper.

    Uses the framework's Cache class for storage (set-associative). We store StreamDescriptor
    values keyed by branch_pc (int). If the project-supplied Cache is not desirable, this class
    falls back to an in-memory dict.

    Methods:
      - lookup(branch_pc) -> StreamDescriptor | None
      - insert(branch_pc, desc)
      - update_confidence(branch_pc, delta)
      - remove(branch_pc)
    """

    def __init__(self, cache_factory: Optional[Any] = None):
        """
        cache_factory: optional callable (num_sets, num_ways, replacement_policy_cls) -> Cache-like
        If None, we use an in-memory dict with deterministic eviction policy.
        """
        # Try to construct a Cache if factory provided
        self.use_cache = False
        self.cache = None
        if cache_factory is not None:
            try:
                # instantiate with num_sets and assoc (ways)
                self.cache = cache_factory(CONFIG["BST_NUM_SETS"], CONFIG["BST_ASSOC"])
                self.use_cache = True
            except Exception:
                self.use_cache = False
                self.cache = {}
        else:
            self.cache = {}
        # fallback dict insertion order tracking for deterministic eviction
        self._order = []

    def _key(self, branch_pc: int) -> int:
        return branch_pc

    def lookup(self, branch_pc: int) -> Optional[StreamDescriptor]:
        k = self._key(branch_pc)
        if self.use_cache:
            v = self.cache.get(k)
            return v
        return self.cache.get(k)

    def insert(self, branch_pc: int, desc: StreamDescriptor) -> None:
        k = self._key(branch_pc)
        # use limited size logical enforcement if dict fallback
        if self.use_cache:
            self.cache.put(k, desc)
            logger.debug("BST insert via Cache key %x -> %s", k, desc)
            return
        # dict fallback
        if k not in self.cache and len(self.cache) >= CONFIG["BST_NUM_ENTRIES"]:
            # evict oldest
            victim = self._order.pop(0)
            del self.cache[victim]
            logger.debug("BST evict fallback key %x", victim)
        self.cache[k] = desc
        if k in self._order:
            try:
                self._order.remove(k)
            except ValueError:
                pass
        self._order.append(k)
        logger.debug("BST insert fallback key %x -> %s", k, desc)

    def update_conf(self, branch_pc: int, delta: int) -> None:
        k = self._key(branch_pc)
        ent = self.lookup(k)
        if not ent:
            return
        ent.conf = max(0, min(BST_CONF_MAX, ent.conf + delta))
        logger.debug("BST update_conf %x -> %d", k, ent.conf)
        # write back if using Cache
        if self.use_cache:
            self.cache.put(k, ent)

    def remove(self, branch_pc: int) -> None:
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
class BFetchPrefetcher:
    """
    B-Fetch prefetcher implementation adapted to frameworks without branch predictor signals.

    Approximations:
    - A non-sequential fetch (addr != last_pc + INSTR_WIDTH) is treated as a branch-prediction event
      where branch_pc == last_pc and predicted_target == addr.
    - A later demand access equal to predicted_target is treated as branch resolution/hit.
    - Use BST to remember stream descriptors keyed by branch_pc.
    """

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
            # Consult BST for branch_pc
            ent = self.bst.lookup(branch_pc)
            if ent and ent.conf > 0:
                # prepare candidates: target block + lookahead blocks in-order
                base = block_base(ent.target_base)
                candidates = []
                for i in range(ent.blocks + CONFIG["LOOKAHEAD_BLOCKS"]):
                    blk_addr = base + i * CONFIG["BLOCK_SIZE"]
                    candidates.append(blk_addr)
                # avoid duplicates and enforce PREFETCH_DEGREE
                filtered = []
                seen = set()
                for c in candidates:
                    if c not in seen:
                        filtered.append(c)
                        seen.add(c)
                    if len(filtered) >= PREFETCH_DEGREE:
                        break
                # Ask scheduler to issue (it handles MRB and outstanding cap)
                issued = self.scheduler.issue(filtered)
                logger.info(
                    "BFetch: issued %d prefetch(es) for branch_pc=%x",
                    len(issued),
                    branch_pc,
                )
            else:
                # No entry: create one using this observed prediction as training (speculative insert)
                # create descriptor with default BB size
                desc = StreamDescriptor(
                    target_base=block_base(pred_target),
                    blocks=CONFIG["BLOCKS_PER_BB"],
                    conf=CONFIG["BST_INIT_CONF"],
                )
                self.bst.insert(branch_pc, desc)
                logger.debug("BFetch: created BST entry for %x -> %s", branch_pc, desc)
            # Do not update last_pc/last_addr yet; wait for subsequent resolution
            # (We still set last_addr to the current access)
            self.last_addr = addr
            return issued

        # Otherwise this is sequential fetch or first event: update last_pc and last_addr
        self.last_pc = pc
        self.last_addr = addr
        return []

    def close(self) -> None:
        logger.info("BFetchPrefetcher.close")
        # Reset state
        self.bst = BST()
        self.scheduler.clear()
        self.last_pc = None
        self.last_addr = None
        self.initialized = False
