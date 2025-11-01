"""
Instruction Pointer Classifier-based Prefetcher (IPCP)

Algorithm: Bouquet of Instruction Pointers: Instruction Pointer Classifier-based
Spatial Hardware Prefetching by Panda et al.

This prefetcher classifies instruction pointers (PCs) into different classes and uses
specialized sub-prefetchers for each class. It adapts its prefetching strategy based on
the memory access pattern exhibited by each program counter.

Key Components:
- IPTable: Direct-mapped table classifying PCs into classes (CS, GS, CPLX)
- CSPrefetcher: Constant stride prefetcher for CS-class PCs
- GSPrefetcher: Region signature prefetcher for GS-class PCs
- CPLXPrefetcher: Hybrid prefetcher combining stride and spatial patterns
- IPCPPrefetcher: Main prefetcher coordinating sub-prefetchers and scheduler

How it works:
1. Classify each PC into CS (constant stride), GS (spatial signature), or CPLX (complex)
2. Use specialized prefetcher based on PC class
3. Track confidence per PC entry and update based on prefetch feedback
4. Filter predictions to only those in the same region as current access
5. Scheduler caps outstanding prefetches and avoids duplicates
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from prefetchlenz.prefetchingalgorithm.impl._shared import (
    Scheduler,
    align_to_region,
    get_region_offset,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger(__name__)

# -------------------------
# Config parameters
# -------------------------
CONFIG = {
    "REGION_SIZE": 2 * 1024,  # bytes
    "BLOCK_SIZE": 64,  # bytes
    "BLOCKS_PER_REGION": (2 * 1024) // 64,
    "IPTABLE_ENTRIES": 64,  # direct-mapped IP table
    "COUNTER_BITS": 2,
    "PREFETCH_DEGREE": 4,
    "MAX_OUTSTANDING": 16,
    "INIT_CONF": 2,  # init counter for observed
    "CONF_MAX": (1 << 2) - 1,
    "CONF_MIN": 0,
}


# -------------------------
# Helper
# -------------------------


def region_base(addr: int) -> int:
    return align_to_region(addr, CONFIG["REGION_SIZE"])


def block_index(addr: int) -> int:
    return get_region_offset(addr, CONFIG["REGION_SIZE"], CONFIG["BLOCK_SIZE"])


# -------------------------
# Data structures
# -------------------------


@dataclass
class IPEntry:
    """Instruction pointer entry.

    Attributes:
        ip: Instruction pointer (program counter).
        class_id: Classification ID (CS, GS, or CPLX).
        conf: Confidence counter.
        stride: Detected stride if applicable.
        last_addr: Last accessed address.
        signature: Per-block counters for spatial patterns.
    """

    ip: int
    class_id: str
    conf: int
    stride: Optional[int] = None
    last_addr: Optional[int] = None
    signature: Optional[List[int]] = None  # Per-block counters.


class IPTable:
    """Direct-mapped table mapping PC to IPEntry."""

    def __init__(self, entries: int = CONFIG["IPTABLE_ENTRIES"]):
        """Initialize IP table.

        Args:
            entries: Number of entries in the table.
        """
        self.entries = entries
        self.table: Dict[int, IPEntry] = {}

    def _index(self, ip: int) -> int:
        """Compute direct-mapped table index from instruction pointer."""
        return ip % self.entries

    def lookup(self, ip: int) -> Optional[IPEntry]:
        """Look up IP entry by instruction pointer."""
        return self.table.get(self._index(ip))

    def insert(self, ip: int, class_id: str) -> IPEntry:
        """Insert a new IP entry with the given classification."""
        table_index = self._index(ip)
        entry = IPEntry(
            ip=ip,
            class_id=class_id,
            conf=CONFIG["INIT_CONF"],
            stride=None,
            last_addr=None,
            signature=[CONFIG["INIT_CONF"]] * CONFIG["BLOCKS_PER_REGION"],
        )
        self.table[table_index] = entry
        logger.debug(f"Inserted IPEntry {entry}")
        return entry


# -------------------------
# Sub-prefetchers
# -------------------------


class CSPrefetcher:
    """Constant stride prefetcher for CS-class instruction pointers."""

    def update(self, entry: IPEntry, access_addr: int) -> None:
        """Update stride detection from the last and current address."""
        if entry.last_addr is not None:
            stride = (access_addr - entry.last_addr) // CONFIG["BLOCK_SIZE"]
            if entry.stride != stride:
                entry.stride = stride
        entry.last_addr = access_addr

    def predict(self, entry: IPEntry, access_addr: int) -> List[int]:
        """Generate stride-based prefetch predictions."""
        if entry.stride is None:
            return []
        base = access_addr + entry.stride * CONFIG["BLOCK_SIZE"]
        predictions = [
            base + step_ahead * entry.stride * CONFIG["BLOCK_SIZE"]
            for step_ahead in range(1, CONFIG["PREFETCH_DEGREE"] + 1)
        ]
        return predictions


class GSPrefetcher:
    """Region signature prefetcher for GS-class instruction pointers."""

    def update(self, entry: IPEntry, addr: int) -> None:
        """Update spatial signature by incrementing the block's confidence."""
        block_idx = block_index(addr)
        signature = entry.signature
        if signature is None:
            signature = [CONFIG["INIT_CONF"]] * CONFIG["BLOCKS_PER_REGION"]
            entry.signature = signature
        signature[block_idx] = min(CONFIG["CONF_MAX"], signature[block_idx] + 1)

    def predict(self, entry: IPEntry, addr: int) -> List[int]:
        """Generate spatial pattern predictions from signature."""
        base = region_base(addr)
        signature = entry.signature or [0] * CONFIG["BLOCKS_PER_REGION"]
        predictions = []
        for block_idx, conf in enumerate(signature):
            if conf >= CONFIG["INIT_CONF"]:
                block_addr = base + block_idx * CONFIG["BLOCK_SIZE"]
                predictions.append(block_addr)
        return predictions[: CONFIG["PREFETCH_DEGREE"]]


class CPLXPrefetcher:
    """Hybrid prefetcher mixing stride and signature patterns."""

    def __init__(self):
        self.cs = CSPrefetcher()
        self.gs = GSPrefetcher()

    def update(self, entry: IPEntry, addr: int) -> None:
        """Update both stride and spatial signature components."""
        self.cs.update(entry, addr)
        self.gs.update(entry, addr)

    def predict(self, entry: IPEntry, addr: int) -> List[int]:
        """Combine stride and spatial predictions."""
        combined = self.cs.predict(entry, addr) + self.gs.predict(entry, addr)
        return combined[: CONFIG["PREFETCH_DEGREE"]]


# -------------------------
# IPCP Prefetcher
# -------------------------


class IPCPPrefetcher(PrefetchAlgorithm):
    """Main IPCP prefetcher entrypoint.

    Instruction Pointer Classifier-based prefetcher that classifies PCs into
    classes (CS, GS, CPLX) and uses specialized sub-prefetchers per class.
    """

    def __init__(self):
        self.ip_table = IPTable()
        self.cs = CSPrefetcher()
        self.gs = GSPrefetcher()
        self.cplx = CPLXPrefetcher()
        self.scheduler = Scheduler(
            max_outstanding=CONFIG["MAX_OUTSTANDING"],
            prefetch_degree=CONFIG["PREFETCH_DEGREE"],
            mrbsz=64,
        )
        self.initialized = False

    def init(self) -> None:
        self.initialized = True
        logger.info("IPCP initialized")

    def close(self) -> None:
        self.initialized = False
        logger.info("IPCP closed")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """Process memory access and return prefetch addresses."""
        program_counter = access.pc
        access_address = access.address

        self.scheduler.new_cycle()
        entry = self._get_or_create_entry(program_counter)
        candidates = self._update_predictor_and_get_candidates(entry, access_address)
        self._update_confidence(entry, prefetch_hit, access_address)
        filtered = self._filter_candidates_by_region(candidates, access_address)

        return self.scheduler.issue(filtered)

    def _get_or_create_entry(self, program_counter: int) -> IPEntry:
        """Look up existing entry or create new one (default: GS class)."""
        entry = self.ip_table.lookup(program_counter)
        if entry is None:
            entry = self.ip_table.insert(program_counter, "GS")
        return entry

    def _update_predictor_and_get_candidates(
        self, entry: IPEntry, access_address: int
    ) -> List[int]:
        """Update the predictor for entry's class and return candidates."""
        if entry.class_id == "CS":
            self.cs.update(entry, access_address)
            return self.cs.predict(entry, access_address)
        elif entry.class_id == "GS":
            self.gs.update(entry, access_address)
            return self.gs.predict(entry, access_address)
        else:
            self.cplx.update(entry, access_address)
            return self.cplx.predict(entry, access_address)

    def _update_confidence(
        self, entry: IPEntry, prefetch_hit: bool, access_address: int
    ) -> None:
        """Update confidence counter based on prefetch feedback."""
        if prefetch_hit:
            entry.conf = min(CONFIG["CONF_MAX"], entry.conf + 1)
            self.scheduler.credit(access_address)
        else:
            entry.conf = max(CONFIG["CONF_MIN"], entry.conf - 1)

    def _filter_candidates_by_region(
        self, candidates: List[int], access_address: int
    ) -> List[int]:
        """Filter candidates to only those in the same region."""
        access_region = region_base(access_address)
        return [
            candidate
            for candidate in candidates
            if region_base(candidate) == access_region
        ]
