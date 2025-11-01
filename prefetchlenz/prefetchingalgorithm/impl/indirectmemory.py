"""
Indirect Memory Prefetcher by Yu et al.

Overview:
- Tracks indirect memory access patterns (pointer chasing)
- Uses Prefetch Table to detect streaming accesses and track indices
- Uses Indirect Pattern Detector (IPD) to correlate index values with base addresses
- Generates prefetches based on learned base+index*shift patterns
- Supports multiple shift values for different pointer sizes

Key Components:
- PrefetchTable: Tracks streaming accesses and index values per PC
- IndirectPatternDetector: Learns correlations between indices and base addresses
- SaturatingCounter: Confidence tracking for indirect patterns
- IndirectMemoryPrefetcher: Main class implementing PrefetchAlgorithm interface

References:
- Based on the gem5 Indirect Memory Prefetcher
- Adapted to Python with simplified data extraction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.lru import LruReplacementPolicy
from prefetchlenz.prefetchingalgorithm.impl._shared import (
    SaturatingCounter as BaseSaturatingCounter,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.impl.indirectmemory")

# ----------------------
# Configuration
# ----------------------
CONFIG = {
    "MAX_PREFETCH_DISTANCE": 16,  # Maximum number of prefetches to generate
    "SHIFT_VALUES": [3, 4, 5, 6],  # Shift values to try (8, 16, 32, 64 bytes)
    "PREFETCH_THRESHOLD": 2,  # Minimum confidence to start prefetching
    "STREAM_COUNTER_THRESHOLD": 4,  # Accesses needed to detect streaming
    "STREAMING_DISTANCE": 4,  # Number of stream prefetches to issue
    "PT_TABLE_ENTRIES": 256,  # Prefetch table entries
    "PT_TABLE_ASSOC": 4,  # Prefetch table associativity
    "IPD_TABLE_ENTRIES": 256,  # IPD table entries
    "IPD_TABLE_ASSOC": 4,  # IPD table associativity
    "ADDR_ARRAY_LEN": 4,  # Number of misses to track per shift value
    "NUM_INDIRECT_COUNTER_BITS": 2,  # Bits for saturating counter
}

# Derived constants
MAX_COUNTER_VALUE = (1 << CONFIG["NUM_INDIRECT_COUNTER_BITS"]) - 1


# ----------------------
# Saturating Counter
# ----------------------


# Extend shared SaturatingCounter with indirectmemory-specific methods
class SaturatingCounter(BaseSaturatingCounter):
    """Saturating counter with configurable bit width and indirect memory extensions."""

    def __init__(self, bits: int = CONFIG["NUM_INDIRECT_COUNTER_BITS"]):
        super().__init__(bits=bits, initial_value=0)

    def reset(self) -> None:
        """Reset counter to maximum value."""
        self.value = self.max_value

    def calc_saturation(self) -> float:
        """Calculate saturation ratio [0.0, 1.0]."""
        if self.max_value == 0:
            return 1.0
        return self.value / self.max_value

    def __gt__(self, other) -> bool:
        if isinstance(other, int):
            return self.value > other
        return self.value > other.value

    def __repr__(self) -> str:
        return f"SaturatingCounter({self.value}/{self.max_value})"


# ----------------------
# Data Structures
# ----------------------
@dataclass
class PrefetchTableEntry:
    """Entry in the prefetch table."""

    address: int = 0
    secure: bool = False
    enabled: bool = False
    index: int = 0
    baseAddr: int = 0
    shift: int = 0
    streamCounter: int = 0
    indirectCounter: SaturatingCounter = field(default_factory=SaturatingCounter)
    increasedIndirectCounter: bool = False


@dataclass
class IndirectPatternDetectorEntry:
    """Entry in the indirect pattern detector."""

    idx1: int = 0
    idx2: int = 0
    secondIndexSet: bool = False
    baseAddr: List[List[int]] = field(default_factory=list)
    numMisses: int = 0

    def __post_init__(self):
        # Initialize baseAddr array if not provided
        if not self.baseAddr:
            addr_array_len = CONFIG["ADDR_ARRAY_LEN"]
            num_shifts = len(CONFIG["SHIFT_VALUES"])
            self.baseAddr = [
                [0 for _ in range(num_shifts)] for _ in range(addr_array_len)
            ]


# ----------------------
# Prefetch Table
# ----------------------
class PrefetchTable:
    """Prefetch table using set-associative cache."""

    def __init__(self, entries: int, assoc: int):
        num_sets = entries // assoc
        self.cache = Cache(
            num_sets=num_sets,
            num_ways=assoc,
            replacement_policy_cls=LruReplacementPolicy,
        )
        self.entries: Dict[int, PrefetchTableEntry] = {}

    def find_entry(self, pc: int) -> Optional[PrefetchTableEntry]:
        """Find entry by PC."""
        result = self.cache.get(pc)
        if result is not None:
            return self.entries.get(pc)
        return None

    def access_entry(self, entry: PrefetchTableEntry) -> None:
        """Touch entry (update LRU)."""
        # Find PC for this entry
        for pc, e in self.entries.items():
            if e is entry:
                self.cache.get(pc)  # Touch to update LRU
                break

    def insert_entry(self, pc: int, entry: PrefetchTableEntry) -> None:
        """Insert entry."""
        self.cache.put(pc, pc)
        self.entries[pc] = entry
        # Clean up entries not in cache
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove entries from dict that are no longer in cache."""
        to_remove = []
        for pc in list(self.entries.keys()):
            if self.cache.get(pc) is None:
                to_remove.append(pc)
        for pc in to_remove:
            del self.entries[pc]

    def __iter__(self):
        """Iterate over all valid entries."""
        # Clean up before iteration
        self._cleanup()
        return iter(self.entries.values())


# ----------------------
# Indirect Pattern Detector
# ----------------------
class IndirectPatternDetector:
    """Indirect pattern detector table."""

    def __init__(self, entries: int, assoc: int):
        num_sets = entries // assoc
        self.cache = Cache(
            num_sets=num_sets,
            num_ways=assoc,
            replacement_policy_cls=LruReplacementPolicy,
        )
        self.entries: Dict[int, IndirectPatternDetectorEntry] = {}

    def find_entry(self, addr: int) -> Optional[IndirectPatternDetectorEntry]:
        """Find entry by address (tag)."""
        result = self.cache.get(addr)
        if result is not None:
            return self.entries.get(addr)
        return None

    def access_entry(self, entry: IndirectPatternDetectorEntry) -> None:
        """Touch entry (update LRU)."""
        for addr, e in self.entries.items():
            if e is entry:
                self.cache.get(addr)
                break

    def insert_entry(self, addr: int, entry: IndirectPatternDetectorEntry) -> None:
        """Insert entry."""
        self.cache.put(addr, addr)
        self.entries[addr] = entry
        # Clean up entries not in cache
        self._cleanup()

    def invalidate(self, entry: IndirectPatternDetectorEntry) -> None:
        """Invalidate entry."""
        for addr, e in list(self.entries.items()):
            if e is entry:
                self.cache.remove(addr)
                del self.entries[addr]
                break

    def _cleanup(self) -> None:
        """Remove entries from dict that are no longer in cache."""
        to_remove = []
        for addr in list(self.entries.keys()):
            if self.cache.get(addr) is None:
                to_remove.append(addr)
        for addr in to_remove:
            del self.entries[addr]


# ----------------------
# Indirect Memory Prefetcher
# ----------------------
class IndirectMemoryPrefetcher(PrefetchAlgorithm):
    """Indirect Memory Prefetcher implementation."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize prefetcher with optional config overrides."""
        # Update config if provided
        if config:
            for k, v in config.items():
                if k in CONFIG:
                    CONFIG[k] = v

        # Initialize tables
        self.prefetchTable = PrefetchTable(
            CONFIG["PT_TABLE_ENTRIES"], CONFIG["PT_TABLE_ASSOC"]
        )
        self.ipd = IndirectPatternDetector(
            CONFIG["IPD_TABLE_ENTRIES"], CONFIG["IPD_TABLE_ASSOC"]
        )

        # Tracking for IPD entry currently accumulating misses
        self.ipdEntryTrackingMisses: Optional[IndirectPatternDetectorEntry] = None

        self.initialized = False
        logger.debug("IndirectMemoryPrefetcher created")

    def init(self) -> None:
        """Initialize prefetcher state."""
        self.initialized = True
        logger.info("IndirectMemoryPrefetcher initialized")

    def checkAccessMatchOnActiveEntries(self, addr: int) -> None:
        """Check if address matches any active indirect pattern."""
        for pt_entry in self.prefetchTable:
            if pt_entry.enabled:
                predicted_addr = pt_entry.baseAddr + (pt_entry.index << pt_entry.shift)
                if addr == predicted_addr:
                    pt_entry.indirectCounter.increment()
                    pt_entry.increasedIndirectCounter = True
                    logger.debug(
                        f"Match on active entry: addr={addr:#x}, counter={pt_entry.indirectCounter}"
                    )

    def allocateOrUpdateIPDEntry(
        self, pt_entry: PrefetchTableEntry, index: int
    ) -> None:
        """Allocate or update IPD entry for pattern detection."""
        # Use pt_entry address as IPD key
        ipd_entry_addr = id(pt_entry)
        ipd_entry = self.ipd.find_entry(ipd_entry_addr)

        if ipd_entry is not None:
            self.ipd.access_entry(ipd_entry)
            if not ipd_entry.secondIndexSet:
                # Second time we see an index, fill idx2
                ipd_entry.idx2 = index
                ipd_entry.secondIndexSet = True
                self.ipdEntryTrackingMisses = ipd_entry
                logger.debug(f"IPD: Set idx2={index}, tracking misses")
            else:
                # Third access! no pattern has been found so far, release the IPD entry
                self.ipd.invalidate(ipd_entry)
                self.ipdEntryTrackingMisses = None
                logger.debug(f"IPD: Third access, invalidating entry")
        else:
            # Create new IPD entry
            ipd_entry = IndirectPatternDetectorEntry()
            ipd_entry.idx1 = index
            self.ipd.insert_entry(ipd_entry_addr, ipd_entry)
            self.ipdEntryTrackingMisses = ipd_entry
            logger.debug(f"IPD: Created new entry with idx1={index}")

    def trackMissIndex1(self, miss_addr: int) -> None:
        """Track miss address for first index."""
        if self.ipdEntryTrackingMisses is None:
            return

        entry = self.ipdEntryTrackingMisses

        # If the second index is not set, we are filling the baseAddr vector
        if entry.numMisses >= len(entry.baseAddr):
            self.ipdEntryTrackingMisses = None
            return

        ba_array = entry.baseAddr[entry.numMisses]
        for idx, shift in enumerate(CONFIG["SHIFT_VALUES"]):
            ba_array[idx] = miss_addr - (entry.idx1 << shift)

        entry.numMisses += 1
        logger.debug(
            f"Track miss idx1: addr={miss_addr:#x}, numMisses={entry.numMisses}"
        )

        if entry.numMisses >= len(entry.baseAddr):
            # Stop tracking misses once we have tracked enough
            self.ipdEntryTrackingMisses = None

    def trackMissIndex2(self, miss_addr: int) -> None:
        """Track miss address for second index and find pattern."""
        if self.ipdEntryTrackingMisses is None:
            return

        entry = self.ipdEntryTrackingMisses

        # Compare addresses generated during previous misses (using idx1)
        # against newly generated values using idx2
        for midx in range(entry.numMisses):
            ba_array = entry.baseAddr[midx]
            for idx, shift in enumerate(CONFIG["SHIFT_VALUES"]):
                if ba_array[idx] == (miss_addr - (entry.idx2 << shift)):
                    # Match found!
                    # Find the corresponding pt_entry (stored as IPD key)
                    pt_entry = None
                    for addr, ipd_e in self.ipd.entries.items():
                        if ipd_e is entry:
                            # addr is id(pt_entry), need to find actual pt_entry
                            for pc, pt_e in self.prefetchTable.entries.items():
                                if id(pt_e) == addr:
                                    pt_entry = pt_e
                                    break
                            break

                    if pt_entry is not None:
                        pt_entry.baseAddr = ba_array[idx]
                        pt_entry.shift = shift
                        pt_entry.enabled = True
                        pt_entry.indirectCounter.reset()
                        logger.debug(
                            f"Pattern found! base={pt_entry.baseAddr:#x}, shift={shift}"
                        )

                    # Release the current IPD Entry
                    self.ipd.invalidate(entry)
                    self.ipdEntryTrackingMisses = None
                    return

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process memory access and generate prefetches.

        Args:
            access: Memory access information
            prefetch_hit: Whether this access hit a prefetched line

        Returns:
            List of addresses to prefetch
        """
        if not self.initialized:
            self.init()

        addr = access.address
        pc = access.pc
        is_miss = not prefetch_hit

        logger.debug(f"Access: addr={addr:#x}, pc={pc:#x}, miss={is_miss}")

        # Check if this access matches any active indirect pattern
        self.checkAccessMatchOnActiveEntries(addr)

        # First check if this is a miss, if the prefetcher is tracking misses
        if self.ipdEntryTrackingMisses is not None and is_miss:
            # Check if the entry tracking misses has already set its second index
            if not self.ipdEntryTrackingMisses.secondIndexSet:
                self.trackMissIndex1(addr)
            else:
                self.trackMissIndex2(addr)
            return []

        # If misses are not being tracked, attempt to detect stream accesses
        pt_entry = self.prefetchTable.find_entry(pc)

        if pt_entry is not None:
            self.prefetchTable.access_entry(pt_entry)

            if pt_entry.address != addr:
                # Streaming access found
                pt_entry.streamCounter += 1

                prefetches = []
                if pt_entry.streamCounter >= CONFIG["STREAM_COUNTER_THRESHOLD"]:
                    # Generate streaming prefetches
                    delta = addr - pt_entry.address
                    for i in range(1, CONFIG["STREAMING_DISTANCE"] + 1):
                        prefetches.append(addr + delta * i)
                    logger.debug(f"Streaming: {len(prefetches)} prefetches")

                pt_entry.address = addr

                # Try to read index from access
                # For simplicity, assume index is in lower bits or provided separately
                # In real implementation, would read from cache data
                # Here we use a simplified model: index = (addr >> 6) & 0xFFFF
                # Only do this on cache hits (not misses) like the original C++ code
                index = (addr >> 6) & 0xFFFF

                if not is_miss and not pt_entry.enabled:
                    # Not enabled (no pattern detected in this stream)
                    # Add or update an entry in the pattern detector
                    # Only when data is available (not a miss)
                    self.allocateOrUpdateIPDEntry(pt_entry, index)
                elif not is_miss and pt_entry.enabled:
                    # Enabled entry, update the index
                    pt_entry.index = index
                    if not pt_entry.increasedIndirectCounter:
                        pt_entry.indirectCounter.decrement()
                    else:
                        # Set this to false, to see if the new index has any match
                        pt_entry.increasedIndirectCounter = False

                    # If the counter is high enough, start prefetching
                    if pt_entry.indirectCounter > CONFIG["PREFETCH_THRESHOLD"]:
                        saturation = pt_entry.indirectCounter.calc_saturation()
                        distance = int(CONFIG["MAX_PREFETCH_DISTANCE"] * saturation)
                        distance = max(1, distance)  # At least 1 prefetch

                        for delta in range(1, distance + 1):
                            pf_addr = pt_entry.baseAddr + (
                                (pt_entry.index + delta) << pt_entry.shift
                            )
                            prefetches.append(pf_addr)
                        logger.debug(
                            f"Indirect: {len(prefetches)} prefetches, distance={distance}"
                        )

                return prefetches
        else:
            # Create new entry
            pt_entry = PrefetchTableEntry(address=addr, secure=False)
            self.prefetchTable.insert_entry(pc, pt_entry)
            logger.debug(f"Created PT entry for pc={pc:#x}")

        return []

    def close(self) -> None:
        """Clean up prefetcher state."""
        # Clear all tables
        self.prefetchTable = PrefetchTable(
            CONFIG["PT_TABLE_ENTRIES"], CONFIG["PT_TABLE_ASSOC"]
        )
        self.ipd = IndirectPatternDetector(
            CONFIG["IPD_TABLE_ENTRIES"], CONFIG["IPD_TABLE_ASSOC"]
        )
        self.ipdEntryTrackingMisses = None
        self.initialized = False
        logger.info("IndirectMemoryPrefetcher closed")
