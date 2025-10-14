"""
DSPatch: Dual Spatial Pattern Prefetcher - Python implementation

Overview:
- Learns two modulated spatial bit-patterns per physical page/PC signature
- Coverage-biased pattern (CoP): Uses OR operations for higher coverage
- Accuracy-biased pattern (AccP): Uses AND operations for higher accuracy
- Dynamically selects pattern based on memory bandwidth utilization
- Tracks accesses per 4KB page, learns patterns associated with trigger PC

Key Components:
- PageBuffer (PB): Records observed spatial bit-patterns per physical page
- SignaturePatternTable (SPT): Stores two modulated bit-patterns (CoP, AccP) per PC signature
- Bandwidth tracking: Simple mechanism to track DRAM bandwidth utilization
- Pattern selection: Dynamic selection based on bandwidth and pattern quality

References:
- DSPatch: Dual Spatial Pattern Prefetcher (MICRO 2019)
- Rahul Bera, Anant V. Nori, Onur Mutlu, Sreenivas Subramoney
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.impl.dspatch")

# ----------------------
# Configuration
# ----------------------
CONFIG = {
    "PAGE_SIZE": 4096,  # 4KB pages
    "BLOCK_SIZE": 64,  # 64B cache blocks
    "REGION_SIZE": 2048,  # 2KB regions (half page)
    "BLOCKS_PER_PAGE": 64,  # 4KB / 64B = 64 blocks
    "BLOCKS_PER_REGION": 32,  # 2KB / 64B = 32 blocks
    "PB_ENTRIES": 64,  # Page Buffer entries
    "SPT_ENTRIES": 256,  # Signature Pattern Table entries
    "COMPRESSION": True,  # Use 128B-granularity compression
    "PATTERN_BITS": 32,  # Bits per pattern (after compression)
    "MAX_OR_COUNT": 3,  # Maximum OR operations for CoP
    "ACCURACY_THRESHOLD": 50,  # Accuracy threshold in percent
    "COVERAGE_THRESHOLD": 50,  # Coverage threshold in percent
    "BW_QUARTILES": 4,  # Bandwidth utilization quartiles
}

# Derived constants
BLOCKS_PER_PAGE = CONFIG["BLOCKS_PER_PAGE"]
BLOCKS_PER_REGION = CONFIG["BLOCKS_PER_REGION"]
PATTERN_BITS = CONFIG["PATTERN_BITS"]


# ----------------------
# Helper Functions
# ----------------------
def page_number(address: int) -> int:
    """Get page number from address."""
    return address // CONFIG["PAGE_SIZE"]


def page_offset(address: int) -> int:
    """Get block offset within page."""
    return (address % CONFIG["PAGE_SIZE"]) // CONFIG["BLOCK_SIZE"]


def region_in_page(offset: int) -> int:
    """Get which 2KB region (0 or 1) the offset belongs to."""
    return offset // BLOCKS_PER_REGION


def compress_pattern(pattern: int, pattern_size: int = 64) -> int:
    """
    Compress 64-bit pattern to 32-bit using 128B-granularity.
    Each bit in compressed pattern represents two adjacent 64B blocks.
    """
    if not CONFIG["COMPRESSION"]:
        return pattern

    compressed = 0
    for i in range(pattern_size // 2):
        # OR adjacent bits together
        bit1 = (pattern >> (i * 2)) & 1
        bit2 = (pattern >> (i * 2 + 1)) & 1
        if bit1 or bit2:
            compressed |= (1 << i)
    return compressed


def decompress_pattern(compressed: int, pattern_size: int = 64) -> int:
    """
    Decompress 32-bit pattern back to 64-bit.
    Each bit expands to two adjacent bits.
    """
    if not CONFIG["COMPRESSION"]:
        return compressed

    pattern = 0
    for i in range(pattern_size // 2):
        if (compressed >> i) & 1:
            pattern |= (3 << (i * 2))  # Set both adjacent bits
    return pattern


def rotate_pattern(pattern: int, offset: int, pattern_size: int = PATTERN_BITS) -> int:
    """Rotate pattern left by offset positions (anchoring)."""
    offset = offset % pattern_size
    mask = (1 << pattern_size) - 1
    return ((pattern << offset) | (pattern >> (pattern_size - offset))) & mask


def popcount(pattern: int) -> int:
    """Count number of set bits in pattern."""
    return bin(pattern).count('1')


def hash_pc(pc: int) -> int:
    """Folded XOR hash of PC for SPT indexing."""
    index_bits = 8  # 256 entries = 2^8
    result = 0
    for i in range(0, 64, index_bits):
        result ^= (pc >> i) & ((1 << index_bits) - 1)
    return result % CONFIG["SPT_ENTRIES"]


# ----------------------
# Data Structures
# ----------------------
@dataclass
class PageBufferEntry:
    """Entry in Page Buffer tracking a 4KB page."""
    page_num: int = 0
    pattern: int = 0  # 64-bit pattern
    trigger_pc: List[int] = field(default_factory=lambda: [0, 0])  # 2 triggers per page
    trigger_offset: List[int] = field(default_factory=lambda: [0, 0])
    triggered: List[bool] = field(default_factory=lambda: [False, False])
    valid: bool = False


@dataclass
class SignaturePatternEntry:
    """Entry in Signature Pattern Table."""
    # Coverage-biased pattern (per 2KB region)
    cov_pattern: List[int] = field(default_factory=lambda: [0, 0])  # 2x 32-bit patterns
    measure_cov: List[int] = field(default_factory=lambda: [0, 0])  # 2-bit saturating counters
    or_count: List[int] = field(default_factory=lambda: [0, 0])  # 2-bit counters

    # Accuracy-biased pattern (per 2KB region)
    acc_pattern: List[int] = field(default_factory=lambda: [0, 0])  # 2x 32-bit patterns
    measure_acc: List[int] = field(default_factory=lambda: [0, 0])  # 2-bit saturating counters


# ----------------------
# Bandwidth Tracker
# ----------------------
class BandwidthTracker:
    """Tracks DRAM bandwidth utilization."""

    def __init__(self):
        self.counter = 0
        self.window_size = 1000  # Simplified window
        self.access_count = 0
        self.current_quartile = 0  # 0-3 (0=<25%, 3=>=75%)

    def record_access(self) -> None:
        """Record a memory access."""
        self.counter += 1
        self.access_count += 1

        if self.access_count >= self.window_size:
            # Update quartile based on utilization
            utilization = self.counter / self.window_size
            if utilization < 0.25:
                self.current_quartile = 0
            elif utilization < 0.50:
                self.current_quartile = 1
            elif utilization < 0.75:
                self.current_quartile = 2
            else:
                self.current_quartile = 3

            # Halve counter for next window (hysteresis)
            self.counter = self.counter // 2
            self.access_count = 0

    def get_quartile(self) -> int:
        """Get current bandwidth utilization quartile (0-3)."""
        return self.current_quartile


# ----------------------
# Page Buffer
# ----------------------
class PageBuffer:
    """Tracks accesses to recently-accessed physical pages."""

    def __init__(self, entries: int):
        self.entries: Dict[int, PageBufferEntry] = {}
        self.max_entries = entries
        self.lru_order: List[int] = []

    def access(self, page_num: int, offset: int, pc: int) -> Optional[PageBufferEntry]:
        """Record access to page and return entry."""
        if page_num not in self.entries:
            # Allocate new entry
            if len(self.entries) >= self.max_entries:
                # Evict LRU
                victim = self.lru_order.pop(0)
                del self.entries[victim]

            self.entries[page_num] = PageBufferEntry(page_num=page_num, valid=True)
            self.lru_order.append(page_num)

        entry = self.entries[page_num]

        # Update LRU
        if page_num in self.lru_order:
            self.lru_order.remove(page_num)
        self.lru_order.append(page_num)

        # Set bit in pattern
        entry.pattern |= (1 << offset)

        # Check if this is a trigger access (first access to 2KB region)
        region = region_in_page(offset)
        if not entry.triggered[region]:
            entry.trigger_pc[region] = pc
            entry.trigger_offset[region] = offset
            entry.triggered[region] = True

        return entry

    def get_entry(self, page_num: int) -> Optional[PageBufferEntry]:
        """Get entry for page."""
        return self.entries.get(page_num)

    def evict_lru(self) -> Optional[PageBufferEntry]:
        """Evict and return LRU entry."""
        if not self.lru_order:
            return None
        victim = self.lru_order.pop(0)
        entry = self.entries.pop(victim, None)
        return entry


# ----------------------
# Signature Pattern Table
# ----------------------
class SignaturePatternTable:
    """Stores modulated bit-patterns indexed by PC signature."""

    def __init__(self, entries: int):
        self.entries: Dict[int, SignaturePatternEntry] = {}
        self.max_entries = entries

    def lookup(self, pc: int) -> SignaturePatternEntry:
        """Lookup entry by PC (creates if not exists)."""
        index = hash_pc(pc)
        if index not in self.entries:
            self.entries[index] = SignaturePatternEntry()
        return self.entries[index]

    def update(self, pc: int, region: int, program_pattern: int,
               cov_pattern: int, acc_pattern: int,
               measure_cov: int, measure_acc: int,
               or_count: int) -> None:
        """Update SPT entry."""
        entry = self.lookup(pc)
        entry.cov_pattern[region] = cov_pattern
        entry.acc_pattern[region] = acc_pattern
        entry.measure_cov[region] = measure_cov
        entry.measure_acc[region] = measure_acc
        entry.or_count[region] = or_count


# ----------------------
# DSPatch Prefetcher
# ----------------------
class DSPatchPrefetcher(PrefetchAlgorithm):
    """DSPatch: Dual Spatial Pattern Prefetcher."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize DSPatch with optional config overrides."""
        if config:
            for k, v in config.items():
                if k in CONFIG:
                    CONFIG[k] = v

        self.page_buffer = PageBuffer(CONFIG["PB_ENTRIES"])
        self.spt = SignaturePatternTable(CONFIG["SPT_ENTRIES"])
        self.bw_tracker = BandwidthTracker()
        self.initialized = False

        logger.debug("DSPatchPrefetcher created")

    def init(self) -> None:
        """Initialize prefetcher state."""
        self.initialized = True
        logger.info("DSPatchPrefetcher initialized")

    def _quantify_accuracy(self, program_pattern: int, pred_pattern: int) -> int:
        """Quantify prediction accuracy (0-3 quartiles)."""
        if pred_pattern == 0:
            return 0

        accurate_count = popcount(program_pattern & pred_pattern)
        pred_count = popcount(pred_pattern)

        accuracy = (accurate_count * 100) // pred_count if pred_count > 0 else 0

        # Convert to quartile (0-3)
        if accuracy < 25:
            return 0
        elif accuracy < 50:
            return 1
        elif accuracy < 75:
            return 2
        else:
            return 3

    def _quantify_coverage(self, program_pattern: int, pred_pattern: int) -> int:
        """Quantify prediction coverage (0-3 quartiles)."""
        if program_pattern == 0:
            return 0

        accurate_count = popcount(program_pattern & pred_pattern)
        real_count = popcount(program_pattern)

        coverage = (accurate_count * 100) // real_count if real_count > 0 else 0

        # Convert to quartile (0-3)
        if coverage < 25:
            return 0
        elif coverage < 50:
            return 1
        elif coverage < 75:
            return 2
        else:
            return 3

    def _update_patterns(self, spt_entry: SignaturePatternEntry, region: int,
                        program_pattern: int) -> None:
        """Update CoP and AccP patterns."""
        # Update Coverage-biased Pattern (CoP)
        old_cov = spt_entry.cov_pattern[region]
        new_cov = old_cov | program_pattern  # OR operation

        if new_cov != old_cov and spt_entry.or_count[region] < CONFIG["MAX_OR_COUNT"]:
            spt_entry.or_count[region] = min(3, spt_entry.or_count[region] + 1)
            spt_entry.cov_pattern[region] = new_cov

        # Check if CoP needs reset
        accuracy = self._quantify_accuracy(program_pattern, spt_entry.cov_pattern[region])
        coverage = self._quantify_coverage(program_pattern, spt_entry.cov_pattern[region])

        if accuracy < 2 or coverage < 2:  # Less than 50%
            spt_entry.measure_cov[region] = min(3, spt_entry.measure_cov[region] + 1)

        # Reset CoP if measure saturated
        bw_quartile = self.bw_tracker.get_quartile()
        if spt_entry.measure_cov[region] >= 3:
            if bw_quartile >= 3 or coverage < 2:
                spt_entry.cov_pattern[region] = program_pattern
                spt_entry.measure_cov[region] = 0
                spt_entry.or_count[region] = 0

        # Update Accuracy-biased Pattern (AccP)
        spt_entry.acc_pattern[region] = spt_entry.cov_pattern[region] & program_pattern

        # Update AccP measure
        acc_accuracy = self._quantify_accuracy(program_pattern, spt_entry.acc_pattern[region])
        if acc_accuracy < 2:  # Less than 50%
            spt_entry.measure_acc[region] = min(3, spt_entry.measure_acc[region] + 1)
        else:
            spt_entry.measure_acc[region] = max(0, spt_entry.measure_acc[region] - 1)

    def _select_pattern(self, spt_entry: SignaturePatternEntry, region: int) -> Optional[int]:
        """Select CoP or AccP based on bandwidth utilization."""
        bw_quartile = self.bw_tracker.get_quartile()

        # Highest bandwidth utilization (>=75%)
        if bw_quartile >= 3:
            if spt_entry.measure_acc[region] < 3:
                return spt_entry.acc_pattern[region]
            else:
                return None  # Throttle completely

        # Medium-high bandwidth (50-75%)
        elif bw_quartile >= 2:
            if spt_entry.measure_cov[region] >= 3:
                return spt_entry.acc_pattern[region]
            else:
                return spt_entry.cov_pattern[region]

        # Low bandwidth (<50%)
        else:
            return spt_entry.cov_pattern[region]

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

        # Track bandwidth
        if not prefetch_hit:
            self.bw_tracker.record_access()

        # Get page info
        pnum = page_number(addr)
        offset = page_offset(addr)
        region = region_in_page(offset)

        logger.debug(f"Access: addr={addr:#x}, pc={pc:#x}, page={pnum:#x}, offset={offset}, region={region}")

        # Update page buffer
        pb_entry = self.page_buffer.access(pnum, offset, pc)

        prefetches = []

        # Check if this is a trigger access
        if pb_entry and pb_entry.triggered[region]:
            trigger_pc = pb_entry.trigger_pc[region]
            trigger_offset = pb_entry.trigger_offset[region]

            # Only first trigger can generate prefetches
            if offset == trigger_offset:
                # Lookup SPT
                spt_entry = self.spt.lookup(trigger_pc)

                # Select pattern
                pattern = self._select_pattern(spt_entry, region)

                if pattern is not None and pattern != 0:
                    # Decompress pattern
                    full_pattern = decompress_pattern(pattern)

                    # Anchor pattern to trigger offset
                    region_offset = trigger_offset % BLOCKS_PER_REGION
                    anchored = rotate_pattern(full_pattern, region_offset, BLOCKS_PER_REGION)

                    # Generate prefetches from pattern
                    base_offset = region * BLOCKS_PER_REGION
                    for i in range(BLOCKS_PER_REGION):
                        if (anchored >> i) & 1:
                            pf_offset = base_offset + i
                            if pf_offset != offset:  # Don't prefetch current address
                                pf_addr = (pnum * CONFIG["PAGE_SIZE"]) + (pf_offset * CONFIG["BLOCK_SIZE"])
                                prefetches.append(pf_addr)

                    logger.debug(f"Generated {len(prefetches)} prefetches from pattern")

        return prefetches

    def close(self) -> None:
        """Clean up and commit remaining patterns."""
        # Commit all page buffer entries
        for page_num, pb_entry in list(self.page_buffer.entries.items()):
            if pb_entry.valid:
                # Process each triggered region
                for region in range(2):
                    if pb_entry.triggered[region]:
                        trigger_pc = pb_entry.trigger_pc[region]

                        # Extract region pattern
                        base_offset = region * BLOCKS_PER_REGION
                        region_pattern = 0
                        for i in range(BLOCKS_PER_REGION):
                            if (pb_entry.pattern >> (base_offset + i)) & 1:
                                region_pattern |= (1 << i)

                        # Anchor pattern
                        trigger_offset = pb_entry.trigger_offset[region]
                        region_offset = trigger_offset % BLOCKS_PER_REGION
                        anchored = rotate_pattern(region_pattern, -region_offset, BLOCKS_PER_REGION)

                        # Compress
                        compressed = compress_pattern(anchored, BLOCKS_PER_REGION)

                        # Update SPT
                        spt_entry = self.spt.lookup(trigger_pc)
                        self._update_patterns(spt_entry, region, compressed)

        # Reset state
        self.page_buffer = PageBuffer(CONFIG["PB_ENTRIES"])
        self.initialized = False
        logger.info("DSPatchPrefetcher closed")
