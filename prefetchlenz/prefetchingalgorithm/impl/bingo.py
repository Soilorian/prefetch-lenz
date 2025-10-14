"""
Bingo Spatial Prefetcher - Python implementation

Place this file in: prefetchlenz/prefetchingalgorithm/impl/bingo.py

Overview:
- Implements spatial pattern prefetching using region-based tracking
- Three-stage lifecycle: FilterTable -> AccumulationTable -> PatternHistoryTable
- Four Pattern History Tables with different indexing schemes (PC+Address, PC+Offset, Address, PC)
- Rotation logic for offset-independent pattern storage

Key Components:
- Cache Framework: Base classes for set-associative and LRU caches
- FilterTable: Tracks first access to regions
- AccumulationTable: Accumulates spatial patterns within regions
- PatternHistoryTable: Stores learned patterns with different indexing
- BingoPrefetcher: Main class implementing PrefetchAlgorithm interface

References:
- Based on the Bingo spatial prefetcher
- Adapted to framework without explicit eviction callbacks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.impl.bingo")

# ----------------------
# Configuration
# ----------------------
CONFIG = {
    "PATTERN_LEN": 32,  # blocks per region (2KB / 64B = 32)
    "REGION_SIZE": 2 * 1024,  # 2KB regions
    "BLOCK_SIZE": 64,  # 64B cache blocks
    "PC_WIDTH": 16,  # bits from PC
    "ADDR_WIDTH": 16,  # bits from address
    "PHT_SIZE": 16 * 1024,  # Pattern History Table size in bytes
    "PHT_NUM_SETS": 256,  # Number of sets in PHT
    "PHT_ASSOC": 16,  # Associativity of PHT
    "FILTER_TABLE_SIZE": 64,  # Filter table entries
    "ACCUM_TABLE_SIZE": 128,  # Accumulation table entries
    "PREFETCH_DEGREE": 4,  # Max prefetches per trigger
    "MAX_OUTSTANDING": 32,  # Max outstanding prefetches
}

# Derived constants
PATTERN_LEN = CONFIG["PATTERN_LEN"]
BLOCK_SIZE = CONFIG["BLOCK_SIZE"]
REGION_SIZE = CONFIG["REGION_SIZE"]


# ----------------------
# Helper functions
# ----------------------
def region_number(block_number: int) -> int:
    """Get region number from block number."""
    return block_number // PATTERN_LEN


def region_offset(block_number: int) -> int:
    """Get offset within region from block number."""
    return block_number % PATTERN_LEN


def rotate_pattern(pattern: List[bool], n: int) -> List[bool]:
    """Rotate pattern by n positions (left if n>0, right if n<0)."""
    if not pattern:
        return pattern
    length = len(pattern)
    n = n % length
    return pattern[n:] + pattern[:n]


# ----------------------
# Cache Framework
# ----------------------
@dataclass
class CacheEntry:
    """Generic cache entry."""
    key: int
    valid: bool
    data: any


class LRUSetAssociativeCache:
    """LRU Set-Associative Cache base class."""

    def __init__(self, size: int, num_ways: int):
        assert size % num_ways == 0, "Size must be divisible by num_ways"
        self.size = size
        self.num_ways = num_ways
        self.num_sets = size // num_ways
        self.entries: List[List[Optional[CacheEntry]]] = [
            [None for _ in range(num_ways)] for _ in range(self.num_sets)
        ]
        self.lru: List[List[int]] = [
            [0 for _ in range(num_ways)] for _ in range(self.num_sets)
        ]
        self.time = 1

    def _get_index_tag(self, key: int) -> Tuple[int, int]:
        """Get set index and tag from key."""
        index = key % self.num_sets
        tag = key // self.num_sets
        return index, tag

    def find(self, key: int) -> Optional[any]:
        """Find entry by key."""
        index, tag = self._get_index_tag(key)
        set_entries = self.entries[index]

        for way in range(self.num_ways):
            entry = set_entries[way]
            if entry and entry.valid and entry.key == key:
                # Update LRU
                self.lru[index][way] = self.time
                self.time += 1
                return entry.data
        return None

    def insert(self, key: int, data: any) -> Optional[CacheEntry]:
        """Insert entry, return evicted entry if any."""
        index, tag = self._get_index_tag(key)
        set_entries = self.entries[index]

        # Check if already exists
        for way in range(self.num_ways):
            entry = set_entries[way]
            if entry and entry.valid and entry.key == key:
                # Update existing
                old_data = entry.data
                entry.data = data
                self.lru[index][way] = self.time
                self.time += 1
                return None

        # Find invalid slot
        for way in range(self.num_ways):
            if not set_entries[way] or not set_entries[way].valid:
                set_entries[way] = CacheEntry(key=key, valid=True, data=data)
                self.lru[index][way] = self.time
                self.time += 1
                return None

        # Evict LRU
        lru_way = min(range(self.num_ways), key=lambda w: self.lru[index][w])
        old_entry = set_entries[lru_way]
        set_entries[lru_way] = CacheEntry(key=key, valid=True, data=data)
        self.lru[index][lru_way] = self.time
        self.time += 1
        return old_entry

    def erase(self, key: int) -> Optional[any]:
        """Erase entry by key, return data if found."""
        index, tag = self._get_index_tag(key)
        set_entries = self.entries[index]

        for way in range(self.num_ways):
            entry = set_entries[way]
            if entry and entry.valid and entry.key == key:
                entry.valid = False
                data = entry.data
                set_entries[way] = None
                return data
        return None


class LRUFullyAssociativeCache:
    """LRU Fully-Associative Cache (special case of set-associative with 1 set)."""

    def __init__(self, size: int):
        self.size = size
        self.entries: Dict[int, any] = {}
        self.lru_order: List[int] = []

    def find(self, key: int) -> Optional[any]:
        """Find entry by key."""
        if key in self.entries:
            # Update LRU
            self.lru_order.remove(key)
            self.lru_order.append(key)
            return self.entries[key]
        return None

    def insert(self, key: int, data: any) -> None:
        """Insert entry."""
        if key in self.entries:
            # Update existing
            self.entries[key] = data
            self.lru_order.remove(key)
            self.lru_order.append(key)
        else:
            # Check if full
            if len(self.entries) >= self.size:
                # Evict LRU
                victim = self.lru_order.pop(0)
                del self.entries[victim]
                logger.debug(f"LRU FA Cache: evicted key {victim:#x}")

            self.entries[key] = data
            self.lru_order.append(key)

    def erase(self, key: int) -> Optional[any]:
        """Erase entry by key."""
        if key in self.entries:
            data = self.entries[key]
            del self.entries[key]
            self.lru_order.remove(key)
            return data
        return None


# ----------------------
# FilterTable
# ----------------------
@dataclass
class FilterTableData:
    """Data stored in filter table."""
    pc: int
    offset: int


class FilterTable:
    """Filter table tracks first access to regions."""

    def __init__(self, size: int):
        self.size = size
        self.cache = LRUFullyAssociativeCache(size)

    def find(self, region_num: int) -> Optional[FilterTableData]:
        """Find entry for region."""
        return self.cache.find(region_num)

    def insert(self, region_num: int, pc: int, offset: int) -> None:
        """Insert new filter entry."""
        data = FilterTableData(pc=pc, offset=offset)
        self.cache.insert(region_num, data)
        logger.debug(f"FilterTable: insert region {region_num:#x}, pc {pc:#x}, offset {offset}")

    def erase(self, region_num: int) -> Optional[FilterTableData]:
        """Remove entry for region."""
        return self.cache.erase(region_num)


# ----------------------
# AccumulationTable
# ----------------------
@dataclass
class AccumulationTableData:
    """Data stored in accumulation table."""
    pc: int
    offset: int
    pattern: List[bool]


class AccumulationTable:
    """Accumulation table tracks spatial patterns within regions."""

    def __init__(self, size: int, pattern_len: int):
        self.size = size
        self.pattern_len = pattern_len
        self.cache = LRUFullyAssociativeCache(size)

    def find(self, region_num: int) -> Optional[AccumulationTableData]:
        """Find entry for region."""
        return self.cache.find(region_num)

    def set_pattern(self, region_num: int, offset: int) -> bool:
        """Set bit in pattern, return True if entry exists."""
        data = self.cache.find(region_num)
        if data is None:
            return False

        if 0 <= offset < len(data.pattern):
            data.pattern[offset] = True
        return True

    def insert(self, filter_data: FilterTableData, region_num: int) -> Optional[AccumulationTableData]:
        """Insert from filter table entry, return evicted entry if any."""
        # Create new pattern with trigger bit set
        pattern = [False] * self.pattern_len
        if 0 <= filter_data.offset < self.pattern_len:
            pattern[filter_data.offset] = True

        data = AccumulationTableData(
            pc=filter_data.pc,
            offset=filter_data.offset,
            pattern=pattern
        )

        # Check if we need to evict
        evicted = None
        if len(self.cache.entries) >= self.size and region_num not in self.cache.entries:
            # Will evict LRU
            victim_key = self.cache.lru_order[0]
            evicted = self.cache.entries[victim_key]
            logger.debug(f"AccumulationTable: evicting region {victim_key:#x}")

        self.cache.insert(region_num, data)
        logger.debug(f"AccumulationTable: insert region {region_num:#x}")
        return evicted

    def erase(self, region_num: int) -> Optional[AccumulationTableData]:
        """Remove entry for region."""
        return self.cache.erase(region_num)


# ----------------------
# PatternHistoryTable
# ----------------------
@dataclass
class PatternHistoryTableEntry:
    """Entry in pattern history table."""
    pattern: List[bool]


class PatternHistoryTable:
    """Pattern history table with configurable PC/Address indexing."""

    def __init__(self, size: int, pattern_len: int, addr_width: int, pc_width: int, num_ways: int = 16):
        self.pattern_len = pattern_len
        self.addr_width = addr_width
        self.pc_width = pc_width
        self.num_sets = size // num_ways
        self.num_ways = num_ways

        # Calculate index length
        self.index_len = (self.num_sets - 1).bit_length() if self.num_sets > 0 else 0

        self.cache = LRUSetAssociativeCache(size, num_ways)

        logger.debug(f"PHT: addr_width={addr_width}, pc_width={pc_width}, "
                    f"num_sets={self.num_sets}, index_len={self.index_len}")

    def _build_key(self, pc: int, address: int) -> int:
        """Build key from PC and address."""
        # Mask inputs
        pc_masked = pc & ((1 << self.pc_width) - 1) if self.pc_width > 0 else 0
        addr_masked = address & ((1 << self.addr_width) - 1) if self.addr_width > 0 else 0

        # Combine
        key = (pc_masked << self.addr_width) | addr_masked

        # Hash index to distribute keys
        if self.index_len > 0:
            tag = key >> self.index_len
            while tag > 0:
                key ^= tag & ((1 << self.index_len) - 1)
                tag >>= self.index_len

        return key

    def find(self, pc: int, address: int) -> Optional[List[bool]]:
        """Find pattern for PC and address."""
        key = self._build_key(pc, address)
        entry = self.cache.find(key)
        if entry is None:
            return None

        # Rotate pattern based on offset
        offset = address % self.pattern_len
        rotated = rotate_pattern(entry.pattern, offset)
        return rotated

    def insert(self, pc: int, address: int, pattern: List[bool]) -> None:
        """Insert pattern for PC and address."""
        # Rotate pattern to be offset-independent
        offset = address % self.pattern_len
        rotated = rotate_pattern(pattern, -offset)

        key = self._build_key(pc, address)
        entry = PatternHistoryTableEntry(pattern=rotated)
        self.cache.insert(key, entry)
        logger.debug(f"PHT: insert key {key:#x} (pc={pc:#x}, addr={address:#x})")


# ----------------------
# BingoPrefetcher
# ----------------------
class BingoPrefetcher(PrefetchAlgorithm):
    """Bingo spatial prefetcher implementation."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize Bingo prefetcher with optional config overrides."""
        # Update config if provided
        if config:
            for k, v in config.items():
                if k in CONFIG:
                    CONFIG[k] = v

        # Initialize tables
        self.filter_table = FilterTable(CONFIG["FILTER_TABLE_SIZE"])
        self.accumulation_table = AccumulationTable(
            CONFIG["ACCUM_TABLE_SIZE"],
            CONFIG["PATTERN_LEN"]
        )

        # Four Pattern History Tables with different indexing
        pht_size = CONFIG["PHT_SIZE"] // 4  # Divide among 4 tables
        self.pht_pc_address = PatternHistoryTable(
            pht_size,
            CONFIG["PATTERN_LEN"],
            CONFIG["ADDR_WIDTH"],
            CONFIG["PC_WIDTH"],
            CONFIG["PHT_ASSOC"]
        )
        self.pht_pc_offset = PatternHistoryTable(
            pht_size,
            CONFIG["PATTERN_LEN"],
            (CONFIG["PATTERN_LEN"] - 1).bit_length(),  # log2(pattern_len) bits
            CONFIG["PC_WIDTH"],
            CONFIG["PHT_ASSOC"]
        )
        self.pht_address = PatternHistoryTable(
            pht_size,
            CONFIG["PATTERN_LEN"],
            CONFIG["ADDR_WIDTH"],
            0,  # No PC bits
            CONFIG["PHT_ASSOC"]
        )
        self.pht_pc = PatternHistoryTable(
            pht_size,
            CONFIG["PATTERN_LEN"],
            0,  # No address bits
            CONFIG["PC_WIDTH"],
            CONFIG["PHT_ASSOC"]
        )

        self.initialized = False
        logger.debug("BingoPrefetcher created")

    def init(self) -> None:
        """Initialize prefetcher state."""
        self.initialized = True
        logger.info("BingoPrefetcher initialized")

    def _find_in_pht(self, pc: int, address: int) -> List[bool]:
        """Find pattern in PHT tables (try in priority order)."""
        # Try PC+Address first (most specific)
        pattern = self.pht_pc_address.find(pc, address)
        if pattern and any(pattern):
            logger.debug(f"PHT hit: PC+Address")
            return pattern

        # Try PC+Offset
        pattern = self.pht_pc_offset.find(pc, address)
        if pattern and any(pattern):
            logger.debug(f"PHT hit: PC+Offset")
            return pattern

        # Try Address only
        pattern = self.pht_address.find(pc, address)
        if pattern and any(pattern):
            logger.debug(f"PHT hit: Address")
            return pattern

        # Try PC only
        pattern = self.pht_pc.find(pc, address)
        if pattern and any(pattern):
            logger.debug(f"PHT hit: PC")
            return pattern

        return []

    def _insert_in_pht(self, pc: int, address: int, pattern: List[bool]) -> None:
        """Insert pattern into all PHT tables."""
        self.pht_pc_address.insert(pc, address, pattern)
        self.pht_pc_offset.insert(pc, address, pattern)
        self.pht_address.insert(pc, address, pattern)
        self.pht_pc.insert(pc, address, pattern)
        logger.debug(f"Inserted pattern into all PHTs")

    def _commit_pattern(self, accum_data: AccumulationTableData, region_num: int) -> None:
        """Commit accumulated pattern to PHT."""
        # Calculate address from region and offset
        address = region_num * PATTERN_LEN + accum_data.offset

        # Insert into all PHT tables
        self._insert_in_pht(accum_data.pc, address, accum_data.pattern)
        logger.debug(f"Committed pattern: region {region_num:#x}, pc {accum_data.pc:#x}")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process memory access.

        Args:
            access: Memory access information
            prefetch_hit: Whether this access hit a prefetched line

        Returns:
            List of block addresses to prefetch
        """
        if not self.initialized:
            self.init()

        addr = access.address
        pc = access.pc

        # Convert to block number
        block_num = addr // BLOCK_SIZE
        reg_num = region_number(block_num)
        reg_off = region_offset(block_num)

        logger.debug(f"Access: addr={addr:#x}, pc={pc:#x}, block={block_num:#x}, "
                    f"region={reg_num:#x}, offset={reg_off}")

        # Try to add to accumulation table
        if self.accumulation_table.set_pattern(reg_num, reg_off):
            logger.debug(f"Updated accumulation table for region {reg_num:#x}")
            return []

        # Check filter table
        filter_entry = self.filter_table.find(reg_num)

        if filter_entry is None:
            # Trigger access - first access to this region
            self.filter_table.insert(reg_num, pc, reg_off)

            # Look up pattern in PHT
            pattern = self._find_in_pht(pc, block_num)

            if not pattern or not any(pattern):
                logger.debug(f"No pattern found in PHT")
                return []

            # Generate prefetches based on pattern
            prefetches = []
            base_block = reg_num * PATTERN_LEN

            for i in range(len(pattern)):
                if pattern[i] and len(prefetches) < CONFIG["PREFETCH_DEGREE"]:
                    pf_block = base_block + i
                    pf_addr = pf_block * BLOCK_SIZE
                    prefetches.append(pf_addr)

            logger.debug(f"Generated {len(prefetches)} prefetches from PHT pattern")
            return prefetches

        # Entry exists in filter table
        if filter_entry.offset != reg_off:
            # Different offset - promote to accumulation table
            self.filter_table.erase(reg_num)

            # Insert into accumulation table (may evict)
            evicted = self.accumulation_table.insert(filter_entry, reg_num)

            # Set the current offset
            self.accumulation_table.set_pattern(reg_num, reg_off)

            # If evicted, commit to PHT
            if evicted:
                # Find the region number of evicted entry
                for key in list(self.accumulation_table.cache.entries.keys()):
                    if self.accumulation_table.cache.entries[key] == evicted:
                        self._commit_pattern(evicted, key)
                        break

            logger.debug(f"Promoted region {reg_num:#x} to accumulation table")

        return []

    def eviction(self, block_number: int) -> None:
        """
        Handle block eviction from cache.

        Args:
            block_number: Block number being evicted
        """
        reg_num = region_number(block_number)

        # Remove from filter table
        self.filter_table.erase(reg_num)

        # Remove from accumulation table and commit if exists
        accum_data = self.accumulation_table.erase(reg_num)
        if accum_data:
            self._commit_pattern(accum_data, reg_num)
            logger.debug(f"Eviction: committed pattern for region {reg_num:#x}")

    def close(self) -> None:
        """Clean up prefetcher state."""
        # Commit all remaining accumulation table entries
        for reg_num in list(self.accumulation_table.cache.entries.keys()):
            accum_data = self.accumulation_table.cache.entries[reg_num]
            if accum_data:
                self._commit_pattern(accum_data, reg_num)

        # Reset tables
        self.filter_table = FilterTable(CONFIG["FILTER_TABLE_SIZE"])
        self.accumulation_table = AccumulationTable(
            CONFIG["ACCUM_TABLE_SIZE"],
            CONFIG["PATTERN_LEN"]
        )

        self.initialized = False
        logger.info("BingoPrefetcher closed")
