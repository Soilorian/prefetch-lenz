"""
TEMPO (Translation-Enabled Memory Prefetching Optimizations) - Python implementation

Paper: "Translation-Triggered Prefetching" by Abhishek Bhattacharjee
Conference: ASPLOS '17

Overview:
- Exploits virtual memory translation to boost memory performance
- When a memory reference triggers a DRAM page table lookup, TEMPO prefetches
  the data that the translation points to
- Non-speculative: always prefetches the correct address
- Key insight: 98%+ of DRAM page table lookups are followed by DRAM data accesses

Key Components:
- TLB (Translation Lookaside Buffer): Caches virtual-to-physical translations
- Page Table Walker: Simulates page table walks on TLB misses
- Translation-triggered prefetching: Prefetches data when translations are cold
- Metrics tracking: TLB hits/misses, prefetch accuracy, coverage

Architecture:
- x86-64 style 4-level page tables (L4→L3→L2→L1)
- Support for 4KB base pages and 2MB superpages
- Simulates memory hierarchy with translation caching

References:
- ASPLOS '17 paper: "Translation-Triggered Prefetching"
- Adapted to Python simulation framework
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.impl.tempo")

# ----------------------
# Configuration
# ----------------------
CONFIG = {
    # TLB Configuration
    "TLB_SIZE": 64,                      # Data TLB entries
    "TLB_ASSOC": 4,                      # TLB associativity

    # Memory Configuration
    "PAGE_SIZE_4KB": 4096,               # 4KB base page size
    "PAGE_SIZE_2MB": 2 * 1024 * 1024,    # 2MB superpage size
    "CACHE_LINE_SIZE": 64,               # 64-byte cache lines
    "SUPPORT_2MB_PAGES": True,           # Enable 2MB superpage support

    # Page Table Configuration
    "PT_LEVELS": 4,                      # x86-64 4-level page tables
    "PT_CACHE_SIZE": 32,                 # MMU cache for upper PT levels

    # Prefetch Configuration
    "PREFETCH_DEGREE": 1,                # Number of lines to prefetch per trigger
    "ENABLE_PREFETCHING": True,          # Master switch for prefetching

    # Cold Translation Detection
    "COLD_TRANSLATION_THRESHOLD": 20,    # Access count threshold for "cold" translations
    "TRACK_TRANSLATION_REUSE": True,     # Track translation reuse for cold detection

    # Metrics
    "TRACK_METRICS": True,               # Enable detailed metrics tracking
}


# ----------------------
# Data Structures
# ----------------------
@dataclass
class TLBEntry:
    """TLB entry storing virtual-to-physical translation."""
    virtual_page: int = 0
    physical_page: int = 0
    page_size: int = CONFIG["PAGE_SIZE_4KB"]  # 4KB or 2MB
    valid: bool = False
    access_count: int = 0  # Track access frequency for cold detection

    def is_superpage(self) -> bool:
        """Check if this is a 2MB superpage."""
        return self.page_size == CONFIG["PAGE_SIZE_2MB"]


@dataclass
class PageTableEntry:
    """Page table entry in the simulated page table hierarchy."""
    virtual_page: int = 0
    physical_page: int = 0
    level: int = 1  # PT level (1=L1/leaf, 2=L2, 3=L3, 4=L4)
    valid: bool = True
    access_count: int = 0  # Track how many times this translation is accessed
    is_superpage: bool = False  # True for 2MB pages at L2 level


@dataclass
class TempoMetrics:
    """Metrics for TEMPO prefetcher performance."""
    total_accesses: int = 0
    tlb_hits: int = 0
    tlb_misses: int = 0

    page_table_walks: int = 0
    dram_page_table_accesses: int = 0  # Cold translations requiring DRAM

    prefetches_issued: int = 0
    prefetch_hits: int = 0  # Prefetched data was actually accessed
    prefetch_coverage: float = 0.0  # % of DRAM replay accesses that were prefetched

    dram_replay_accesses: int = 0  # DRAM accesses for data after translation
    dram_replay_hits: int = 0  # Replay accesses that hit prefetched data

    superpages_used: int = 0
    base_pages_used: int = 0


# ----------------------
# TLB Implementation
# ----------------------
class TLB:
    """Translation Lookaside Buffer - caches virtual-to-physical translations."""

    def __init__(self, size: int, associativity: int):
        self.size = size
        self.assoc = associativity
        self.num_sets = size // associativity

        # Set-associative structure: [set_index][way]
        self.entries: List[List[Optional[TLBEntry]]] = [
            [None for _ in range(self.assoc)] for _ in range(self.num_sets)
        ]

        # LRU tracking: [set_index][way] = timestamp
        self.lru: List[List[int]] = [
            [0 for _ in range(self.assoc)] for _ in range(self.num_sets)
        ]
        self.timestamp = 0

        logger.debug(f"TLB initialized: {size} entries, {associativity}-way, {self.num_sets} sets")

    def _get_set_index(self, virtual_page: int) -> int:
        """Get set index from virtual page number."""
        return virtual_page % self.num_sets

    def lookup(self, virtual_page: int, page_size: int = CONFIG["PAGE_SIZE_4KB"]) -> Optional[TLBEntry]:
        """
        Lookup virtual page in TLB.

        Args:
            virtual_page: Virtual page number
            page_size: Page size (4KB or 2MB)

        Returns:
            TLBEntry if hit, None if miss
        """
        set_idx = self._get_set_index(virtual_page)

        for way in range(self.assoc):
            entry = self.entries[set_idx][way]
            if entry and entry.valid and entry.virtual_page == virtual_page:
                # TLB hit - update LRU
                self.lru[set_idx][way] = self.timestamp
                self.timestamp += 1
                entry.access_count += 1
                logger.debug(f"TLB hit: vpn={virtual_page:#x} -> ppn={entry.physical_page:#x}")
                return entry

        logger.debug(f"TLB miss: vpn={virtual_page:#x}")
        return None

    def insert(self, virtual_page: int, physical_page: int,
               page_size: int = CONFIG["PAGE_SIZE_4KB"]) -> None:
        """
        Insert translation into TLB.

        Args:
            virtual_page: Virtual page number
            physical_page: Physical page number
            page_size: Page size (4KB or 2MB)
        """
        set_idx = self._get_set_index(virtual_page)

        # Check if already exists
        for way in range(self.assoc):
            entry = self.entries[set_idx][way]
            if entry and entry.valid and entry.virtual_page == virtual_page:
                # Update existing entry
                entry.physical_page = physical_page
                entry.page_size = page_size
                entry.access_count = 0  # Reset on refill
                self.lru[set_idx][way] = self.timestamp
                self.timestamp += 1
                logger.debug(f"TLB update: vpn={virtual_page:#x} -> ppn={physical_page:#x}")
                return

        # Find invalid entry
        for way in range(self.assoc):
            if not self.entries[set_idx][way] or not self.entries[set_idx][way].valid:
                self.entries[set_idx][way] = TLBEntry(
                    virtual_page=virtual_page,
                    physical_page=physical_page,
                    page_size=page_size,
                    valid=True,
                    access_count=0
                )
                self.lru[set_idx][way] = self.timestamp
                self.timestamp += 1
                logger.debug(f"TLB insert: vpn={virtual_page:#x} -> ppn={physical_page:#x}")
                return

        # Evict LRU entry
        lru_way = min(range(self.assoc), key=lambda w: self.lru[set_idx][w])
        old_entry = self.entries[set_idx][lru_way]
        if old_entry:
            logger.debug(f"TLB evict: vpn={old_entry.virtual_page:#x}")

        self.entries[set_idx][lru_way] = TLBEntry(
            virtual_page=virtual_page,
            physical_page=physical_page,
            page_size=page_size,
            valid=True,
            access_count=0
        )
        self.lru[set_idx][lru_way] = self.timestamp
        self.timestamp += 1
        logger.debug(f"TLB insert (evicted): vpn={virtual_page:#x} -> ppn={physical_page:#x}")

    def invalidate(self, virtual_page: int) -> None:
        """Invalidate TLB entry for virtual page."""
        set_idx = self._get_set_index(virtual_page)
        for way in range(self.assoc):
            entry = self.entries[set_idx][way]
            if entry and entry.valid and entry.virtual_page == virtual_page:
                entry.valid = False
                logger.debug(f"TLB invalidate: vpn={virtual_page:#x}")
                return


# ----------------------
# Page Table Walker
# ----------------------
class PageTableWalker:
    """
    Simulates page table walks on TLB misses.
    Models x86-64 4-level page tables (L4→L3→L2→L1).
    """

    def __init__(self):
        # Simulated page table: virtual_page -> PageTableEntry
        self.page_table: Dict[int, PageTableEntry] = {}

        # MMU cache for upper-level page tables (L4, L3, L2)
        self.mmu_cache: Dict[int, PageTableEntry] = {}
        self.mmu_cache_max_size = CONFIG["PT_CACHE_SIZE"]

        # Track which translations are "cold" (require DRAM access)
        self.translation_access_counts: Dict[int, int] = {}

        logger.debug("PageTableWalker initialized")

    def walk(self, virtual_page: int) -> Tuple[Optional[PageTableEntry], bool]:
        """
        Perform page table walk for virtual page.

        Args:
            virtual_page: Virtual page number to translate

        Returns:
            Tuple of (PageTableEntry or None, is_cold_translation)
            is_cold_translation: True if this translation required DRAM access
        """
        # Check if translation exists
        if virtual_page not in self.page_table:
            # Allocate new translation (simplified identity mapping for simulation)
            physical_page = self._allocate_physical_page(virtual_page)

            # Decide if this should be a superpage (simple heuristic: aligned pages)
            is_superpage = (
                CONFIG["SUPPORT_2MB_PAGES"] and
                virtual_page % (CONFIG["PAGE_SIZE_2MB"] // CONFIG["PAGE_SIZE_4KB"]) == 0
            )

            page_size = CONFIG["PAGE_SIZE_2MB"] if is_superpage else CONFIG["PAGE_SIZE_4KB"]

            entry = PageTableEntry(
                virtual_page=virtual_page,
                physical_page=physical_page,
                level=1,  # Leaf PT entry
                valid=True,
                access_count=0,
                is_superpage=is_superpage
            )
            self.page_table[virtual_page] = entry
            logger.debug(f"PTW: Allocated new translation vpn={virtual_page:#x} -> ppn={physical_page:#x}")

        entry = self.page_table[virtual_page]
        entry.access_count += 1

        # Determine if this is a "cold" translation (would access DRAM)
        # Cold if: access_count is low (translation not frequently used)
        is_cold = entry.access_count <= CONFIG["COLD_TRANSLATION_THRESHOLD"]

        if is_cold:
            logger.debug(f"PTW: Cold translation detected vpn={virtual_page:#x} (count={entry.access_count})")

        return entry, is_cold

    def _allocate_physical_page(self, virtual_page: int) -> int:
        """
        Allocate physical page for virtual page.
        For simulation, use a simple mapping function.
        """
        # Use hash function to create pseudo-random but deterministic mapping
        return (virtual_page * 2654435761) % (2**20)  # Limit to reasonable physical address space


# ----------------------
# TEMPO Prefetcher
# ----------------------
class TempoPrefetcher(PrefetchAlgorithm):
    """
    TEMPO (Translation-Enabled Memory Prefetching Optimizations) implementation.

    Key idea: When a memory access causes a TLB miss and the resulting page table
    walk accesses DRAM (cold translation), prefetch the actual data that the
    translation points to, since it will likely also be in DRAM.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize TEMPO prefetcher with optional config overrides."""
        # Update config if provided
        if config:
            for k, v in config.items():
                if k in CONFIG:
                    CONFIG[k] = v

        # Initialize TLB
        self.tlb = TLB(CONFIG["TLB_SIZE"], CONFIG["TLB_ASSOC"])

        # Initialize page table walker
        self.page_table_walker = PageTableWalker()

        # Track prefetched addresses (for measuring accuracy)
        self.prefetched_addresses: Set[int] = set()

        # Track recent accesses (for detecting DRAM replay accesses)
        self.recent_translations: Dict[int, int] = {}  # virtual_page -> physical_page
        self.recent_translation_times: Dict[int, int] = {}  # virtual_page -> timestamp

        # Metrics
        self.metrics = TempoMetrics()

        # State
        self.initialized = False
        self.access_count = 0

        logger.debug("TempoPrefetcher created")

    def init(self) -> None:
        """Initialize prefetcher state."""
        self.initialized = True
        logger.info("TempoPrefetcher initialized")

    def _get_virtual_page(self, address: int, page_size: int = CONFIG["PAGE_SIZE_4KB"]) -> int:
        """Extract virtual page number from address."""
        return address // page_size

    def _get_page_offset(self, address: int, page_size: int = CONFIG["PAGE_SIZE_4KB"]) -> int:
        """Extract offset within page from address."""
        return address % page_size

    def _is_recently_translated(self, virtual_page: int, window: int = 100) -> bool:
        """Check if this virtual page was recently translated (within window accesses)."""
        if virtual_page in self.recent_translation_times:
            return (self.access_count - self.recent_translation_times[virtual_page]) < window
        return False

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process memory access and generate translation-triggered prefetches.

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
        self.access_count += 1

        self.metrics.total_accesses += 1

        # Extract virtual page number and offset
        virtual_page = self._get_virtual_page(addr)
        page_offset = self._get_page_offset(addr)

        logger.debug(f"Access #{self.access_count}: addr={addr:#x}, pc={pc:#x}, "
                    f"vpn={virtual_page:#x}, offset={page_offset}")

        # Check if this access hit a prefetch
        if addr in self.prefetched_addresses:
            self.metrics.prefetch_hits += 1
            self.prefetched_addresses.remove(addr)
            logger.debug(f"Prefetch hit: addr={addr:#x}")

        # Step 1: TLB lookup
        tlb_entry = self.tlb.lookup(virtual_page)

        if tlb_entry is not None:
            # TLB hit - no prefetch needed
            self.metrics.tlb_hits += 1
            logger.debug(f"TLB hit - no prefetch")

            # Track if this is a superpage
            if tlb_entry.is_superpage():
                self.metrics.superpages_used += 1
            else:
                self.metrics.base_pages_used += 1

            return []

        # Step 2: TLB miss - perform page table walk
        self.metrics.tlb_misses += 1
        self.metrics.page_table_walks += 1
        logger.debug(f"TLB miss - initiating page table walk")

        pt_entry, is_cold = self.page_table_walker.walk(virtual_page)

        if pt_entry is None:
            logger.warning(f"Page table walk failed for vpn={virtual_page:#x}")
            return []

        # Fill TLB with translation
        page_size = CONFIG["PAGE_SIZE_2MB"] if pt_entry.is_superpage else CONFIG["PAGE_SIZE_4KB"]
        self.tlb.insert(virtual_page, pt_entry.physical_page, page_size)

        # Track recent translation
        self.recent_translations[virtual_page] = pt_entry.physical_page
        self.recent_translation_times[virtual_page] = self.access_count

        # Step 3: Translation-triggered prefetching
        prefetches = []

        if is_cold and CONFIG["ENABLE_PREFETCHING"]:
            # This translation required DRAM access (cold)
            # The data it points to is likely also in DRAM
            self.metrics.dram_page_table_accesses += 1

            # Calculate the physical address that will be accessed after translation
            physical_address = pt_entry.physical_page * page_size + page_offset

            # Generate prefetch for the actual data
            prefetches.append(physical_address)
            self.prefetched_addresses.add(physical_address)
            self.metrics.prefetches_issued += 1

            logger.info(f"TEMPO prefetch triggered: "
                       f"vpn={virtual_page:#x} -> ppn={pt_entry.physical_page:#x}, "
                       f"prefetch_addr={physical_address:#x}")

            # Optionally prefetch additional cache lines (prefetch degree > 1)
            for i in range(1, CONFIG["PREFETCH_DEGREE"]):
                next_line_addr = physical_address + (i * CONFIG["CACHE_LINE_SIZE"])
                # Stay within page boundary
                if self._get_page_offset(next_line_addr, page_size) < page_size:
                    prefetches.append(next_line_addr)
                    self.prefetched_addresses.add(next_line_addr)
                    self.metrics.prefetches_issued += 1

        # Track if this would be a DRAM replay access
        # (access after page table walk, data also cold)
        if is_cold:
            self.metrics.dram_replay_accesses += 1

        return prefetches

    def close(self) -> None:
        """Clean up and print final metrics."""
        if CONFIG["TRACK_METRICS"]:
            self._print_metrics()

        # Reset state
        self.tlb = TLB(CONFIG["TLB_SIZE"], CONFIG["TLB_ASSOC"])
        self.page_table_walker = PageTableWalker()
        self.prefetched_addresses.clear()
        self.recent_translations.clear()
        self.recent_translation_times.clear()
        self.metrics = TempoMetrics()
        self.initialized = False
        self.access_count = 0

        logger.info("TempoPrefetcher closed")

    def _print_metrics(self) -> None:
        """Print detailed performance metrics."""
        m = self.metrics

        logger.info("=" * 60)
        logger.info("TEMPO Prefetcher Metrics")
        logger.info("=" * 60)

        logger.info(f"Total accesses: {m.total_accesses}")
        if m.total_accesses > 0:
            logger.info(f"TLB hits: {m.tlb_hits} ({m.tlb_hits/m.total_accesses*100:.2f}%)")
            logger.info(f"TLB misses: {m.tlb_misses} ({m.tlb_misses/m.total_accesses*100:.2f}%)")
        else:
            logger.info(f"TLB hits: {m.tlb_hits}")
            logger.info(f"TLB misses: {m.tlb_misses}")
        logger.info(f"Page table walks: {m.page_table_walks}")
        logger.info(f"DRAM page table accesses (cold): {m.dram_page_table_accesses}")

        logger.info("")
        logger.info(f"Prefetches issued: {m.prefetches_issued}")
        logger.info(f"Prefetch hits: {m.prefetch_hits}")
        if m.prefetches_issued > 0:
            accuracy = m.prefetch_hits / m.prefetches_issued * 100
            logger.info(f"Prefetch accuracy: {accuracy:.2f}%")

        logger.info("")
        logger.info(f"DRAM replay accesses: {m.dram_replay_accesses}")
        if m.dram_replay_accesses > 0:
            coverage = m.prefetch_hits / m.dram_replay_accesses * 100
            logger.info(f"Prefetch coverage: {coverage:.2f}%")

        logger.info("")
        logger.info(f"Superpages used: {m.superpages_used}")
        logger.info(f"Base pages used: {m.base_pages_used}")

        logger.info("=" * 60)
