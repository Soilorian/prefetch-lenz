# file: prefetchlenz/prefetchingalgorithm/impl/f_tdc_prefetcher.py
from __future__ import annotations

import logging
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import List, Optional

from prefetchlenz.cache import Cache
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.f_tdc")

# === Configurable constants ===
PAGE_SIZE = 4096
VECTOR_BITS = 8
CHUNK_SIZE = PAGE_SIZE // VECTOR_BITS  # 512 bytes per footprint bit
BLOCK_SIZE = 64  # 64B cache block
BLOCKS_PER_CHUNK = CHUNK_SIZE // BLOCK_SIZE  # 8 blocks per footprint bit
TLB_ENTRIES_DEFAULT = 64  # number of in-TLB tracked pages (configurable)
WBQ_IMMEDIATE = True  # deterministic default for unit tests


@dataclass
class FootprintEntry:
    """
    Stored per-page footprint entry that simulates the paper's PTE-embedded footprint.

    Fields:
      - ref_vector: int (VECTOR_BITS bits) storing historical reference bits OR'ed over
                    previous sojourns. Bit i == 1 means chunk i was seen referenced.
      - valid_vector: int (VECTOR_BITS bits) storing which chunks were resident/valid
                      at the last resident period.
    """

    ref_vector: int = 0
    valid_vector: int = 0

    def merge_ref(self, vec: int):
        """Merge reference bits (OR semantics as paper describes for concurrent writes)."""
        logger.debug(
            f"Merging ref_vector: old=0b{self.ref_vector:0{VECTOR_BITS}b} new=0b{vec:0{VECTOR_BITS}b}"
        )
        self.ref_vector |= vec


class FootprintTracker:
    """
    Tracks a resident page's reference bits (per-chunk) while the page is present in cTLB/DRAM cache.

    Responsibilities:
      - set chunk reference bit on each access
      - expose current reference vector on eviction
      - keep track of which chunks are valid (resident) for coherence with in-cache state
    """

    def __init__(self, page_id: int):
        self.page_id = page_id
        # in-memory, per-residency reference vector (resets on new residency)
        self.ref_vector: int = 0
        # which chunks are valid (resident). This is updated when we selectively fetch blocks.
        self.valid_vector: int = 0

    def touch_address(self, addr: int):
        """Set reference bit for chunk corresponding to addr (page-local address)."""
        offset = addr % PAGE_SIZE
        block_index = offset // BLOCK_SIZE
        chunk_index = block_index // BLOCKS_PER_CHUNK
        bit = 1 << chunk_index
        old = self.ref_vector
        self.ref_vector |= bit
        logger.debug(
            f"Page {self.page_id} touch addr={addr} -> chunk={chunk_index} bit=0b{bit:0{VECTOR_BITS}b} "
            f"ref_vector 0b{old:0{VECTOR_BITS}b}->0b{self.ref_vector:0{VECTOR_BITS}b}"
        )

    def set_valid_chunks(self, bits: int):
        """Mark chunks that are resident/valid after a selective fill."""
        logger.debug(
            f"Page {self.page_id} set_valid_chunks 0b{bits:0{VECTOR_BITS}b} (old 0b{self.valid_vector:0{VECTOR_BITS}b})"
        )
        self.valid_vector = bits

    def export_and_clear(self) -> FootprintEntry:
        """
        Called on eviction: produce a FootprintEntry to be written back to PTE store and clear
        the in-residency ref vector (paper semantics: footprint is spilled to PTE on eviction).
        """
        entry = FootprintEntry(
            ref_vector=self.ref_vector, valid_vector=self.valid_vector
        )
        logger.info(
            f"Evicting page {self.page_id}: exporting footprint ref=0b{entry.ref_vector:0{VECTOR_BITS}b} "
            f"valid=0b{entry.valid_vector:0{VECTOR_BITS}b}"
        )
        # Clear trackers for next residency
        self.ref_vector = 0
        self.valid_vector = 0
        return entry


class FootprintWritebackQueue:
    """
    Simulated WBQ (write-back queue) for deterministic tests.

    - By default (WBQ_IMMEDIATE==True) writebacks happen synchronously during enqueue.
    - Optionally, one can configure a small delay or queue depth to simulate asynchronous behavior.
    """

    def __init__(
        self, store_cache: Cache, immediate: bool = WBQ_IMMEDIATE, max_delay: int = 0
    ):
        """
        store_cache: Cache instance to use as PTE-footprint store.
        immediate: if True, perform writeback immediately (deterministic).
        max_delay: if >0, emulate delayed writeback by storing requests in FIFO and apply
                   them when tick() is called enough times. (Deterministic behavior requires
                   you call tick() in tests.)
        """
        self.store_cache = store_cache
        self.immediate = immediate
        self.queue = deque()
        self.max_delay = max_delay
        self.ticks = 0

    def enqueue(self, page_id: int, entry: FootprintEntry):
        """
        Enqueue a writeback of 'entry' for page_id to the PTE store (Cache).
        If immediate -> apply merge right away (OR semantics).
        """
        logger.info(
            f"WBQ enqueue page={page_id} ref=0b{entry.ref_vector:0{VECTOR_BITS}b} "
            f"valid=0b{entry.valid_vector:0{VECTOR_BITS}b}"
        )
        if self.immediate:
            self._apply_writeback(page_id, entry)
        else:
            # push with a timestamp (tick-based delay)
            self.queue.append((self.ticks + self.max_delay, page_id, entry))

    def tick(self):
        """Progress simulated time: apply any queued writebacks whose time has arrived."""
        self.ticks += 1
        applied = 0
        while self.queue and self.queue[0][0] <= self.ticks:
            _, page_id, entry = self.queue.popleft()
            self._apply_writeback(page_id, entry)
            applied += 1
        if applied:
            logger.info(f"WBQ.tick applied {applied} writebacks (tick={self.ticks})")

    def _apply_writeback(self, page_id: int, entry: FootprintEntry):
        """
        Merge the incoming entry into the PTE store (Cache). If the PTE already exists, OR the
        ref_vector with stored one (paper says writes merge); valid_vector not merged (we store last-known).
        """
        existing: Optional[FootprintEntry] = self.store_cache.get(page_id)
        if existing is None:
            logger.debug(f"WBQ writeback adding new PTE entry for page {page_id}")
            # store a new FootprintEntry (copy)
            self.store_cache.put(
                page_id,
                FootprintEntry(
                    ref_vector=entry.ref_vector, valid_vector=entry.valid_vector
                ),
            )
        else:
            logger.debug(
                f"WBQ writeback merging PTE page {page_id}: existing_ref=0b{existing.ref_vector:0{VECTOR_BITS}b} "
                f"merge_with=0b{entry.ref_vector:0{VECTOR_BITS}b}"
            )
            existing.merge_ref(entry.ref_vector)
            # update valid_vector with latest known valid chunks (paper behavior: valid corresponds to
            # resident chunks at last fill; we overwrite with latest).
            existing.valid_vector = entry.valid_vector
            self.store_cache.put(page_id, existing)


class FootprintPrefetcher(PrefetchAlgorithm):
    """
    F-TDC (Footprint-driven Tagless DRAM Cache) Prefetcher implementation.

    Key mapping to the paper:
      - Tracks per-residency reference bits (FootprintTracker) for pages that are "resident" in the
        simulated cTLB/TDC. On eviction of a tracker (TLB entry eviction), the tracker exports
        its footprint and the WBQ writes it into a PTE-footprint store that is backed by the
        provided Cache instance.
      - On a page allocation (first access to a page when no tracker exists), the prefetcher
        consults the PTE-footprint store and issues selective fetches — one prefetch per 64-byte block
        inside each chunk whose bit is set. Returned prefetch addresses are canonical 64B-aligned
        block addresses (byte addresses aligned to BLOCK_SIZE).
    """

    def __init__(
        self,
        pte_store: Cache,
        tlb_entries: int = TLB_ENTRIES_DEFAULT,
        wbq_immediate: bool = True,
        wbq_delay: int = 0,
    ):
        """
        pte_store: Cache instance used to store FootprintEntry mapped by page_id.
        tlb_entries: number of simultaneous tracked pages (size of the simulated cTLB).
        wbq_immediate: if True, synchronous writeback from WBQ.
        wbq_delay: if >0 and wbq_immediate=False, simulated WBQ applies writebacks after delay ticks.
        """
        self.pte_store = pte_store
        self.tlb_capacity = tlb_entries
        self.trackers: "OrderedDict[int, FootprintTracker]" = OrderedDict()
        self.wbq = FootprintWritebackQueue(
            store_cache=pte_store, immediate=wbq_immediate, max_delay=wbq_delay
        )
        self.initialized = False

    # ---- PrefetchAlgorithm API ----
    def init(self):
        logger.info(
            f"F-TDC FootprintPrefetcher initializing: PAGE_SIZE={PAGE_SIZE} VECTOR_BITS={VECTOR_BITS}"
        )
        self.trackers.clear()
        self.initialized = True

    def close(self):
        # On close, writeback all resident trackers to PTE store
        logger.info(
            f"F-TDC FootprintPrefetcher closing: flushing {len(self.trackers)} trackers"
        )
        for page_id, tr in list(self.trackers.items()):
            entry = tr.export_and_clear()
            self.wbq.enqueue(page_id, entry)
        self.initialized = False

    def _page_id(self, addr: int) -> int:
        return addr // PAGE_SIZE

    def _is_new_allocation(self, page_id: int) -> bool:
        """
        Determine whether this access corresponds to a page allocation event:
        we treat an allocation as the moment we first see an access for a page not currently tracked.
        At that moment we consult the PTE store for a stored footprint to selectively fetch.
        """
        return page_id not in self.trackers

    def _evict_one_tracker_if_needed(self):
        """Evict the least-recently-used tracker when the TLB capacity is exceeded."""
        if len(self.trackers) > self.tlb_capacity:
            evict_page_id, evict_tr = self.trackers.popitem(last=False)
            logger.info(
                f"TLB capacity exceeded. Evicting page {evict_page_id} from tracker (LRU)"
            )
            entry = evict_tr.export_and_clear()
            self.wbq.enqueue(evict_page_id, entry)

    def _install_tracker_for_page(self, page_id: int):
        """
        Create a new tracker for the page. If the PTE store contains a footprint, issue selective
        prefetches for all bits set in stored ref_vector. Also set valid_vector for the tracker
        according to which chunks we prefetched.
        """
        tracker = FootprintTracker(page_id)
        # Insert into OrderedDict as most recently used (append at end)
        self.trackers[page_id] = tracker
        self.trackers.move_to_end(page_id, last=True)
        # Enforce capacity
        if len(self.trackers) > self.tlb_capacity:
            # evict LRU
            evict_page_id, evict_tr = self.trackers.popitem(last=False)
            logger.info(f"Evicting page {evict_page_id} due to new tracker install")
            entry = evict_tr.export_and_clear()
            self.wbq.enqueue(evict_page_id, entry)

        # Consult PTE store for stored footprint
        stored: Optional[FootprintEntry] = self.pte_store.get(page_id)
        prefetch_addrs: List[int] = []
        if stored:
            vec = stored.ref_vector
            if vec:
                logger.info(
                    f"On allocation of page {page_id} found stored footprint 0b{vec:0{VECTOR_BITS}b} -> issuing selective fill"
                )
            for bit_index in range(VECTOR_BITS):
                if (vec >> bit_index) & 1:
                    # compute chunk base address (page-local)
                    chunk_base = bit_index * CHUNK_SIZE
                    # Fill each BLOCK inside chunk
                    for b in range(BLOCKS_PER_CHUNK):
                        block_offset = chunk_base + (b * BLOCK_SIZE)
                        global_addr = page_id * PAGE_SIZE + block_offset
                        prefetch_addrs.append(global_addr)
                    # Mark as valid in tracker
                    tracker.valid_vector |= 1 << bit_index
        else:
            logger.debug(f"On allocation of page {page_id} no stored footprint found")
        return tracker, prefetch_addrs

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Called for every memory access. Updates per-page reference bits and, when an access
        touches a page for which we did not have a tracker, treats that as an allocation event:
        installs a tracker and issues selective-prefetch addresses based on the PTE store.
        """
        if not self.initialized:
            raise RuntimeError(
                "Prefetcher not initialized. Call init() before progress()."
            )

        addr = getattr(access, "address", None)
        if addr is None:
            return []

        page_id = self._page_id(addr)
        prefetches: List[int] = []

        # Allocation check: if page is not tracked, this is an allocation -> consult PTE store
        if self._is_new_allocation(page_id):
            tracker, addrs = self._install_tracker_for_page(page_id)
            # We already inserted tracker in install. Return prefetch addresses to caller to actually issue.
            prefetches.extend(addrs)
            if addrs:
                logger.info(
                    f"Issued {len(addrs)} selective prefetches for page {page_id}"
                )
        else:
            # mark as MRU (touch)
            tracker = self.trackers[page_id]
            self.trackers.move_to_end(page_id, last=True)

        # Update tracker with the access's reference bit
        tracker.touch_address(addr)

        # Enforce TLB capacity eviction if needed (safe check)
        self._evict_one_tracker_if_needed()

        return prefetches

    # Hook to allow tests to force WBQ ticks (when not immediate)
    def tick_wbq(self):
        self.wbq.tick()
