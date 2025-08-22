from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.sos")


@dataclass
class StoreOrderedStreamer(PrefetchAlgorithm):
    """
    Store-Ordered Streaming (SOS) Prefetcher
    ----------------------------------------
    Practical software implementation based on the idea of *store-ordered
    streaming*: record each thread's stream of store targets in program order;
    when another thread (a consumer) accesses an address that appeared in some
    producer's store stream, prefetch subsequent store targets from that
    producer's stream for the consumer. :contentReference[oaicite:1]{index=1}

    Assumptions about MemoryAccess (adapt if your types differ):
      - address: int               # byte address of access
      - pc: int | None             # optional instruction id
      - tid: int | None            # optional thread id (defaults to 0 if absent)
      - is_store: bool | None      # True if the access is a store
      - is_cache_miss: bool | None # if available, used to trigger on misses

    Behavior:
      - Maintain per-thread store streams as sequences of *line addresses*
        (address // line_size) with de-duplication of consecutive repeats.
      - Maintain an index mapping from line -> (producer_tid, last_seq_idx).
      - On a triggering consumer access (default: cache miss & not a prefetch
        hit), if the accessed line exists in the producer map, issue prefetches
        for the next K lines *after* that index from the same producer stream.
      - Track 'outstanding' prefetches to avoid duplicates.
    """

    # ---- Tunables ----
    line_size: int = 64  # bytes per cache line (mapping addr -> line)
    stream_window: int = 4096  # max entries kept per producer stream
    prefetch_degree: int = 8  # how many next store lines to fetch
    trigger_on_miss_only: bool = True  # only issue on miss events
    max_producers_tracked: Optional[int] = None  # None = unlimited producers
    dedup_consecutive: bool = True  # collapse immediate repeats in a stream

    # ---- Internal state (init() sets these) ----
    # Per-producer (tid) store stream: deque of line addrs
    producer_streams: Dict[int, Deque[int]] = field(default_factory=dict, init=False)
    # For each producer, last seen index per line in its stream
    producer_line_last_idx: Dict[int, Dict[int, int]] = field(
        default_factory=dict, init=False
    )
    # Global reverse map: line -> (producer_tid, last_idx_in_that_producer)
    line_to_producer_idx: Dict[int, Tuple[int, int]] = field(
        default_factory=dict, init=False
    )
    # Outstanding prefetch lines to avoid duplicates
    outstanding_lines: Set[int] = field(default_factory=set, init=False)

    def init(self):
        self.producer_streams = {}
        self.producer_line_last_idx = {}
        self.line_to_producer_idx = {}
        self.outstanding_lines = set()
        logger.debug("StoreOrderedStreamer initialized.")

    def close(self):
        logger.debug(
            "StoreOrderedStreamer closed. Producers tracked=%d",
            len(self.producer_streams),
        )

    # ---- Core API ----
    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        addr = getattr(access, "address", None)
        if addr is None:
            return []

        tid = getattr(access, "tid", 0) or 0
        is_store = bool(getattr(access, "is_store", False))
        line = addr // self.line_size

        # 1) If this is a store, append to the producer's store stream.
        if is_store:
            self._record_store(tid, line)
            return []  # no prefetch on producer's own store by default

        # 2) Decide if this consumer access triggers prefetching.
        if not self._should_trigger(access, prefetch_hit):
            return []

        # 3) If this line exists in some producer stream, prefetch successors.
        prod_info = self.line_to_producer_idx.get(line)
        if not prod_info:
            return []

        producer_tid, idx = prod_info
        candidates = self._successors_from_producer(
            producer_tid, idx, self.prefetch_degree
        )

        # 4) Filter duplicates/outstanding and return as *byte addresses* (convert from line)
        issued: List[int] = []
        for l in candidates:
            if l not in self.outstanding_lines and l != line:
                self.outstanding_lines.add(l)
                issued.append(l * self.line_size)

        if issued:
            logger.debug(
                "SOS: consumer tid=%s touched L%d -> prefetch from producer tid=%s: %s",
                tid,
                line,
                producer_tid,
                [x // self.line_size for x in issued],
            )

        return issued

    # Optional hooks your cache/sim can call:
    def prefetch_completed(self, addr: int):
        self.outstanding_lines.discard(addr // self.line_size)

    def notify_prefetch_hit(self, addr: int):
        self.outstanding_lines.discard(addr // self.line_size)

    # ---- Internals ----
    def _should_trigger(self, access: MemoryAccess, prefetch_hit: bool) -> bool:
        if not self.trigger_on_miss_only:
            return True
        is_miss = getattr(access, "is_cache_miss", None)
        if is_miss is not None:
            return bool(is_miss) and not bool(prefetch_hit)
        # fallback: trigger when not a prefetch hit
        return not bool(prefetch_hit)

    def _record_store(self, producer_tid: int, line: int):
        # Respect producer count cap (optional)
        if producer_tid not in self.producer_streams:
            if (
                self.max_producers_tracked is not None
                and len(self.producer_streams) >= self.max_producers_tracked
            ):
                # Simple policy: don't track new producers if at capacity
                return
            self.producer_streams[producer_tid] = deque(maxlen=self.stream_window)
            self.producer_line_last_idx[producer_tid] = {}

        stream = self.producer_streams[producer_tid]
        # Optional de-dup of consecutive identical lines to keep stream crisp
        if self.dedup_consecutive and stream and stream[-1] == line:
            # update last index mapping but don't append a new entry
            last_idx = self.producer_line_last_idx[producer_tid].get(
                line, len(stream) - 1
            )
            self.producer_line_last_idx[producer_tid][line] = last_idx  # unchanged
            self.line_to_producer_idx[line] = (producer_tid, last_idx)
            return

        # Append and update indexes
        stream.append(line)
        idx = len(stream) - 1
        self.producer_line_last_idx[producer_tid][line] = idx
        self.line_to_producer_idx[line] = (producer_tid, idx)

    def _successors_from_producer(
        self, producer_tid: int, idx: int, k: int
    ) -> List[int]:
        """
        Return up to k successor *line* addresses after position idx in producer stream.
        """
        stream = self.producer_streams.get(producer_tid)
        if not stream:
            return []
        out: List[int] = []
        # stream is a deque; convert indices relative to its head
        # We may have wrapped (maxlen), so indices map to current window tail.
        base = max(0, len(stream) - self.stream_window)
        # Effective index in current deque
        start = idx + 1 - base
        if start < 0:
            # The reference idx may have been evicted from the deque; start from head
            start = 0
        # Walk forward
        for i in range(start, min(start + k, len(stream))):
            out.append(stream[i])
        return out
