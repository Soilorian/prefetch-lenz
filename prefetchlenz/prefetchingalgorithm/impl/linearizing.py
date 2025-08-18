# file: prefetchers/linearizing_prefetcher.py
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.linearizing")


@dataclass
class LinearizingPrefetcher(PrefetchAlgorithm):
    """
    Linearizing Prefetcher (based on Irregular Stream Buffers - MICRO'13).
    Core principle:
      - Observe pointer-chasing accesses: addr -> loaded_value
      - Linearize these into a logical stream (like a virtual array)
      - Prefetch ahead in logical order using a conventional lookahead

    Assumptions about MemoryAccess:
      - access.address : int
      - access.pc : int
      - access.loaded_value : Optional[int] (valid when this is a load returning a pointer)
      - access.is_cache_miss : Optional[bool]
    """

    prefetch_degree: int = 4  # how many steps ahead to prefetch per stream
    stream_history_len: int = 64  # max addresses to keep per stream
    trigger_on_miss_only: bool = True

    # internal state
    streams: Dict[int, Deque[int]] = field(default_factory=dict, init=False)
    addr_to_stream: Dict[int, int] = field(default_factory=dict, init=False)
    next_stream_id: int = field(default=0, init=False)
    outstanding: Set[int] = field(default_factory=set, init=False)

    def init(self):
        self.streams.clear()
        self.addr_to_stream.clear()
        self.next_stream_id = 0
        self.outstanding.clear()
        logger.debug("LinearizingPrefetcher initialized.")

    def close(self):
        logger.debug("LinearizingPrefetcher closed. Streams=%d", len(self.streams))

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        addr = getattr(access, "address", None)
        if addr is None:
            return []

        triggered = self._should_trigger(access, prefetch_hit)
        prefetches: List[int] = []

        # Check if this access extends a known stream
        stream_id = self.addr_to_stream.get(addr, None)

        # If the load produced a pointer, create/extend stream
        next_ptr = getattr(access, "loaded_value", None)
        if next_ptr is not None:
            if stream_id is None:
                # Start a new stream with this address
                stream_id = self.next_stream_id
                self.next_stream_id += 1
                self.streams[stream_id] = deque(maxlen=self.stream_history_len)
                logger.debug("Started new stream %d at addr=%d", stream_id, addr)
            # extend stream
            self.streams[stream_id].append(next_ptr)
            self.addr_to_stream[next_ptr] = stream_id

        # If triggered, issue prefetches for this stream
        if triggered and stream_id is not None:
            prefetches = self._issue_prefetches(stream_id)

        return prefetches

    def _should_trigger(self, access: MemoryAccess, prefetch_hit: bool) -> bool:
        if not self.trigger_on_miss_only:
            return True
        is_miss = getattr(access, "is_cache_miss", None)
        if is_miss is not None:
            return is_miss and not prefetch_hit
        return not prefetch_hit

    def _issue_prefetches(self, stream_id: int) -> List[int]:
        stream = self.streams.get(stream_id)
        if not stream:
            return []
        # Take last element of stream as "current pointer"
        current = stream[-1]
        issued: List[int] = []
        # Issue prefetches for next prefetch_degree elements if we know them
        for i in range(min(self.prefetch_degree, len(stream))):
            cand = stream[
                -1 - i
            ]  # look backward since we enqueue newly discovered nodes
            if cand not in self.outstanding:
                issued.append(cand)
                self.outstanding.add(cand)
        if issued:
            logger.debug("Stream %d issuing prefetches: %s", stream_id, issued)
        return issued

    def prefetch_completed(self, addr: int):
        self.outstanding.discard(addr)

    def notify_prefetch_hit(self, addr: int):
        self.outstanding.discard(addr)
