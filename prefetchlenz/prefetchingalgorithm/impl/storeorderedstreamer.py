# sos_prefetcher.py
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from prefetchlenz.prefetchingalgorithm.access.storeorderedmemoryaccess import (
    StoreOrderedMemoryAccess,
)

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.sos")
logger.addHandler(logging.NullHandler())


@dataclass
class StoreOrderedStreamer:
    """
    Store-Ordered Streaming (SOS) Prefetcher.

    Records per-producer store streams (line addresses). When a consumer access
    triggers (by default a cache miss and not a prefetch hit), and the accessed
    line appears in some producer's stream, SOS issues prefetches for the next
    `prefetch_degree` lines that the producer stored after that line.

    Returned prefetches are byte addresses (line * line_size).
    """

    # Tunables
    line_size: int = 64
    stream_window: int = 4096
    prefetch_degree: int = 8
    max_producers_tracked: Optional[int] = None
    dedup_consecutive: bool = True

    # Internal state (initialized in init)
    producer_streams: Dict[int, Deque[int]] = field(default_factory=dict, init=False)
    producer_line_last_idx: Dict[int, Dict[int, int]] = field(
        default_factory=dict, init=False
    )
    line_to_producer_idx: Dict[int, Tuple[int, int]] = field(
        default_factory=dict, init=False
    )
    outstanding_lines: Set[int] = field(default_factory=set, init=False)

    def init(self):
        """Reset all internal state."""
        self.producer_streams = {}
        self.producer_line_last_idx = {}
        self.line_to_producer_idx = {}
        self.outstanding_lines = set()
        logger.debug("SOS: initialized")

    def close(self):
        """Clean up state for end of simulation."""
        logger.debug(
            "SOS: closed. producers=%d outstanding=%d",
            len(self.producer_streams),
            len(self.outstanding_lines),
        )

    # Main API used by simulator
    def progress(
        self, access: StoreOrderedMemoryAccess, prefetch_hit: bool
    ) -> List[int]:
        """
        Process a StoreOrderedMemoryAccess and return list of byte-address prefetches.

        - if access.isWrite: record into producer stream
        - else (read): possibly trigger prefetch based on recorded producer streams
        """
        addr = access.address
        tid = int(access.tid) if access.tid is not None else 0
        is_store = access.isWrite
        line = addr // self.line_size

        if is_store:
            self._record_store(tid, line)
            return []

        prod_info = self.line_to_producer_idx.get(line)
        if not prod_info:
            return []

        producer_tid, idx = prod_info
        succ_lines = self._successors_from_producer(
            producer_tid, idx, self.prefetch_degree
        )

        issued: List[int] = []
        for l in succ_lines:
            if l not in self.outstanding_lines and l != line:
                self.outstanding_lines.add(l)
                issued.append(l * self.line_size)

        if issued:
            logger.debug(
                "SOS: consumer tid=%s line=%d -> prefetch producer=%s lines=%s",
                tid,
                line,
                producer_tid,
                [l for l in succ_lines[: len(issued)]],
            )

        return issued

    def prefetch_completed(self, addr: int):
        """Called when a prefetch completes (install)."""
        self.outstanding_lines.discard(addr // self.line_size)

    def notify_prefetch_hit(self, addr: int):
        """Called when a prefetched line was used (hit)."""
        self.outstanding_lines.discard(addr // self.line_size)

    def _record_store(self, producer_tid: int, line: int):
        """Append a store-line to the producer's stream and update index maps."""
        if producer_tid not in self.producer_streams:
            if (
                self.max_producers_tracked is not None
                and len(self.producer_streams) >= self.max_producers_tracked
            ):
                # do not track new producers when at capacity
                logger.debug(
                    "SOS: not tracking new producer %d (capacity reached)", producer_tid
                )
                return
            self.producer_streams[producer_tid] = deque(maxlen=self.stream_window)
            self.producer_line_last_idx[producer_tid] = {}

        stream = self.producer_streams[producer_tid]

        if self.dedup_consecutive and stream and stream[-1] == line:
            # update mappings but don't append duplicate
            last_idx = self.producer_line_last_idx[producer_tid].get(
                line, len(stream) - 1
            )
            self.producer_line_last_idx[producer_tid][line] = last_idx
            self.line_to_producer_idx[line] = (producer_tid, last_idx)
            return

        stream.append(line)
        idx = len(stream) - 1
        self.producer_line_last_idx[producer_tid][line] = idx
        self.line_to_producer_idx[line] = (producer_tid, idx)
        logger.debug(
            "SOS: recorded store producer=%d line=%d idx=%d", producer_tid, line, idx
        )

    def _successors_from_producer(
        self, producer_tid: int, idx: int, k: int
    ) -> List[int]:
        """Return up to k successor line addresses after position idx in producer stream."""
        stream = self.producer_streams.get(producer_tid)
        if not stream:
            return []
        out: List[int] = []
        # compute base index of current deque window
        # deque has at most stream_window elements; idx is relative to when added
        # if idx older than present window, treat as starting from head
        if idx < 0:
            return []
        # If idx >= len(stream) then stale; clamp start to 0
        if idx >= len(stream):
            start = 0
        else:
            start = idx + 1
        for i in range(start, min(start + k, len(stream))):
            out.append(stream[i])
        return out
