"""
Linearizing Irregular Memory Accesses for Improved by Jain et al.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

from prefetchlenz.prefetchingalgorithm.access.linearmemoryaccess import (
    LinearMemoryAccess,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.linearizing_hw")

# --- NOTE / CITATION ---
# This implementation follows the high-level hardware ideas described in the
# "Linearizing Irregular Memory Accesses for Improved ..." (Irregular Stream Buffers)
# paper that you uploaded. It implements a software-side emulation of the
# hardware stream buffer/tag-table + a lightweight adaptive controller that
# inspects prefetch / miss counters and adjusts aggressiveness.
# See paper for original hardware design and evaluation. :contentReference[oaicite:1]{index=1}
# ----------------------


@dataclass
class Stream:
    """Emulates a small hardware stream buffer entry.

    Attributes:
        id: Stream identifier.
        buf: Deque of predicted addresses in logical order (front == earliest).
        confidence: Small saturating counter used to decide whether to issue prefetches.
        last_issued_index: Index into the buffer of the last element we issued
            (so we issue forward). -1 means nothing issued yet.
    """

    id: int
    buf: Deque[int] = field(default_factory=deque)
    confidence: int = 0
    last_issued_index: int = -1  # -1 means nothing issued yet.

    def push(self, addr: int, maxlen: int) -> None:
        """Push address into stream buffer.

        Args:
            addr: Address to push.
            maxlen: Maximum buffer length.
        """
        if addr not in self.buf:
            if len(self.buf) >= maxlen:
                # Mimic small hardware stream buffer by popping left (earliest).
                self.buf.popleft()
                # Shift last_issued_index left by one if > -1.
                if self.last_issued_index >= 0:
                    self.last_issued_index = max(-1, self.last_issued_index - 1)
            self.buf.append(addr)

    def available_after_last_issued(self) -> List[int]:
        """Return addresses after last issued index in logical order.

        Returns:
            List of addresses after the last issued index.
        """
        start = self.last_issued_index + 1
        return list(self.buf)[start:]


@dataclass
class LinearizingPrefetcher(PrefetchAlgorithm):
    """Linearizing prefetcher with hardware-like stream buffers and adaptive control.

    This class extends the original software linearizer by emulating the hardware
    prefetching components described in the Irregular Stream Buffers paper:
        - Small per-stream buffer (stream buffer entry).
        - Tag table (to avoid duplicate prefetches).
        - Outstanding prefetch tracking.
        - Confidence counters to avoid issuing prefetches for noisy streams.
        - Adaptive controller that tweaks prefetch_degree using observed counters.

    Note: This is a software emulation / runtime autotuner. To use real hardware
    counters replace the internal counters (cache_misses, prefetch_hits, ...) with
    readings from perf/PAPI/etc.

    Reference: original uploaded paper.
    """

    # tunables (defaults)
    prefetch_degree: int = 4  # how many steps ahead to prefetch per stream
    stream_history_len: int = 64  # how many addresses to keep per stream buffer
    trigger_on_miss_only: bool = True

    # hardware-emulation parameters
    confidence_threshold: int = 2  # minimal confidence before aggressively prefetching
    max_confidence: int = 7
    adapt_interval: int = 1024  # number of progress() calls between adaptation steps
    adapt_low_hit_ratio: float = 0.1
    adapt_high_hit_ratio: float = 0.7

    # internal state (initialized in init())
    streams: Dict[int, Stream] = field(default_factory=dict, init=False)
    addr_to_stream: Dict[int, int] = field(default_factory=dict, init=False)
    next_stream_id: int = field(default=0, init=False)

    # outstanding prefetch/tag table
    outstanding: Set[int] = field(default_factory=set, init=False)
    tag_table: Set[int] = field(
        default_factory=set, init=False
    )  # tracks addresses recently prefetched/targeted

    # counters for adaptation (simulated or updated externally)
    total_progress_calls: int = field(default=0, init=False)
    hw_prefetch_requests: int = field(default=0, init=False)
    hw_prefetch_hits: int = field(default=0, init=False)
    hw_cache_misses: int = field(default=0, init=False)

    def init(self) -> None:
        """Initialize or reset prefetcher state."""
        self.streams.clear()
        self.addr_to_stream.clear()
        self.next_stream_id = 0
        self.outstanding.clear()
        self.tag_table.clear()
        self.total_progress_calls = 0
        self.hw_prefetch_requests = 0
        self.hw_prefetch_hits = 0
        self.hw_cache_misses = 0
        logger.debug("LinearizingPrefetcher (HW-emulation) initialized.")

    def close(self) -> None:
        """Clean up prefetcher state."""
        logger.info(
            "LinearizingPrefetcher closed. Streams=%d outstanding=%d prefetch_reqs=%d hits=%d misses=%d",
            len(self.streams),
            len(self.outstanding),
            self.hw_prefetch_requests,
            self.hw_prefetch_hits,
            self.hw_cache_misses,
        )

    # ---------- public runtime hooks ----------

    def progress(self, access: LinearMemoryAccess, prefetch_hit: bool) -> List[int]:
        """Process a memory access and return prefetch requests.

        Called for every memory access seen by the monitor.
        Returns a list of addresses we request to prefetch (these are requested).
        The caller should call `prefetch_completed(addr)` when a prefetch finishes,
        and `notify_prefetch_hit(addr)` when a prefetch results in a hit.

        Args:
            access: Memory access event.
            prefetch_hit: Whether this access was a prefetch hit.

        Returns:
            List of addresses to prefetch.
        """
        addr = access.address
        self.total_progress_calls += 1

        triggered = True if not self.trigger_on_miss_only else not prefetch_hit
        prefetches: List[int] = []

        # Map this access address to a stream if one exists
        stream_id = self.addr_to_stream.get(addr, None)

        # If the load produced a pointer, extend or create a stream buffer
        next_ptr = access.loaded_pointer
        if next_ptr is not None:
            if stream_id is None:
                # start new stream entry
                stream_id = self.next_stream_id
                self.next_stream_id += 1
                s = Stream(id=stream_id)
                self.streams[stream_id] = s
                logger.debug("Started HW-like stream %d at addr=%d", stream_id, addr)
            else:
                s = self.streams[stream_id]
            # push pointer into the stream buffer (hardware would insert predicted addresses)
            s.push(next_ptr, maxlen=self.stream_history_len)
            self.addr_to_stream[next_ptr] = stream_id

            # increase confidence (saturating)
            if s.confidence < self.max_confidence:
                s.confidence += 1

        # if this access was a cache miss (hardware info), increment counter for adaptation
        if not prefetch_hit:
            self.hw_cache_misses += 1

        # If triggered, issue prefetches similar to hardware ISB: forward from last_issued
        if triggered and stream_id is not None:
            prefetches = self._issue_prefetches_hw(stream_id)

        # Periodically adapt degree using counters
        if (
            self.total_progress_calls % self.adapt_interval == 0
            and self.total_progress_calls > 0
        ):
            self._adapt_prefetching_policy()

        return prefetches

    def prefetch_completed(self, addr: int) -> None:
        """Call after a prefetch has completed (line filled into cache).

        Args:
            addr: Address that was prefetched.
        """
        if addr in self.outstanding:
            self.outstanding.discard(addr)
        # When a prefetch completes and it was tracked in tag table, we can remove the tag entry.
        self.tag_table.discard(addr)
        logger.debug(
            "Prefetch completed for addr=%d outstanding=%d", addr, len(self.outstanding)
        )

    def notify_prefetch_hit(self, addr: int) -> None:
        """Call when an earlier prefetch resulted in a hit (observed by monitor).

        Args:
            addr: Address that was prefetched and hit.
        """
        if addr in self.outstanding:
            # Treat as hit: clear outstanding and increase hit counter.
            self.outstanding.discard(addr)
        if addr in self.tag_table:
            self.tag_table.discard(addr)
        self.hw_prefetch_hits += 1
        logger.debug(
            "Prefetch hit for addr=%d total_hits=%d", addr, self.hw_prefetch_hits
        )

    def _issue_prefetches_hw(self, stream_id: int) -> List[int]:
        """Issue prefetches for a stream following a hardware-like policy.

        Only issues forward addresses after last_issued_index.
        Requires a minimal confidence unless we are forced.
        Consults tag_table and outstanding to avoid duplicates.
        Updates hw_prefetch_requests counter.

        Args:
            stream_id: Stream identifier.

        Returns:
            List of addresses to prefetch.
        """
        s = self.streams.get(stream_id)
        if not s:
            return []
        # If confidence is low, be conservative
        if s.confidence < self.confidence_threshold:
            logger.debug(
                "Stream %d low confidence (%d) — skipping prefetch issuance.",
                stream_id,
                s.confidence,
            )
            return []

        issued: List[int] = []
        forward = s.available_after_last_issued()
        # issue up to prefetch_degree addresses in-order
        for idx, cand in enumerate(forward[: self.prefetch_degree]):
            # skip if we already have it outstanding or already in tag_table (recently prefetched)
            if cand in self.outstanding or cand in self.tag_table:
                logger.debug(
                    "Stream %d cand %d skipped (already outstanding/tagged)",
                    stream_id,
                    cand,
                )
                # still advance last_issued_index as hardware would (since we consumed it logically)
                s.last_issued_index += 1
                continue
            # issue
            issued.append(cand)
            self.outstanding.add(cand)
            self.tag_table.add(cand)
            s.last_issued_index += 1
            self.hw_prefetch_requests += 1
            logger.debug(
                "Stream %d issued prefetch for addr=%d (last_issued_index=%d)",
                stream_id,
                cand,
                s.last_issued_index,
            )
        if issued:
            logger.info("Stream %d issuing prefetches: %s", stream_id, issued)
        return issued

    def _adapt_prefetching_policy(self) -> None:
        """Adapt prefetching policy based on hit ratio.

        Computes prefetch hit ratio = hits / requests.
        If hit ratio too low, reduces degree (reduce pollution).
        If hit ratio high and misses still high, increases degree.
        """
        if self.hw_prefetch_requests == 0:
            logger.debug("No prefetch requests, skipping adaptation.")
            return

        hit_ratio = self.hw_prefetch_hits / float(self.hw_prefetch_requests)
        miss_count = self.hw_cache_misses

        logger.info(
            "Adaptation snapshot: requests=%d hits=%d hit_ratio=%.3f cache_misses=%d degree=%d",
            self.hw_prefetch_requests,
            self.hw_prefetch_hits,
            hit_ratio,
            miss_count,
            self.prefetch_degree,
        )

        # very simple controller:
        if hit_ratio < self.adapt_low_hit_ratio and self.prefetch_degree > 1:
            # too many useless prefetches -> scale back
            old = self.prefetch_degree
            self.prefetch_degree = max(1, self.prefetch_degree - 1)
            logger.info(
                "Adaptation: low hit ratio %.3f -> decreasing degree %d->%d",
                hit_ratio,
                old,
                self.prefetch_degree,
            )
        elif hit_ratio > self.adapt_high_hit_ratio and miss_count > 0:
            # prefetches are effective and misses remain -> be more aggressive
            old = self.prefetch_degree
            self.prefetch_degree = min(32, self.prefetch_degree + 1)
            logger.info(
                "Adaptation: high hit ratio %.3f -> increasing degree %d->%d",
                hit_ratio,
                old,
                self.prefetch_degree,
            )

        # reset counters after adaptation to avoid stale influence
        self.hw_prefetch_requests = 0
        self.hw_prefetch_hits = 0
        self.hw_cache_misses = 0
