"""
Temporal Memory Streaming by Ferdman et al.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Set, Tuple

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.temporal_streaming")
logger.addHandler(logging.NullHandler())


@dataclass
class TemporalMemoryStreamingPrefetcher(PrefetchAlgorithm):
    """
    Simplified Temporal Memory Streaming (TMS) prefetcher.

    Behavior summary
    ----------------
    - Records a circular sequence of recent miss addresses (CMOB).
    - Maintains a directory mapping address -> most recent CMOB seq_id.
    - When an access hits an address that appears in the directory and that
      CMOB entry is still resident, prefetch the next `prefetch_degree`
      addresses from the CMOB (replaying the recorded stream).
    - Returned prefetches are byte addresses (same units as MemoryAccess.address).

    Notes
    -----
    - This implementation does not require cache-hit/miss signals from the
      framework. It treats every input access as an observed miss to the CMOB
      (you can adapt this easily if you later receive miss-only events).
    - The CMOB stores full addresses (not tags) for simplicity.
    """

    # Configurables
    cmob_capacity: int = 1024  # number of entries stored in CMOB (circular)
    prefetch_degree: int = 8  # how many subsequent addresses to replay / prefetch
    dedup_outstanding: bool = True  # avoid re-issuing outstanding prefetches
    line_size: int = 64  # optional: outstanding tracked per-line if >1

    # Internal state (initialized by init)
    cmob: Deque[Tuple[int, int]] = field(
        default_factory=deque, init=False
    )  # (seq_id, addr)
    addr_to_seq: Dict[int, int] = field(
        default_factory=dict, init=False
    )  # addr -> seq_id
    next_seq_id: int = field(
        default=0, init=False
    )  # monotonic sequence id for CMOB entries
    outstanding_lines: Set[int] = field(
        default_factory=set, init=False
    )  # tracked as line numbers

    def init(self):
        """Reset CMOB and directory for a fresh run."""
        self.cmob = deque(maxlen=self.cmob_capacity)
        self.addr_to_seq.clear()
        self.next_seq_id = 0
        self.outstanding_lines.clear()
        logger.info(
            "TMS: initialized (cmob_capacity=%d prefetch_degree=%d)",
            self.cmob_capacity,
            self.prefetch_degree,
        )

    def close(self):
        logger.info(
            "TMS: closed; cmob_entries=%d outstanding=%d",
            len(self.cmob),
            len(self.outstanding_lines),
        )
        self.cmob.clear()
        self.addr_to_seq.clear()
        self.outstanding_lines.clear()

    # ---------------- core ----------------
    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process an access:
          1) Look up the address in directory; if found and resident in CMOB, replay
             successors and return prefetch addresses.
          2) Append the access to CMOB and update directory to point to the new seq_id.

        Returns a list of byte addresses to prefetch.
        """
        addr = int(access.address)
        preds: List[int] = []

        # 1) lookup directory -> seq_id and try to replay
        seq_id = self.addr_to_seq.get(addr)
        if seq_id is not None:
            # compute base_seq (oldest seq in current CMOB) if CMOB non-empty
            if self.cmob:
                base_seq = self.cmob[0][0]
                last_seq = self.cmob[-1][0]
                # check if seq_id still within CMOB window
                if base_seq <= seq_id <= last_seq:
                    start_index = seq_id - base_seq  # index into deque
                    # collect successors
                    for offset in range(1, self.prefetch_degree + 1):
                        idx = start_index + offset
                        if idx >= len(self.cmob):
                            break
                        succ_seq, succ_addr = self.cmob[idx]
                        line = succ_addr // self.line_size
                        if self.dedup_outstanding and line in self.outstanding_lines:
                            continue
                        preds.append(succ_addr)
                        self.outstanding_lines.add(line)
                else:
                    # seq_id stale/evicted; no replay
                    logger.debug(
                        "TMS: directory seq %d for addr %x stale (base %d last %d)",
                        seq_id,
                        addr,
                        base_seq,
                        last_seq,
                    )
            else:
                # empty CMOB; nothing to replay
                pass

        # 2) append current access to CMOB and update directory
        seq = self.next_seq_id
        self.next_seq_id += 1
        self.cmob.append((seq, addr))
        self.addr_to_seq[addr] = seq  # point directory to most recent occurrence

        # When deque drops an old entry by overflow, remove any stale addr->seq if it equals that seq.
        # deque with maxlen automatically discards from left; detect and clean accordingly.
        # We can inspect leftmost if its seq < next_seq - cmob_capacity.
        if len(self.cmob) == self.cmob_capacity:
            # leftmost may be soon evicted on next append; perform cleanup proactively:
            leftmost_seq = self.cmob[0][0]
            # iterate shallow copy of addr_to_seq and remove mappings with seq < leftmost_seq
            # (keeps directory small). This is cheap because cmob_capacity is moderate.
            stale = [a for a, s in self.addr_to_seq.items() if s < leftmost_seq]
            for a in stale:
                del self.addr_to_seq[a]

        return preds

    # hooks
    def prefetch_completed(self, addr: int):
        """Call when a prefetch completes/installs in cache. Clears outstanding tracking."""
        line = addr // self.line_size
        self.outstanding_lines.discard(line)

    def notify_prefetch_hit(self, addr: int):
        """Call when a prefetched line is used. Clears outstanding tracking for that line."""
        line = addr // self.line_size
        self.outstanding_lines.discard(line)
