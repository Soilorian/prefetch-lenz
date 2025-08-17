from __future__ import annotations

import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.event_triggered")


@dataclass
class EventTriggeredPrefetcher(PrefetchAlgorithm):
    """
    Event-triggered programmable prefetcher (practical implementation inspired by:
    'An Event-Triggered Programmable Prefetcher for Irregular Workloads'). See
    the project's PDF for conceptual details. :contentReference[oaicite:1]{index=1}

    This implementation:
      - Keeps per-PC recent address history (deque of ints).
      - Learns short delta patterns (sequence -> next-delta frequency table).
      - Detects simple constant stride and performs stride prefetch if found.
      - Triggers prefetch generation on events (configurable; default: on misses).
      - Avoids duplicates via outstanding_prefetches set.
      - Returns list[int] addresses to prefetch when `progress(...)` is called.

    Assumptions (adapt if your system differs):
      - MemoryAccess has attributes:
          - address: int  (accessed address)
          - pc: int or any hashable id for the instruction performing the access
          - is_write: Optional[bool] (ignored by default)
          - is_cache_miss: Optional[bool] (if present, we may use it)
      - The caller will execute the returned addresses as prefetch requests
        (e.g., insert them into cache or schedule them in the simulation).
    """

    # --- tunable parameters ---
    history_size: int = 8  # per-PC history length (addresses)
    pattern_length: int = 3  # number of most recent deltas used as pattern key
    min_pattern_support: int = 2  # minimum support to consider a learned pattern
    prefetch_degree: int = 4  # how many next addresses to prefetch
    prefetch_distance: int = 1  # multiplier for distance when using stride
    trigger_on_miss_only: bool = True  # if True, only generate prefetches on misses

    # --- internal state (initialized in init()) ---
    pc_history: Dict[int, Deque[int]] = field(default_factory=dict, init=False)
    # key: tuple(deltas) -> Counter(next_delta -> counts)
    pattern_table: Dict[Tuple[int, ...], Counter] = field(
        default_factory=dict, init=False
    )
    outstanding_prefetches: Set[int] = field(default_factory=set, init=False)
    enabled: bool = field(default=True, init=False)

    def init(self):
        """Prepare/clear internal state before simulation begins."""
        self.pc_history = {}
        self.pattern_table = {}
        self.outstanding_prefetches = set()
        self.enabled = True
        logger.debug("EventTriggeredPrefetcher initialized.")

    def close(self):
        """Cleanup state after simulation ends (no-op here)."""
        self.enabled = False
        logger.debug("EventTriggeredPrefetcher closed.")

    # ---- helper methods ----
    def _get_history(self, pc) -> Deque[int]:
        if pc not in self.pc_history:
            self.pc_history[pc] = deque(maxlen=self.history_size)
        return self.pc_history[pc]

    def _update_history_and_train(self, pc: int, addr: int):
        """
        Update per-PC history and train the pattern table from recent deltas.
        We produce (pattern_length) recent deltas as key and record next delta.
        """
        hist = self._get_history(pc)
        if len(hist) > 0:
            prev = hist[-1]
            delta = addr - prev
        else:
            delta = None

        # update history
        hist.append(addr)

        # train pattern table if we have enough deltas (pattern_length + 1 addresses)
        if len(hist) >= self.pattern_length + 1:
            # compute latest deltas sequence (pattern_length deltas) and the next_delta
            # Eg addresses [a0, a1, a2, a3] with pattern_length=3 generates pattern (a1-a0, a2-a1, a3-a2)
            addrs = list(hist)
            deltas = [addrs[i + 1] - addrs[i] for i in range(len(addrs) - 1)]
            # pattern is the last `pattern_length` deltas minus the most recent delta
            pattern = tuple(
                deltas[-self.pattern_length - 1 : -1]
            )  # older deltas that predict the last delta
            next_delta = deltas[-1]
            if pattern not in self.pattern_table:
                self.pattern_table[pattern] = Counter()
            self.pattern_table[pattern][next_delta] += 1
            logger.debug(
                "Trained pattern %s -> %d (count=%d)",
                pattern,
                next_delta,
                self.pattern_table[pattern][next_delta],
            )

    def _detect_stride(self, pc: int) -> Optional[int]:
        """
        Detect a simple constant stride for this PC using the last few history addresses.
        If a consistent stride exists (all deltas equal), return that stride; else None.
        """
        hist = self._get_history(pc)
        if len(hist) < 2:
            return None
        addrs = list(hist)
        deltas = [addrs[i + 1] - addrs[i] for i in range(len(addrs) - 1)]
        # check if all deltas equal
        first = deltas[0]
        if all(d == first for d in deltas):
            logger.debug("Detected constant stride %d for PC %s", first, pc)
            return first
        return None

    def _predict_by_stride(self, last_addr: int, stride: int, degree: int) -> List[int]:
        """Predict next addresses by constant stride."""
        return [
            last_addr + (i + 1) * stride * self.prefetch_distance for i in range(degree)
        ]

    def _predict_by_pattern(self, pc: int, last_addr: int, degree: int) -> List[int]:
        """
        Use the pattern table to predict next-deltas repeatedly (autoregressive).
        We look up the most probable next delta for the current pattern, then append it
        and slide the pattern forward to predict further steps.
        """
        hist = self._get_history(pc)
        if len(hist) < self.pattern_length + 1:
            return []

        # Build current pattern: last pattern_length deltas before the last delta
        addrs = list(hist)
        deltas = [addrs[i + 1] - addrs[i] for i in range(len(addrs) - 1)]
        # current pattern is the last pattern_length deltas (the most recent pattern)
        pattern = tuple(deltas[-self.pattern_length :])
        predictions = []
        cur_addr = last_addr
        cur_pattern = list(pattern)

        for _ in range(degree):
            key = tuple(cur_pattern)
            if key not in self.pattern_table:
                break
            # choose most common next delta
            next_delta, count = self.pattern_table[key].most_common(1)[0]
            # require minimal support
            if count < self.min_pattern_support:
                break
            cur_addr = cur_addr + next_delta
            predictions.append(cur_addr)
            # slide pattern: drop oldest, append predicted delta
            cur_pattern = cur_pattern[1:] + [next_delta]
        return predictions

    # ---- core API ----
    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Called for each memory access. Returns a list of addresses to prefetch
        (may be empty). This function:
          - updates history & pattern-table training
          - if an event triggers (by default: a miss that was not a prefetch_hit),
            attempts pattern-based prediction; if that fails, falls back to stride
            prediction.
        """
        if not self.enabled:
            return []

        addr = getattr(access, "address", None)
        pc = getattr(access, "pc", None)

        if addr is None:
            logger.warning(
                "Access missing 'address' attribute; skipping prefetch logic."
            )
            return []

        # Train/update history first
        self._update_history_and_train(pc, addr)

        # Determine whether to trigger on this access
        # Event selection heuristics (adapt if your MemoryAccess provides extra fields)
        is_prefetch_hit = bool(prefetch_hit)
        is_miss = getattr(access, "is_cache_miss", None)
        # If simulator provides explicit miss info, prefer it; otherwise treat non-prefetch as miss candidate
        triggered = False
        if self.trigger_on_miss_only:
            if is_miss is not None:
                triggered = is_miss and (not is_prefetch_hit)
            else:
                # fallback heuristic: if access was not a prefetch hit then treat as trigger
                triggered = not is_prefetch_hit
        else:
            triggered = True

        if not triggered:
            return []

        # Avoid generating prefetches for already-outstanding addresses.
        last_addr = self._get_history(pc)[-1]

        # Try pattern-based prediction first
        predictions: List[int] = self._predict_by_pattern(
            pc, last_addr, self.prefetch_degree
        )

        # If pattern-based failed, try stride detection
        if not predictions:
            stride = self._detect_stride(pc)
            if stride is not None:
                predictions = self._predict_by_stride(
                    last_addr, stride, self.prefetch_degree
                )

        # If still nothing, fall back to naive sequential prefetch (next-line style)
        if not predictions:
            predictions = [
                last_addr + (i + 1) * self.prefetch_distance
                for i in range(self.prefetch_degree)
            ]

        # Filter duplicates and outstanding prefetches
        to_issue: List[int] = []
        for a in predictions:
            if a not in self.outstanding_prefetches:
                to_issue.append(a)
                self.outstanding_prefetches.add(a)

        logger.debug(
            "For PC %s and addr %d -> issuing prefetches: %s", pc, addr, to_issue
        )
        return to_issue

    # Optional helper for simulator to notify when a prefetch has completed (so we can clear outstanding set)
    def prefetch_completed(self, addr: int, was_inserted: bool = True):
        """
        Call this when a prefetch completes (successful or not). If the prefetch inserted a block,
        it may be considered a 'prefetched into cache' entry. We always remove outstanding marker.
        """
        if addr in self.outstanding_prefetches:
            self.outstanding_prefetches.remove(addr)

    # Optional: if cache calls back on prefetch hits, propagate to internal state
    def notify_prefetch_hit(self, addr: int):
        """If the cache reports a prefetch hit, remove outstanding marker."""
        self.prefetch_completed(addr, was_inserted=True)
