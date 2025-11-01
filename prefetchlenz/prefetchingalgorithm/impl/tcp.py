"""
Tag Correlating Prefetcher (TCP)

Algorithm: TCP Tag Correlating Prefetchers by Hu et al.

This prefetcher operates on cache line tags rather than full addresses. It learns
correlations between consecutive line tags and predicts successor tags. It supports
chaining predictions to prefetch multiple levels ahead and can optionally use per-PC
context or periodic decay.

Key Components:
- CorrelationEntry: Frequency table tracking successors for a tag with integer counts
- CorrelationTable: Global tag-to-successors mapping with capacity management
- TCPPrefetchAlgorithm: Main prefetcher implementing tag-based correlation

How it works:
1. Convert addresses to line tags by dropping low-order bits
2. Train correlation table by strengthening link from previous tag to current tag
3. Boost correlation weight on prefetch hits to reinforce successful predictions
4. Predict successors of current tag, optionally chaining predictions multiple levels
5. Support per-PC context or global correlation tracking
6. Optional periodic decay to age out stale correlations
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger(__name__)


# -----------------------------
# TCP data structures
# -----------------------------
@dataclass
class CorrelationEntry:
    """Successor frequency table for a tag (line address).

    Tracks up to max_succ most frequent successors with integer counts.
    """

    max_succ: int
    freq: Counter = field(default_factory=Counter)
    total: int = 0

    def update(self, nxt: int, w: int = 1) -> None:
        """Update correlation count and evict low-count entries if table is full."""
        self.freq[nxt] += w
        self.total += w
        if len(self.freq) > self.max_succ:
            victim, _ = min(self.freq.items(), key=lambda kv: kv[1])
            del self.freq[victim]

    def predict(self, tolerance: float, top_k: int) -> List[int]:
        """Return up to top_k successors within tolerance of the best probability."""
        if not self.freq or self.total <= 0:
            return []
        items = [(s, c / float(self.total)) for s, c in self.freq.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        best = items[0][1]
        keep: List[int] = []
        for s, p in items:
            if p >= best - tolerance:
                keep.append(s)
            else:
                break
            if len(keep) >= top_k:
                break
        return keep


@dataclass
class CorrelationTable:
    """Tag-to-successors mapping with hard capacity limit.

    On overflow, evicts the entry with smallest total count.
    """

    capacity: int
    max_succ_per_tag: int
    table: Dict[int, CorrelationEntry] = field(default_factory=dict)

    def _admit(self, tag: int) -> CorrelationEntry:
        if tag in self.table:
            return self.table[tag]
        if len(self.table) >= self.capacity:
            victim = min(self.table.items(), key=lambda kv: kv[1].total)[0]
            logger.debug(
                "CT: evicting tag=%d total=%d", victim, self.table[victim].total
            )
            del self.table[victim]
        ce = CorrelationEntry(max_succ=self.max_succ_per_tag)
        self.table[tag] = ce
        return ce

    def update(self, tag: int, successor: int, weight: int = 1) -> None:
        ce = self._admit(tag)
        ce.update(successor, w=weight)

    def predict(self, tag: int, tolerance: float, top_k: int) -> List[int]:
        ce = self.table.get(tag)
        if not ce:
            return []
        return ce.predict(tolerance=tolerance, top_k=top_k)


# -----------------------------
# TCP Prefetcher
# -----------------------------
@dataclass
class TCPPrefetchAlgorithm(PrefetchAlgorithm):
    """Tag Correlating Prefetcher (TCP).

    Operates on line tags (address >> line_size_bits). Strengthens correlation
    between consecutive tags and predicts successors. Supports chaining predictions
    and optional periodic decay.
    """

    # Geometry / interpretation
    line_size_bits: int = 6  # 64B lines by default
    use_per_pc_context: bool = False

    # Correlation-table sizing
    table_capacity: int = 1 << 12
    max_succ_per_tag: int = 8

    # Prediction behavior
    top_k: int = 4
    tolerance: float = 0.05
    chain_depth: int = 2
    avoid_duplicates: bool = True

    # Training behavior
    prefetch_hit_boost: int = 2
    base_weight: int = 1
    decay_every: int = 0
    decay_factor: float = 0.99

    # Internal state (set in init)
    corr: CorrelationTable = field(init=False)
    _prev_tag_global: Optional[int] = field(default=None, init=False)
    _prev_tag_per_pc: Dict[int, Optional[int]] = field(
        default_factory=lambda: defaultdict(lambda: None), init=False
    )
    _updates: int = field(default=0, init=False)

    # -------------------------
    # Lifecycle
    # -------------------------
    def init(self) -> None:
        """Create a fresh correlation table and reset contexts."""
        self.corr = CorrelationTable(
            capacity=self.table_capacity, max_succ_per_tag=self.max_succ_per_tag
        )
        self._prev_tag_global = None
        self._prev_tag_per_pc.clear()
        self._updates = 0
        logger.info(
            "TCP init: cap=%d succ/tag=%d top_k=%d tol=%.3f chain=%d",
            self.table_capacity,
            self.max_succ_per_tag,
            self.top_k,
            self.tolerance,
            self.chain_depth,
        )

    def close(self) -> None:
        """Clean up prefetcher state."""
        logger.info("TCP closed")

    # -------------------------
    # Helpers
    # -------------------------
    def _tag(self, addr: int) -> int:
        """Extract tag from address.

        Args:
            addr: Memory address.

        Returns:
            Address tag (address shifted right by line_size_bits).
        """
        return addr >> self.line_size_bits

    def _addr_from_tag(self, tag: int) -> int:
        """Convert tag back to address.

        Args:
            tag: Address tag.

        Returns:
            Full memory address.
        """
        return tag << self.line_size_bits

    def _get_prev_tag(self, pc: int) -> Optional[int]:
        """Get previous tag for a program counter.

        Args:
            pc: Program counter.

        Returns:
            Previous tag if available, None otherwise.
        """
        return (
            self._prev_tag_per_pc[pc]
            if self.use_per_pc_context
            else self._prev_tag_global
        )

    def _set_prev_tag(self, pc: int, tag: int) -> None:
        """Set previous tag for a program counter.

        Args:
            pc: Program counter.
            tag: Tag to store.
        """
        if self.use_per_pc_context:
            self._prev_tag_per_pc[pc] = tag
        else:
            self._prev_tag_global = tag

    def _maybe_decay(self) -> None:
        """Apply periodic decay to correlation table if enabled."""
        if self.decay_every <= 0:
            return
        self._updates += 1
        if self._updates % self.decay_every != 0:
            return
        logger.debug("TCP: applying decay")
        for entry in list(self.corr.table.values()):
            new_freq = Counter()
            new_total = 0
            for successor, count in entry.freq.items():
                decayed_count = int(max(0, round(count * self.decay_factor)))
                if decayed_count > 0:
                    new_freq[successor] = decayed_count
                    new_total += decayed_count
            entry.freq = new_freq
            entry.total = new_total

    # -------------------------
    # Core: process an access
    # -------------------------
    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """Process memory access and return prefetch predictions."""
        program_counter = int(access.pc)
        address = int(access.address)
        current_tag = self._tag(address)

        self._train_correlation(program_counter, current_tag, prefetch_hit)
        self._set_prev_tag(program_counter, current_tag)

        return self._predict_with_chaining(current_tag)

    def _train_correlation(
        self, program_counter: int, current_tag: int, prefetch_hit: bool
    ) -> None:
        """Train correlation table from previous tag to current tag."""
        previous_tag = self._get_prev_tag(program_counter)
        if previous_tag is not None:
            weight = self.base_weight + (self.prefetch_hit_boost if prefetch_hit else 0)
            self.corr.update(previous_tag, current_tag, weight=weight)
            self._maybe_decay()

    def _predict_with_chaining(self, current_tag: int) -> List[int]:
        """Generate predictions using chaining from current tag."""
        predictions: List[int] = []
        seen_tags = set()
        frontier = [current_tag]
        steps = max(1, self.chain_depth)
        budget = self.top_k

        for _ in range(steps):
            if not frontier or budget <= 0:
                break
            next_frontier: List[int] = []
            for source_tag in frontier:
                successors = self.corr.predict(
                    source_tag, tolerance=self.tolerance, top_k=budget
                )
                for successor_tag in successors:
                    if self.avoid_duplicates and (
                        successor_tag in seen_tags or successor_tag == current_tag
                    ):
                        continue
                    seen_tags.add(successor_tag)
                    predictions.append(self._addr_from_tag(successor_tag))
                    budget -= 1
                    next_frontier.append(successor_tag)
                    if budget <= 0:
                        break
                if budget <= 0:
                    break
            frontier = next_frontier

        return predictions[: self.top_k]
