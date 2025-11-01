"""
Perceptron-Based Prefetch Filtering by Bhatia et al.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class PerceptronEntry:
    """Tracks a candidate prefetch for later training.

    Attributes:
        address: The prefetch address.
        pc: Program counter associated with this prefetch.
        features: Dictionary of feature indices used for prediction.
        depth: Prefetch depth (how many strides ahead).
        stride: Detected stride value.
        valid: Whether this entry is valid.
    """

    address: int
    pc: int
    features: Dict[str, Any]
    depth: int = 0
    stride: int = 0
    valid: bool = False


class PerceptronPrefetcher(PrefetchAlgorithm):
    """Perceptron-based prefetch filtering (PPF).

    Implements a perceptron filter on top of a base stride prefetcher.

    References:
        - "Perceptron-Based Prefetch Filtering" (HPCA 2005, Jiménez et al.)
    """

    def __init__(
        self,
        cache_line_size_bytes: int = 64,
        page_size_bytes: int = 4096,
        max_stride_history: int = 16,
        weight_table_size: int = 256,
    ):
        """Initialize perceptron prefetcher.

        Args:
            cache_line_size_bytes: Size of a cache line in bytes.
            page_size_bytes: Size of a memory page in bytes.
            max_stride_history: Maximum number of stride history entries.
            weight_table_size: Size of each weight table.
        """
        self.cache_line_size_bytes = cache_line_size_bytes
        self.page_size_bytes = page_size_bytes
        self.max_stride_history = max_stride_history
        self.weight_table_size = weight_table_size

        # Perceptron state: fixed-size hashed weight tables
        self.weight_tables = {
            "pc": [0] * weight_table_size,
            "address": [0] * weight_table_size,
            "depth": [0] * weight_table_size,
            "pc_xor_depth": [0] * weight_table_size,
            "pc_xor_stride": [0] * weight_table_size,
        }

        # Tables for post-decision training
        self.prefetch_table: Dict[int, PerceptronEntry] = {}
        self.reject_table: Dict[int, PerceptronEntry] = {}
        self.last_address_per_pc: Dict[int, int] = {}

        # Parameters
        self.TAU_LO = -8  # reject threshold
        self.TAU_HI = 8  # high confidence (could guide cache placement)
        self.WEIGHT_MAX = 15
        self.WEIGHT_MIN = -16

    def init(self) -> None:
        """Initialize or reset prefetcher state."""
        for k in self.weight_tables:
            self.weight_tables[k] = [0] * self.weight_table_size
        self.prefetch_table.clear()
        self.reject_table.clear()
        self.last_address_per_pc.clear()
        logger.info("PerceptronPrefetcher initialized.")

    # ----------------- Internal Helpers -----------------

    def _hash(self, feature_val: int) -> int:
        """Hash a feature value into table index.

        Args:
            feature_val: Feature value to hash.

        Returns:
            Table index.
        """
        return abs(hash(feature_val)) % self.weight_table_size

    def _get_features(
        self, access: MemoryAccess, depth: int, stride: int
    ) -> Dict[str, Any]:
        """Extract hashed features for prediction.

        Args:
            access: Memory access event.
            depth: Prefetch depth.
            stride: Detected stride.

        Returns:
            Dictionary mapping feature names to table indices.
        """
        page_addr = access.address // self.page_size_bytes
        cache_line_addr = access.address // self.cache_line_size_bytes
        return {
            "pc": self._hash(access.pc),
            "address": self._hash(cache_line_addr),
            "depth": self._hash(depth),
            "pc_xor_depth": self._hash(access.pc ^ depth),
            "pc_xor_stride": self._hash(access.pc ^ stride),
        }

    def _predict(self, features: Dict[str, Any]) -> int:
        """Compute perceptron score as sum of feature weights.

        Args:
            features: Dictionary of feature indices.

        Returns:
            Perceptron score.
        """
        return sum(self.weight_tables[name][idx] for name, idx in features.items())

    def _train(self, features: Dict[str, Any], useful: bool) -> None:
        """Update weights using perceptron rule with saturating counters.

        Args:
            features: Dictionary of feature indices.
            useful: Whether the prefetch was useful.
        """
        direction = 1 if useful else -1
        for name, idx in features.items():
            old = self.weight_tables[name][idx]
            new = old + direction
            self.weight_tables[name][idx] = min(
                self.WEIGHT_MAX, max(self.WEIGHT_MIN, new)
            )

    # ----------------- Main API -----------------

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """Process a memory access and return prefetch predictions.

        Args:
            access: The current memory access.
            prefetch_hit: Whether this access was a prefetch hit.

        Returns:
            List of addresses to prefetch.
        """
        prefetches: List[int] = []
        stride = 0

        # ---------------- TRAINING ----------------
        if prefetch_hit and access.address in self.prefetch_table:
            entry = self.prefetch_table.pop(access.address)
            self._train(entry.features, useful=True)

        if access.address in self.reject_table:
            entry = self.reject_table.pop(access.address)
            self._train(entry.features, useful=True)

        # ---------------- PREDICTION ----------------
        if access.pc in self.last_address_per_pc:
            last_addr = self.last_address_per_pc[access.pc]
            stride = access.address - last_addr
            if stride != 0:
                for depth in range(1, 5):  # up to 4 candidates
                    cand = access.address + stride * depth
                    features = self._get_features(access, depth, stride)
                    score = self._predict(features)

                    if score > self.TAU_LO:
                        prefetches.append(cand)
                        self.prefetch_table[cand] = PerceptronEntry(
                            cand, access.pc, features, depth, stride, True
                        )
                        logger.debug(
                            f"ACCEPT prefetch addr=0x{cand:x}, pc=0x{access.pc:x}, score={score}"
                        )
                    else:
                        self.reject_table[cand] = PerceptronEntry(
                            cand, access.pc, features, depth, stride, True
                        )
                        logger.debug(
                            f"REJECT prefetch addr=0x{cand:x}, pc=0x{access.pc:x}, score={score}"
                        )

        # Update stride history
        self.last_address_per_pc[access.pc] = access.address
        return prefetches

    def close(self) -> None:
        """Clean up prefetcher state."""
        logger.info("PerceptronPrefetcher closed.")
        self.prefetch_table.clear()
        self.reject_table.clear()
