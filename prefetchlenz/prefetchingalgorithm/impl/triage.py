"""
Triage Prefetcher

Algorithm: Temporal Prefetching Without the Off-Chip Metadata by Wu et al.

This prefetcher learns temporal correlations between consecutive memory addresses
without requiring off-chip metadata storage. It uses a PC-localized training unit
to discover address pairs (neighbors) and maintains a metadata cache that adaptively
resizes based on prefetch effectiveness.

Key Components:
- TrainingUnit: PC-localized training that tracks last address per PC
- TriagePrefetcherMetaData: Stores neighbor address and confidence for each tracked address
- TriagePrefetcher: Main prefetcher with adaptive metadata cache sizing

How it works:
1. Training unit tracks last address accessed by each PC to discover neighbor pairs
2. Metadata cache stores (address -> neighbor) mappings with 1-bit confidence
3. Generate prefetches from cached neighbor relationships
4. On prefetch hits, credit the metadata cache entry
5. Periodically resize metadata cache based on prefetch hit rate
6. Confidence counters prevent prefetching when correlation is uncertain
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.hawkeye import HawkeyeReplacementPolicy
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm
from prefetchlenz.util.size import Size

logger = logging.getLogger(__name__)


@dataclass
class TriagePrefetcherMetaData:
    """
    Metadata associated with each tracked memory address.

    Attributes:
        neighbor (int): The next correlated memory address observed after this one.
        confidence (int): A 1-bit confidence counter (0 or 1) indicating whether the
                          correlation is strong enough to be trusted for prefetching.
        score (int): score used for hawkeye algorithm.
    """

    neighbor: int
    confidence: int = 0


class TrainingUnit:
    """PC-localized training unit for the Triage prefetcher.

    Tracks the last address accessed by each program counter (PC)
    to help discover localized memory stream patterns.
    """

    def __init__(self):
        """Initialize training unit."""
        self.pc_histories = {}

    def train(self, access: MemoryAccess) -> tuple[int, int] | None:
        """Update training state with a new access.

        Args:
            access: The current memory access.

        Returns:
            Tuple of (prev_address, curr_address) if previous access exists,
            None otherwise.
        """
        result = None
        if access.pc in self.pc_histories:
            result = self.pc_histories[access.pc], access.address

        self.pc_histories[access.pc] = access.address
        return result

    def clear(self) -> None:
        """Reset internal state."""
        self.pc_histories.clear()


class TriagePrefetcher(PrefetchAlgorithm):
    """Triage prefetcher with on-chip metadata, dynamic confidence tracking,
    and adaptive resizing using a Hawkeye-inspired replacement policy.

    This prefetcher learns localized memory streams by correlating
    consecutive accesses from the same program counter (PC).
    It stores a small amount of metadata in a tag-store-like cache structure
    and triggers a prefetch if the confidence in the learned correlation is high.

    Features:
        - Localized pattern learning using TrainingUnit.
        - Metadata caching with configurable associativity and sets.
        - Confidence mechanism for prefetch filtering.
        - Online adaptation of table size based on usefulness.
        - Pluggable replacement policy via Cache.
    """

    def __init__(
        self,
        num_ways: int = 1,
        init_size: Size = Size.from_kb(512),
        min_size: Size = Size(0),
        max_size: Size = Size.from_mb(1),
        resize_epoch: int = 50_000,
        grow_thresh: float = 0.05,
        shrink_thresh: float = 0.05,
    ):
        """Initialize Triage prefetcher.

        Args:
            num_ways: Initial number of ways per cache set.
            init_size: Initial total metadata storage size.
            min_size: Minimum allowable table size.
            max_size: Maximum allowable table size.
            resize_epoch: Number of accesses between resize decisions.
            grow_thresh: Usefulness threshold to grow metadata table.
            shrink_thresh: Usefulness threshold to shrink metadata table.
        """
        self.previous_access = None
        self.num_ways = num_ways
        self.num_sets = init_size.bytes // num_ways
        self.min_size = min_size
        self.max_size = max_size

        self.cache = Cache(
            num_sets=self.num_sets,
            num_ways=self.num_ways,
            replacement_policy_cls=HawkeyeReplacementPolicy,
        )

        self.training_unit = TrainingUnit()

        self.meta_accesses = 0
        self.useful_prefetches = 0
        self.resize_epoch = resize_epoch
        self.grow_thresh = grow_thresh
        self.shrink_thresh = shrink_thresh

        logger.debug("TriagePrefetcher instantiated")

    def init(self) -> None:
        """Initialize or reset internal state."""
        self.cache.flush()
        self.training_unit.clear()
        self.meta_accesses = 0
        self.useful_prefetches = 0
        logger.info(f"Triage init: metadata capacity = {self.num_sets}")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """Process memory access and return prefetch addresses."""
        if prefetch_hit:
            self._handle_prefetch_hit(access)

        predictions = self._generate_predictions(access)
        self._update_metadata_from_training(access)
        self._maybe_update_and_resize()

        return predictions

    def _handle_prefetch_hit(self, access: MemoryAccess) -> None:
        """Update statistics and cache on prefetch hit."""
        self.useful_prefetches += 1
        if self.previous_access is not None:
            self.cache.prefetch_hit(self.previous_access.address)

    def _generate_predictions(self, access: MemoryAccess) -> List[int]:
        """Generate predictions from cached metadata."""
        self.previous_access = access
        predictions: List[int] = []

        metadata_entry = self.cache.get(access.address)
        if metadata_entry is not None:
            predictions.append(metadata_entry.neighbor)

        return predictions

    def _update_metadata_from_training(self, access: MemoryAccess) -> None:
        """Update metadata cache from training unit output."""
        training_result = self.training_unit.train(access)
        if training_result is None:
            return

        previous_address, current_address = training_result
        metadata_entry = self.cache.get(previous_address)

        if metadata_entry is None:
            metadata_entry = TriagePrefetcherMetaData(
                neighbor=current_address, confidence=0
            )
        else:
            self._update_metadata_entry(metadata_entry, current_address)

        self.cache.put(previous_address, metadata_entry)

    def _update_metadata_entry(
        self, metadata_entry: TriagePrefetcherMetaData, new_neighbor: int
    ) -> None:
        """Update metadata entry with new neighbor observation."""
        if metadata_entry.neighbor != new_neighbor:
            if metadata_entry.confidence < 1:
                metadata_entry.neighbor = new_neighbor
            else:
                metadata_entry.confidence = max(metadata_entry.confidence - 1, 0)
        else:
            metadata_entry.confidence = min(1, metadata_entry.confidence + 1)

    def _maybe_update_and_resize(self) -> None:
        """Update access statistics and resize if epoch reached."""
        self.meta_accesses += 1
        if self.meta_accesses >= self.resize_epoch:
            self._maybe_resize()
            self.meta_accesses = 0
            self.useful_prefetches = 0

    def _maybe_resize(self) -> None:
        """Grow or shrink metadata capacity based on usefulness ratio."""
        ratio = self.useful_prefetches / max(1, self.resize_epoch)
        old = int(self.num_ways * self.num_sets)
        if ratio >= self.grow_thresh and old < int(self.max_size):
            self.num_ways += 1
            self.cache.change_num_ways(self.num_ways)
            new_bytes = int(self.num_ways * self.num_sets)
            logger.info(f"Growing table: {old}→{new_bytes} bytes (ratio={ratio:.3f})")
        elif ratio <= self.shrink_thresh and old > int(self.min_size):
            self.num_ways -= 1
            self.cache.change_num_ways(self.num_ways)
            new_bytes = int(self.num_ways * self.num_sets)
            logger.info(f"Shrinking table: {old}→{new_bytes} bytes (ratio={ratio:.3f})")

    def close(self) -> None:
        """Perform final cleanup."""
        logger.info(f"Triage closed: final entries={len(self.cache)}")
        self.cache.flush()
        self.training_unit.clear()
