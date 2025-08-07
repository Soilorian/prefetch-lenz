import logging
from dataclasses import dataclass
from typing import List, Optional

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.Hawkeye import HawkeyeReplacementPolicy
from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm
from prefetchlenz.util.size import Size

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.triage")


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
    """
    PC-localized training unit for the Triage prefetcher.

    Tracks the last address accessed by each program counter (PC)
    to help discover localized memory stream patterns.

    Methods:
        train(access: MemoryAccess) -> tuple[int, int] | None:
            Update training state with a new access and return (prev_address, curr_address) pair.
        clear():
            Reset internal state.
    """

    def __init__(self):
        self.pc_histories = {}

    def train(self, access: MemoryAccess) -> tuple[int, int] | None:
        result = None
        if access.pc in self.pc_histories:
            result = self.pc_histories[access.pc], access.address

        self.pc_histories[access.pc] = access.address
        return result

    def clear(self):
        self.pc_histories.clear()


class TriagePrefetcher(PrefetchAlgorithm):
    """
    Triage Prefetcher with on-chip metadata, dynamic confidence tracking,
    and adaptive resizing using a Hawkeye-inspired replacement policy.

    This prefetcher learns localized memory streams by correlating
    consecutive accesses from the same program counter (PC).
    It stores a small amount of metadata in a tag-store-like cache structure
    and triggers a prefetch if the confidence in the learned correlation is high.

    Features:
    - Localized pattern learning using `TrainingUnit`.
    - Metadata caching with configurable associativity and sets.
    - Confidence mechanism for prefetch filtering.
    - Online adaptation of table size based on usefulness.
    - Pluggable replacement policy via `Cache`.

    Args:
        num_ways (int): Initial number of ways per cache set.
        init_size (Size): Initial total metadata storage size.
        min_size (Size): Minimum allowable table size.
        max_size (Size): Maximum allowable table size.
        resize_epoch (int): Number of accesses between resize decisions.
        grow_thresh (float): Usefulness threshold to grow metadata table.
        shrink_thresh (float): Usefulness threshold to shrink metadata table.

    Methods:
        init():
            Initialize/reset the internal state and metadata cache.
        progress(access: MemoryAccess, prefetch_hit: bool) -> List[int]:
            Process a memory access and potentially issue a prefetch.
        close():
            Clean up resources and log final cache state.
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

    def init(self):
        """
        Initialize or reset internal state.
        """
        self.cache.flush()
        self.training_unit.clear()
        self.meta_accesses = 0
        self.useful_prefetches = 0
        logger.info(f"Triage init: metadata capacity = {self.num_sets}")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process a single memory access.

        :param access: The current memory access event.
        :type access: MemoryAccess
        :param prefetch_hit: Whether this access was already prefetched.
        :type prefetch_hit: bool
        :return: List of predicted addresses to prefetch.
        :rtype: List[int]
        """

        if prefetch_hit:
            self.useful_prefetches += 1
        addr = access.address
        preds: List[int] = []

        # 1) issue the prefetch based on metadata
        prefetch: TriagePrefetcherMetaData = self.cache.get(addr)
        if prefetch is not None:
            preds.append(prefetch.neighbor)

        # 2) Train on PC-localized stream
        metadata = self.training_unit.train(access)
        if metadata is not None:

            # 3) update metadata in cache
            prefetch: TriagePrefetcherMetaData = self.cache.get(metadata[0])
            if prefetch is None:
                prefetch = TriagePrefetcherMetaData(neighbor=metadata[1], confidence=0)
            elif prefetch.neighbor != metadata[1]:
                if prefetch.confidence < 1:
                    prefetch.neighbor = metadata[1]
                else:
                    prefetch.confidence = max((prefetch.confidence - 1), 0)
            else:
                prefetch.confidence = min(1, (prefetch.confidence + 1))

            self.cache.put(metadata[0], prefetch)

        # 4) Update stats and maybe resize
        self.meta_accesses += 1
        if self.meta_accesses >= self.resize_epoch:
            self._maybe_resize()
            self.meta_accesses = 0
            self.useful_prefetches = 0

        return preds

    def _maybe_resize(self):
        """
        Grow/shrink metadata capacity based on usefulness ratio.
        """
        ratio = self.useful_prefetches / max(1, self.resize_epoch)
        old = int(self.num_ways * self.num_sets)
        if ratio > self.grow_thresh and old < int(self.max_size):
            self.num_ways += 1
            self.cache.change_num_ways(self.num_ways)
            new_bytes = int(self.num_ways * self.num_sets)
            logger.info(f"Growing table: {old}→{new_bytes} bytes (ratio={ratio:.3f})")
        elif ratio < self.shrink_thresh and old > int(self.min_size):
            self.num_ways -= 1
            self.cache.change_num_ways(self.num_ways)
            new_bytes = int(self.num_ways * self.num_sets)
            logger.info(f"Shrinking table: {old}→{new_bytes} bytes (ratio={ratio:.3f})")

    def close(self):
        """
        Final cleanup.
        """
        logger.info(f"Triage closed: final entries={len(self.cache)}")
        self.cache.flush()
        self.training_unit.clear()
