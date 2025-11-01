"""
Best-Offset Prefetcher

Algorithm: Best-Offset Hardware Prefetching by Pierre Michaud

This prefetcher learns the best offset value for sequential prefetching by testing
a predefined list of offsets. During a training phase, it tests each offset to see
if it matches a recurring pattern (X, X+d) in recent memory accesses. Once training
completes, it uses the best-scoring offset to generate prefetches.

Key Components:
- OffsetList: Manages a predefined list of offsets with prime factors <= 5 and their scores
- RecentRequestTable: Tracks recent memory accesses to detect offset patterns
- BestOffsetPrefetcher: Main prefetcher that learns and applies the best offset

How it works:
1. Training phase: Test offsets by checking if (current - offset) was recently accessed
2. Score offsets based on how often they match the pattern
3. Training ends when an offset reaches max_score or max_rounds is reached
4. Operational phase: Use learned offset to prefetch (current + offset)
"""

from typing import List, Optional

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.single import SingleReplacementPolicy
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm


class OffsetList:
    """Manages a predefined list of offsets and their corresponding scores.

    The offsets are numbers from 1 to 256 with prime factors less than or equal to 5.
    The class handles score updates and selecting the best offset.
    """

    def __init__(self):
        """Initialize offset list with scores. Offsets have prime factors <= 5."""
        self.offsets = [
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
            12,
            15,
            16,
            18,
            20,
            24,
            25,
            27,
            30,
            32,
            36,
            40,
            45,
            48,
            50,
            54,
            60,
            64,
            72,
            75,
            80,
            81,
            90,
            96,
            100,
            108,
            120,
            125,
            128,
            135,
            144,
            150,
            160,
            162,
            180,
            192,
            200,
            216,
            225,
            240,
            243,
            250,
            256,
        ]
        self.scores = {offset: 0 for offset in self.offsets}
        self.needed = set(self.offsets)

    def get_needed(self) -> List[int]:
        """Return offsets that still need testing."""
        return list(self.needed)

    def update_score(self, offset: int, delta: int) -> None:
        """Increment the score for the given offset."""
        if offset in self.scores:
            self.scores[offset] += delta

    def reset_scores(self) -> None:
        """Reset all scores and mark all offsets as needing testing."""
        self.scores = {offset: 0 for offset in self.offsets}
        self.needed = set(self.offsets)

    def get_best_offset(self, bad_score: int) -> Optional[int]:
        """Find the highest-scoring offset that exceeds bad_score."""
        best_offset = None
        best_score = bad_score
        for offset, score in self.scores.items():
            if score > best_score:
                best_score = score
                best_offset = offset
        return best_offset


class RecentRequestTable:
    """A table to store recent memory accesses using a cache-like structure.

    This is used to check for previous access at a specific address (X - d).
    """

    def __init__(self):
        self.cache = Cache(
            num_sets=256,
            num_ways=1,
            replacement_policy_cls=SingleReplacementPolicy,
        )

    def add_access(self, access: MemoryAccess) -> None:
        """Record a memory access in the table."""
        self.cache.put(access.address, access.address)

    def check_exists(self, address: int) -> bool:
        """Return True if the address exists in the table."""
        return self.cache.get(address) is not None


class BestOffsetPrefetcher(PrefetchAlgorithm):
    """A prefetcher that learns the best offset by testing a predefined list of offsets.

    It works in rounds, testing each offset against recent memory accesses to find a
    recurring pattern (X, X+d). The offset with the highest score becomes the
    prefetching offset.
    """

    def __init__(self, max_score=31, max_rounds=100, bad_score=1):
        """Initialize best-offset prefetcher.

        Args:
            max_score: Score threshold that ends training and selects an offset.
            max_rounds: Max rounds to test offsets before forced selection.
            bad_score: Minimum score required for offset consideration.
        """
        self.max_score = max_score
        self.max_rounds = max_rounds
        self.bad_score = bad_score
        self.offset = None
        self.current_round = 0
        self.current_offset_index = 0
        self.offset_list = OffsetList()
        self.recent_request_table = RecentRequestTable()

    def init(self) -> None:
        """Reset prefetcher state, clearing all learned data."""
        self.offset = None
        self.current_round = 0
        self.current_offset_index = 0
        self.offset_list.reset_scores()
        self.recent_request_table.cache.flush()

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """Process memory access and return prefetch addresses.

        During training, tests offsets by checking if (current - offset) was
        previously accessed. Once training completes, uses the selected offset
        to generate prefetches.
        """
        current_address = access.address
        prefetches = []

        if self.offset is None:
            offsets_to_test = self.offset_list.get_needed()
            if not offsets_to_test:
                self.offset_list.reset_scores()
                offsets_to_test = self.offset_list.get_needed()
                self.current_offset_index = 0

            offset_value = offsets_to_test[
                self.current_offset_index % len(offsets_to_test)
            ]

            if self.recent_request_table.check_exists(current_address - offset_value):
                self.offset_list.update_score(offset_value, 1)
                if self.offset_list.scores[offset_value] >= self.max_score:
                    self.offset = offset_value
                    self.offset_list.reset_scores()
                    self.recent_request_table.cache.flush()

            self.current_offset_index += 1
            if self.current_offset_index >= len(offsets_to_test):
                self.current_offset_index = 0
                self.current_round += 1
                if self.current_round >= self.max_rounds:
                    self.offset = self.offset_list.get_best_offset(self.bad_score)
                    self.current_round = 0
                    self.offset_list.reset_scores()
                    self.recent_request_table.cache.flush()

        self.recent_request_table.add_access(access)

        if self.offset is not None:
            prefetches.append(current_address + self.offset)

        return prefetches

    def close(self) -> None:
        """Clean up prefetcher state."""
        self.recent_request_table.cache.flush()
        self.init()
