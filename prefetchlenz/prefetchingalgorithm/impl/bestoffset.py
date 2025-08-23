import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.single import SingleReplacementPolicy
from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess


class OffsetList:
    """
    Manages a predefined list of offsets and their corresponding scores.

    The offsets are numbers from 1 to 256 with prime factors less than or equal to 5.
    The class handles score updates and selecting the best offset.
    """

    def __init__(self):
        """
        Initializes the offset list and score tracking.
        """
        # The list of offsets is hardcoded based on the provided prime factorization constraint.
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

    def get_needed(self):
        """
        Returns the list of offsets that still need to be tested.
        """
        return list(self.needed)

    def update_score(self, offset: int, delta: int):
        """
        Increases the score of a given offset.

        Args:
            offset (int): The offset to update.
            delta (int): The amount to add to the score.
        """
        if offset in self.scores:
            self.scores[offset] += delta

    def reset_scores(self):
        """
        Resets all offset scores to zero and re-initializes the 'needed' set.
        """
        self.scores = {offset: 0 for offset in self.offsets}
        self.needed = set(self.offsets)

    def get_best_offset(self, bad_score: int):
        """
        Finds the offset with the highest score above a certain threshold.

        Args:
            bad_score (int): The minimum score an offset must have to be considered.

        Returns:
            int | None: The best offset, or None if no offset meets the score threshold.
        """
        best_offset = None
        best_score = bad_score
        for offset, score in self.scores.items():
            if score > best_score:
                best_score = score
                best_offset = offset
        return best_offset


class RecentRequestTable:
    """
    A table to store recent memory accesses using a cache-like structure.

    This is used to check for a previous access at a specific address (X - d).
    """

    def __init__(self):
        """
        Initializes the cache used for the recent request table.
        """
        self.cache = Cache(
            num_sets=256, num_ways=1, replacement_policy_cls=SingleReplacementPolicy
        )

    def add_access(self, access: MemoryAccess):
        """
        Stores a memory access address in the table.

        Args:
            access (MemoryAccess): The memory access to record.
        """
        self.cache.put(access.address, access.address)

    def check_exists(self, address: int) -> bool:
        """
        Checks if a given address is present in the table.

        Args:
            address (int): The address to check.

        Returns:
            bool: True if the address is found, False otherwise.
        """
        return self.cache.get(address) is not None


class BestOffsetPrefetcher:
    """
    A prefetcher that learns the best offset by testing a predefined list of offsets.

    It works in rounds, testing each offset against recent memory accesses to find a
    recurring pattern (X, X+d). The offset with the highest score becomes the
    prefetching offset.
    """

    def __init__(
        self,
        max_score=31,
        max_rounds=100,
        bad_score=1,
    ):
        """
        Initializes the prefetcher with configuration parameters.

        Args:
            max_score (int): The score at which a training round ends and an offset is selected.
            max_rounds (int): The maximum number of rounds to test all offsets before a selection is made.
            bad_score (int): The minimum score for an offset to be considered for selection.
        """
        self.max_score = max_score
        self.max_rounds = max_rounds
        self.bad_score = bad_score
        self.offset = None
        self.current_round = 0
        self.current_offset_index = 0
        self.offset_list = OffsetList()
        self.rr_table = RecentRequestTable()

    def init(self):
        """
        Resets the prefetcher to its initial state, clearing all learned data.
        """
        self.offset = None
        self.current_round = 0
        self.current_offset_index = 0
        self.offset_list.reset_scores()
        self.rr_table.cache.flush()

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Processes a memory access and determines the next prefetch address.

        This method is the core of the prefetcher's learning and operation. It
        tests a potential offset, updates its score, and issues a prefetch if
        an offset has been selected.

        Args:
            access (MemoryAccess): The current memory access from the CPU.
            prefetch_hit (bool): A flag indicating if the current access was a prefetch hit (not used in this implementation).

        Returns:
            List[int]: A list of addresses to prefetch.
        """
        X = access.address
        prefetches = []

        if self.offset is None:
            # Training phase
            offsets_to_test = self.offset_list.get_needed()
            if not offsets_to_test:
                # Should not happen in this implementation, but a safeguard
                self.offset_list.reset_scores()
                offsets_to_test = self.offset_list.get_needed()
                self.current_offset_index = 0

            d = offsets_to_test[self.current_offset_index % len(offsets_to_test)]

            # Check for a matching access in the past
            if self.rr_table.check_exists(X - d):
                self.offset_list.update_score(d, 1)
                if self.offset_list.scores[d] >= self.max_score:
                    # Training complete: an offset reached max_score
                    self.offset = d
                    self.offset_list.reset_scores()
                    self.rr_table.cache.flush()

            self.current_offset_index += 1
            if self.current_offset_index >= len(offsets_to_test):
                # End of a round
                self.current_offset_index = 0
                self.current_round += 1
                if self.current_round >= self.max_rounds:
                    # Training complete: max_rounds reached
                    self.offset = self.offset_list.get_best_offset(self.bad_score)
                    self.current_round = 0
                    self.offset_list.reset_scores()
                    self.rr_table.cache.flush()

        # Add the current access to the RR table for future checks
        self.rr_table.add_access(access)

        # Issue a prefetch if an offset has been selected
        if self.offset is not None:
            prefetches.append(X + self.offset)

        return prefetches

    def close(self):
        """
        Flushes the prefetcher state and resources on shutdown.
        """
        self.rr_table.cache.flush()
        self.init()
