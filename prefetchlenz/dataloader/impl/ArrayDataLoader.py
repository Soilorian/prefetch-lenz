import logging
from abc import ABC
from typing import List

from prefetchlenz.dataloader.dataloader import DataLoader
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

logger = logging.getLogger("prefetchLenz.dataloader.impl")


class ArrayLoader(DataLoader):
    """Loads addresses from a Python list."""

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __init__(self, data: List[MemoryAccess]):
        """
        Args:
            data (List[MemoryAccess]): Pre-collected address sequence.
        """
        self.data = data

    def load(self):
        """Return the array of addresses."""
        logger.debug(f"ArrayLoader loading {len(self.data)} addresses")

    def __getitem__(self, item) -> MemoryAccess:
        return self.data[item]
