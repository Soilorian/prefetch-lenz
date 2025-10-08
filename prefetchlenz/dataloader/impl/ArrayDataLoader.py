import logging
from abc import abstractmethod
from typing import List

from prefetchlenz.dataloader.dataloader import DataLoader
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

logger = logging.getLogger("prefetchLenz.dataloader.impl")


class ArrayLoader(DataLoader):
    """Loads addresses from a Python list."""

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

    @abstractmethod
    def __iter__(self):
        return self.data.__iter__()

    @abstractmethod
    def __len__(self):
        return self.data.__len__()
