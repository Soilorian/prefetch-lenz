import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger("prefetchLenz.dataloader")


class DataLoader(ABC):
    """Interface for data loaders that supply address streams."""

    @abstractmethod
    def load(self) -> List[int]:
        """Return a sequence of memory addresses."""
        pass


class ArrayLoader(DataLoader):
    """Loads addresses from a Python list."""

    def __init__(self, data: List[int]):
        """
        Args:
            data (List[int]): Pre-collected address sequence.
        """
        self.data = data

    def load(self) -> List[int]:
        """Return the array of addresses."""
        logger.debug(f"ArrayLoader loading {len(self.data)} addresses")
        return self.data
