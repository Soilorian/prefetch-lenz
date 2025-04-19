import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm")


class PrefetchAlgorithm(ABC):
    """Interface for prefetching algorithms."""

    @abstractmethod
    def init(self):
        """Initialize any state before simulation begins."""
        pass

    @abstractmethod
    def progress(self, address: int):
        """
        Process a single memory access.

        Args:
            address (int): The current memory address.

        Returns:
            List[int]: Predicted future addresses to prefetch.
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up any state after simulation ends."""
        pass
