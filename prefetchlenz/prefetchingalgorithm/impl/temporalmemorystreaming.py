import logging
from collections import defaultdict, deque
from typing import Dict, List

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.temporal_streaming")


class TemporalMemoryStreamingPrefetcher(PrefetchAlgorithm):
    """
    A simplified implementation of the Temporal Memory Streaming (TMS) prefetcher.

    This prefetcher learns and stores sequences of memory accesses (streams).
    When a known stream pattern is detected, it prefetches the remaining addresses
    in the sequence.
    """

    def __init__(self, history_length: int = 5, stream_length_threshold: int = 4):
        """
        Args:
            history_length (int): The number of recent memory addresses to track.
            stream_length_threshold (int): The minimum length of a sequence to be
                                           considered a "stream" and stored.
        """
        self.history: deque[int] = deque(maxlen=history_length)
        self.streams: Dict[int, List[int]] = defaultdict(list)
        self.stream_length_threshold = stream_length_threshold
        self.last_address: int | None = None

    def init(self):
        """
        Initialize the prefetcher's state for a new simulation run.
        """
        self.history.clear()
        self.streams.clear()
        self.last_address = None
        logger.info("TemporalMemoryStreamingPrefetcher initialized.")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process a single memory access, detect streams, and prefetch.

        Args:
            access (MemoryAccess): The current memory access.
            prefetch_hit (bool): Whether the memory access is prefetched.

        Returns:
            List[int]: Predicted future addresses to prefetch.
        """
        current_address = access.address
        prefetches: List[int] = []

        # We only consider misses for stream learning, as per the paper.
        if not prefetch_hit:
            self.history.append(current_address)

        if len(self.history) >= 2:
            # Check for a match with an existing stream.
            for start_address, stream_seq in self.streams.items():
                if self.last_address is not None and self.last_address == start_address:
                    prefetches.extend(stream_seq)
                    logger.debug(
                        f"Detected stream starting at {start_address}. Prefetching: {stream_seq}"
                    )
                    break

            # Learn a new stream if the history is long enough.
            if len(self.history) == self.stream_length_threshold:
                # The history is our new stream. The first element is the key.
                start_address = self.history[0]
                stream_sequence = list(self.history)[1:]
                if start_address not in self.streams:
                    self.streams[start_address] = stream_sequence
                    logger.debug(f"Learned a new stream starting at {start_address}.")

        self.last_address = current_address
        return prefetches

    def close(self):
        """
        Clean up any state after simulation ends.
        """
        self.history.clear()
        self.streams.clear()
        self.last_address = None
        logger.info("TemporalMemoryStreamingPrefetcher closed.")
