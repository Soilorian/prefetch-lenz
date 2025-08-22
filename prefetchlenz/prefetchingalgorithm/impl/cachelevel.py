import logging
from typing import Dict, List

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.level_predictor")


class CacheLevelPredictor(PrefetchAlgorithm):
    """
    An implementation of the cache level predictor from "Reducing Load Latency
    with Cache Level Prediction," adapted to the PrefetchAlgorithm interface.

    This class predicts the cache hierarchy level where a load instruction's
    data will be found. This is a level predictor, not a prefetcher, so it
    does not generate addresses to prefetch. The `progress` method will
    log the prediction and return an empty list to satisfy the interface.

    The prediction is based on a history of program counters (PC)
    and their associated access patterns.
    """

    def __init__(self, history_size: int = 100):
        """
        Args:
            history_size (int): The number of recent PC accesses to remember for prediction.
        """
        self.history: List[int] = []
        self.history_size = history_size
        self.predictions: Dict[int, str] = {}  # PC -> Predicted Level

    def init(self):
        """
        Initialize the predictor state before a new simulation.
        """
        self.history = []
        self.predictions = {}
        logger.info("CacheLevelPredictor initialized.")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process a memory access, update the prediction history, and
        predict the cache level for the current memory access.
        """
        current_pc = access.pc

        # Add the current PC to our history
        self.history.append(current_pc)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        # A very simplified prediction logic based on PC history
        # A real implementation would use a more sophisticated model like TAGE.
        if current_pc in self.predictions:
            # If we've seen this PC before, use our "learned" prediction.
            predicted_level = self.predictions[current_pc]
            logger.debug(
                f"Access at PC {current_pc}. Predicting a hit at {predicted_level}."
            )
        else:
            # First time seeing this PC, make a simple guess and "learn" it.
            # Example learning rule: even PCs predict L2, odd PCs predict L3.
            if current_pc % 2 == 0:
                self.predictions[current_pc] = "L2"
            else:
                self.predictions[current_pc] = "L3"

            logger.debug(
                f"Access at PC {current_pc}. First time, making a guess and learning."
            )

        # Since this is not a prefetcher, we return an empty list.
        return []

    def close(self):
        """
        Clean up any state after simulation ends.
        """
        self.history = []
        self.predictions = {}
        logger.info("CacheLevelPredictor closed.")
