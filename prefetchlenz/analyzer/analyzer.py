import logging

from prefetchlenz.dataloader.dataloader import DataLoader
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.analysis")


class Analyzer:
    """
    Runs a prefetch algorithm on a data stream and tallies predictions.

    Counts correct vs. incorrect prefetches.
    """

    def __init__(self, algorithm: PrefetchAlgorithm, dataloader: DataLoader):
        """
        Args:
            algorithm (PrefetchAlgorithm): Prefetch algorithm instance.
            dataloader (DataLoader): Source of address stream.
        """
        self.algorithm = algorithm
        self.dataloader = dataloader

    def run(self):
        """Execute the simulation and print results."""
        self.algorithm.init()
        correct = 0
        incorrect = 0
        pending = set()
        prefetch_hit = False

        self.dataloader.load()
        dataSize = len(self.dataloader)
        logger.info(f"Starting analysis on {dataSize} data")

        for idx in range(dataSize):
            pending.clear()
            addr = self.dataloader[idx]
            preds = self.algorithm.progress(addr, prefetch_hit)
            prefetch_hit = False
            for p in preds:
                pending.add(p)

            if idx + 1 < dataSize:
                next_addr = self.dataloader[idx + 1]
                # Extract address from MemoryAccess object if needed
                next_addr_value = next_addr.address if hasattr(next_addr, 'address') else next_addr
                if next_addr_value in pending:
                    correct += 1
                    prefetch_hit = True
                    logger.debug(f"Correct: {next_addr_value}")
                else:
                    incorrect += 1
                    logger.debug(f"Incorrect: {next_addr_value}")

        self.algorithm.close()
        logger.info(f"Correct predictions: {correct}")
        logger.info(f"Incorrect predictions: {incorrect}")
