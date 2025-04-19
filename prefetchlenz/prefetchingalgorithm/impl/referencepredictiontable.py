import logging

from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.referencepredictiontable")


class RPTAlgorithm(PrefetchAlgorithm):
    """
    Reference Prediction Table (RPT) stride-based prefetcher.

    Tracks load PC → last_addr, stride, and a 2-bit FSM to validate strides.
    """

    def __init__(self):
        """Create empty RPT."""
        self.table = {}
        self.last_pc = 0

    def init(self):
        """Clear the table before a new run."""
        logger.debug("RPT init: clearing table")
        self.table.clear()
        self.last_pc = 0

    def progress(self, address: int):
        """
        Feed next access and get prefetch addresses.

        Args:
            address (int): Current accessed address.

        Returns:
            List[int]: Addresses to prefetch (possibly empty).
        """
        pc = self.last_pc
        self.last_pc += 1
        prediction = []

        if pc in self.table:
            last_addr, stride, state = self.table[pc]
            new_stride = address - last_addr

            if new_stride == stride:
                # advance FSM (0→1→2→steady)
                state = min(state + 1, 2)
                if state == 2:
                    prediction = [address + stride]
                    logger.debug(f"RPT predict {prediction} for PC {pc}")
            else:
                state = 0

            self.table[pc] = (address, new_stride, state)
        else:
            # first-seen, no prediction
            self.table[pc] = (address, 0, 0)

        return prediction

    def close(self):
        """Clear the table after run."""
        logger.debug("RPT close: clearing table")
        self.table.clear()
