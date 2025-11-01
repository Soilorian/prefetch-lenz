""" """

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from prefetchlenz.config import RptConfig
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.referencepredictiontable")


class RptState(Enum):
    Init = 0
    Transient = 1
    Unstable = 2
    Steady = 3


@dataclass
class RptRow:
    stride: int
    address: int
    state: RptState


class RptAlgorithm(PrefetchAlgorithm):
    """
    Reference Prediction Table (RPT) stride-based prefetcher.

    Tracks load PC → last_addr, stride, and a 2-bit FSM to validate strides.
    """

    table: Dict[int, RptRow]

    def __init__(self):
        """Create empty RPT."""
        self.table = {}

    def init(self):
        """Clear the table before a new run."""
        logger.debug("RPT init: clearing table")
        self.table.clear()

    def progress(self, access: MemoryAccess, prefetch_hit) -> List[int]:
        """
        Feed next access and get prefetch addresses.

        Args:
            access (MemoryAccess): Current access block.

        Returns:
            List[int]: Addresses to prefetch (possibly empty).
            :param access:
        """
        pc = access.pc
        address = access.address
        predictions = []

        if pc in self.table:
            tbl = self.table[pc]
            last_addr = tbl.address
            stride = tbl.stride
            state = tbl.state
            new_stride = address - last_addr
            new_addr = address + new_stride

            if state == RptState.Init:
                state = RptState.Transient

            elif state == RptState.Transient:
                if stride == new_stride:
                    if RptConfig.prefetch_on_transient:
                        predictions.append(new_addr)
                    state = RptState.Steady

                else:
                    state = RptState.Unstable

            elif state == RptState.Steady:
                if stride == new_stride:
                    predictions.append(new_addr)
                    state = RptState.Steady

                else:
                    state = RptState.Init

            elif state == RptState.Unstable:
                if stride == new_stride:
                    if RptConfig.prefetch_on_unstable:
                        predictions.append(new_addr)
                    state = RptState.Transient

                else:
                    state = RptState.Unstable

            self.table[pc] = RptRow(address=address, stride=new_stride, state=state)
        else:
            self.table[pc] = RptRow(address=address, stride=-1, state=RptState.Init)

        return predictions

    def close(self):
        """Clear the table after run."""
        logger.debug("RPT close: clearing table")
        self.table.clear()
