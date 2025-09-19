# markov_predictor.py
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("markov_predictor")
logger.addHandler(logging.NullHandler())


@dataclass
class MemoryAccess:
    address: int
    pc: int


class HistoryBuffer:
    """
    Sliding buffer holding last `max_order` addresses.
    push(addr) -> (prev_states, curr_states)
      prev_states: dict order->tuple(...) representing state ending at previous access
      curr_states: dict order->tuple(...) representing state ending at current access
    """

    def __init__(self, max_order: int = 2):
        assert max_order >= 1
        self.max_order = max_order
        self.buf: Deque[int] = deque(maxlen=max_order)

    def push(
        self, addr: int
    ) -> Tuple[Dict[int, Tuple[int, ...]], Dict[int, Tuple[int, ...]]]:
        prev_buf = list(self.buf)
        self.buf.append(addr)
        curr_buf = list(self.buf)
        prev_states: Dict[int, Tuple[int, ...]] = {}
        curr_states: Dict[int, Tuple[int, ...]] = {}
        for k in range(1, self.max_order + 1):
            if len(prev_buf) >= k:
                prev_states[k] = tuple(prev_buf[-k:])
            if len(curr_buf) >= k:
                curr_states[k] = tuple(curr_buf[-k:])
        logger.debug(
            "HistoryBuffer: pushed 0x%X prev=%s curr=%s", addr, prev_states, curr_states
        )
        return prev_states, curr_states

    def clear(self):
        self.buf.clear()


class MarkovTable:
    """
    Simple mapping state(tuple) -> Counter(next_addr -> freq).
    Not capacity-limited here for clarity.
    """

    def __init__(self, selection_tolerance: float = 0.05, min_count: int = 1):
        self.table: Dict[Tuple[int, ...], Dict[int, int]] = {}
        self.selection_tolerance = float(selection_tolerance)
        self.min_count = int(min_count)

    def update(self, key: Tuple[int, ...], nxt: int):
        d = self.table.setdefault(key, {})
        d[nxt] = d.get(nxt, 0) + 1
        logger.debug("MarkovTable: update %s -> 0x%X count=%d", key, nxt, d[nxt])

    def predict(self, key: Tuple[int, ...]) -> List[int]:
        d = self.table.get(key)
        if not d:
            return []
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        top = items[0][1]
        threshold = max(self.min_count, int((1.0 - self.selection_tolerance) * top))
        preds = [addr for addr, freq in items if freq >= threshold]
        logger.debug(
            "MarkovTable: predict %s -> %s (top=%d thresh=%d)",
            key,
            preds,
            top,
            threshold,
        )
        return preds

    def clear(self):
        self.table.clear()


class MarkovPredictor:
    """
    Markov predictor supporting first-order and second-order transitions.

    Usage:
      p = MarkovPredictor()
      p.init()
      preds = p.progress(MemoryAccess(address, pc), prefetch_hit=False)
    """

    def __init__(self, selection_tolerance: float = 0.05, min_count: int = 1):
        # support up to second order
        self.history = HistoryBuffer(max_order=2)
        self.order1 = MarkovTable(
            selection_tolerance=selection_tolerance, min_count=min_count
        )
        self.order2 = MarkovTable(
            selection_tolerance=selection_tolerance, min_count=min_count
        )

    def init(self):
        logger.info("MarkovPredictor: init")
        self.history.clear()
        self.order1.clear()
        self.order2.clear()

    def progress(self, access: MemoryAccess, prefetch_hit: bool = False) -> List[int]:
        """
        Process access and return predictions (list of addresses).
        Prediction uses `curr_state` (the state including the current address).
        Update uses prev_state -> curr mapping.
        """
        addr = int(access.address)
        prev_states, curr_states = self.history.push(addr)

        preds: List[int] = []

        # Prediction: prefer second-order using curr state if available
        if 2 in curr_states:
            key2 = curr_states[2]
            preds = self.order2.predict(key2)
            if preds:
                logger.debug(
                    "MarkovPredictor: 2nd-order pred for %s -> %s", key2, preds
                )

        # Fallback to first-order on curr
        if not preds and 1 in curr_states:
            key1 = curr_states[1]
            preds = self.order1.predict(key1)
            if preds:
                logger.debug(
                    "MarkovPredictor: 1st-order pred for %s -> %s", key1, preds
                )

        # Update transitions: for each order k, update prev_state(k) -> curr(addr)
        # (prev_states reflect state ending at previous access)
        if 2 in prev_states:
            self.order2.update(prev_states[2], addr)
        if 1 in prev_states:
            self.order1.update(prev_states[1], addr)

        return preds

    def close(self):
        logger.info("MarkovPredictor: close")
        self.history.clear()
        self.order1.clear()
        self.order2.clear()
