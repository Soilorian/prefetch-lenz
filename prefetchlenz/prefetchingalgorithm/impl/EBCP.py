import logging
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.hawkeye import HawkeyeReplacementPolicy
from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.ebcp")


# ------------------------------- Data Records ---------------------------------


@dataclass
class Epoch:
    first_key: int
    misses: List[int] = field(default_factory=list)  # aligned addresses (keys)


@dataclass
class CorrelEntry:
    """
    Metadata payload for the correlation table:
      - succ1: predicted miss list for the next epoch (i+1)
      - succ2: optional miss list for the epoch after next (i+2)
    Lists are bounded and deduplicated while preserving insertion order.
    """

    succ1: List[int] = field(default_factory=list)
    succ2: List[int] = field(default_factory=list)

    def merge_succ(self, which: int, seq: List[int], max_len: int):
        dst = self.succ1 if which == 1 else self.succ2
        seen = set(dst)
        for a in seq:
            if a not in seen:
                dst.append(a)
                seen.add(a)
            if len(dst) >= max_len:
                break


# --------------------------------- Prefetcher ---------------------------------


class EBCPPrefetcher(PrefetchAlgorithm):
    """
    Epoch-Based Correlation Prefetcher (EBCP).
    Approximates off-chip-miss epochs via a small working-set filter + quiescence rule.
    Builds a correlation table that maps the *first miss of an epoch* to the miss lists of
    the next one/two epochs. On the first miss of a new epoch, prefetch the learned lists.

    Parameters
    ----------
    line_size : int
        Line size used for keying and aligned predictions.
    ws_capacity : int
        Capacity (in lines) of the working-set filter that classifies demand hits/misses.
    quiescence_len : int
        Number of consecutive working-set hits required to terminate the current epoch.
    degree_epochs : int
        How many future epochs to prefetch: 1 (i+1) or 2 (i+1 & i+2).
    max_targets_per_epoch : int
        Cap on the number of addresses taken from each future epoch’s list.
    table_sets, table_ways : int
        Size of the correlation table stored in your `Cache`.
    replacement_policy_cls
        Replacement policy for the correlation table (defaults to Hawkeye).
    """

    def __init__(
        self,
        line_size: int = 64,
        ws_capacity: int = 2048,
        quiescence_len: int = 32,
        degree_epochs: int = 2,
        max_targets_per_epoch: int = 32,
        table_sets: int = 1024,
        table_ways: int = 2,
        replacement_policy_cls=HawkeyeReplacementPolicy,
    ):
        # Parameters
        self.line_size = max(4, line_size)
        self.ws_capacity = max(64, ws_capacity)
        self.quiescence_len = max(1, quiescence_len)
        self.degree_epochs = 1 if degree_epochs <= 1 else 2
        self.max_targets_per_epoch = max(1, max_targets_per_epoch)

        # Correlation table (bounded; uses your pluggable replacement policy)
        self.table = Cache(
            num_sets=table_sets,
            num_ways=table_ways,
            replacement_policy_cls=replacement_policy_cls,
        )

        # Working-set filter for (hit,miss) proxy
        self._ws: "OrderedDict[int, None]" = OrderedDict()

        # Epoch state
        self._active_epoch: Optional[Epoch] = None
        self._prev_epochs: Deque[Epoch] = deque(
            maxlen=2
        )  # keep last two for i→i+1, i→i+2 linking
        self._hit_run: int = 0

        # Bookkeeping for stats and feedback
        self._issued: int = 0
        self._useful: int = 0
        self._last_access: Optional[MemoryAccess] = None

        logger.info(
            "EBCP ctor: line=%d, WS=%d, quiescence=%d, degree=%d, max_targets/epoch=%d, table=%dx%d",
            self.line_size,
            self.ws_capacity,
            self.quiescence_len,
            self.degree_epochs,
            self.max_targets_per_epoch,
            table_sets,
            table_ways,
        )

    # ------------------------------- Lifecycle --------------------------------

    def init(self):
        self.table.flush()
        self._ws.clear()
        self._active_epoch = None
        self._prev_epochs.clear()
        self._hit_run = 0
        self._issued = 0
        self._useful = 0
        self._last_access = None
        logger.info("EBCP initialized")

    def close(self):
        logger.info(
            "EBCP closed: issued=%d useful=%d (%.1f%%) table_entries=%d",
            self._issued,
            self._useful,
            (100.0 * self._useful / max(1, self._issued)),
            len(self.table),
        )
        self.table.flush()
        self._ws.clear()
        self._active_epoch = None
        self._prev_epochs.clear()

    # --------------------------------- Core -----------------------------------

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        addr_aligned = self._align(access.address)
        preds: List[int] = []

        # Credit usefulness feedback (when your driver says the current access was prefetched)
        if prefetch_hit:
            self._useful += 1
            if self._last_access is not None:
                # Let replacement policy learn usefulness (like your Triage does)
                self.table.prefetch_hit(self._align(self._last_access.address))

        is_miss = self._classify_and_update_ws(addr_aligned)

        # Epoch state machine
        if is_miss:
            if self._active_epoch is None:
                # ----- New epoch begins -----
                self._start_epoch(addr_aligned)
                # Issue prefetches for predicted next epochs
                preds = self._predict_for_first_miss(addr_aligned)
                self._issued += len(preds)
                logger.info(
                    "EBCP: start epoch at %#x, preds=%d", addr_aligned, len(preds)
                )
            # Record this epoch’s miss
            self._active_epoch.misses.append(addr_aligned)
            self._hit_run = 0
        else:
            # Track quiescence; close epoch after enough consecutive hits
            self._hit_run += 1
            if self._active_epoch is not None and self._hit_run >= self.quiescence_len:
                self._end_epoch()

        self._last_access = access
        return preds

    # ---------------------------- Epoch Management -----------------------------

    def _start_epoch(self, first_key: int):
        self._active_epoch = Epoch(first_key=first_key, misses=[first_key])

    def _end_epoch(self):
        ep = self._active_epoch
        self._active_epoch = None
        if not ep or not ep.misses:
            return

        # Link correlations for (i-1 -> i) and (i-2 -> i) when available
        if len(self._prev_epochs) >= 1:
            prev1 = self._prev_epochs[-1]
            self._update_table(prev1.first_key, ep.misses, which=1)
        if len(self._prev_epochs) >= 2:
            prev2 = self._prev_epochs[-2]
            self._update_table(prev2.first_key, ep.misses, which=2)

        # Push this epoch into history
        self._prev_epochs.append(ep)
        logger.debug(
            "EBCP: end epoch first=%#x len=%d (history=%d)",
            ep.first_key,
            len(ep.misses),
            len(self._prev_epochs),
        )

    # --------------------------- Correlation Table -----------------------------

    def _predict_for_first_miss(self, first_key: int) -> List[int]:
        preds: List[int] = []
        entry: Optional[CorrelEntry] = self.table.get(first_key)
        if isinstance(entry, CorrelEntry):
            # degree 1
            take1 = entry.succ1[: self.max_targets_per_epoch]
            preds.extend(take1)
            # degree 2 (optional)
            if self.degree_epochs >= 2 and entry.succ2:
                take2 = entry.succ2[: self.max_targets_per_epoch]
                preds.extend(a for a in take2 if a not in preds)
            if preds:
                logger.debug(
                    "EBCP: lookup first=%#x -> succ1=%d succ2=%d (issued=%d)",
                    first_key,
                    len(entry.succ1),
                    len(entry.succ2),
                    len(preds),
                )
        return preds

    def _update_table(self, key: int, misses: List[int], which: int):
        entry = self.table.get(key)
        if not isinstance(entry, CorrelEntry):
            entry = CorrelEntry()
        entry.merge_succ(which, misses, self.max_targets_per_epoch)
        self.table.put(key, entry)
        logger.debug(
            "EBCP: update table key=%#x which=%d add=%d totals=(%d,%d)",
            key,
            which,
            len(misses),
            len(entry.succ1),
            len(entry.succ2),
        )

    # ---------------------------- Working-Set Filter ---------------------------

    def _classify_and_update_ws(self, key: int) -> bool:
        """
        Return True if 'key' is classified as a (proxy) miss: not present in the WS filter.
        Maintains an LRU-ordereddict capped at ws_capacity.
        """
        miss = key not in self._ws
        if not miss:
            # hit → MRU
            self._ws.move_to_end(key, last=True)
        else:
            # miss → insert and evict LRU if full
            self._ws[key] = None
            if len(self._ws) > self.ws_capacity:
                self._ws.popitem(last=False)
        return miss

    # --------------------------------- Utils -----------------------------------

    def _align(self, addr: int) -> int:
        mask = ~(self.line_size - 1)
        return addr & mask
