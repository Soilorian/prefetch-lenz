import logging
from dataclasses import dataclass
from typing import List, Optional

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.lru import LruReplacementPolicy
from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.correlation")


@dataclass
class PredictionWithConfidence:
    """
    Represents a predicted address with an associated confidence counter.
    """

    prediction: int
    confidence: int

    @classmethod
    def create(cls, prediction: int):
        return PredictionWithConfidence(prediction=prediction, confidence=1)


@dataclass
class CorrelationTableEntry:
    """
    Holds correlation predictions for a single trigger address.
    Each entry tracks up to `prediction_limit` predictions with confidence.
    """

    trigger: int
    predictions: List[PredictionWithConfidence]
    prediction_limit: int = 4
    discard_threshold: int = 0
    max_confidence: int = 3  # cap confidence to prevent runaway growth

    def update(self, prediction: int):
        """
        Update the correlation entry with a new observed prediction.
        - If prediction already present, increment confidence.
        - If free slot available, add as new prediction.
        - If full, decay weakest entry and replace if confidence below threshold.
        """
        # Check if already present
        for pwc in self.predictions:
            if pwc.prediction == prediction:
                pwc.confidence = min(pwc.confidence + 1, self.max_confidence)
                return

        # Add new if space available
        if len(self.predictions) < self.prediction_limit:
            self.predictions.append(PredictionWithConfidence.create(prediction))
            return

        # Otherwise, decay weakest prediction
        weakest = min(self.predictions, key=lambda p: p.confidence)
        weakest.confidence -= 1
        if weakest.confidence <= self.discard_threshold:
            self.predictions.remove(weakest)
            self.predictions.append(PredictionWithConfidence.create(prediction))


class CorrelationTable:
    """
    Global correlation table mapping triggers -> predictions with confidence.
    Uses an LRU replacement policy to manage limited entries.
    """

    selection_threshold = 2  # minimum confidence required for issuing a prefetch
    storage: Cache = Cache(
        num_ways=64,  # number of entries
        num_sets=1,
        replacement_policy_cls=LruReplacementPolicy,
    )

    def get(self, trigger: int) -> Optional[CorrelationTableEntry]:
        return self.storage.get(trigger)

    def put(self, trigger: int, prediction: int):
        """
        Update correlation for (trigger -> prediction).
        Creates new entry if trigger not yet in table.
        """
        entry: Optional[CorrelationTableEntry] = self.storage.get(trigger)
        if entry is None:
            entry = CorrelationTableEntry(trigger=trigger, predictions=[])
            self.storage.put(trigger, entry)
        entry.update(prediction)

    def predictions_for(self, trigger: int) -> List[int]:
        """
        Return a list of predictions above the confidence threshold.
        """
        entry: Optional[CorrelationTableEntry] = self.storage.get(trigger)
        if entry is None:
            return []
        return [
            p.prediction
            for p in entry.predictions
            if p.confidence >= self.selection_threshold
        ]


class CorrelationPrefetcher(PrefetchAlgorithm):
    """
    Correlation-based prefetcher.
    Observes address stream, learns correlations (trigger -> predicted addresses),
    and issues prefetches when confidence is high enough.
    """

    def __init__(self):
        self.table = CorrelationTable()
        self.last_access: Optional[int] = None

    def init(self):
        """Initialize state before simulation."""
        self.last_access = None

    def close(self):
        """Clean up state after simulation."""
        self.last_access = None

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Advance prefetcher on new memory access.
        - If prefetch hit observed, reinforce correlation.
        - If current address has high-confidence predictions, issue prefetches.
        - Always update correlation (last_access -> current).
        """
        addr = access.address
        prefetches: List[int] = []

        # reinforce on prefetch hit
        if prefetch_hit and self.last_access is not None:
            self.table.put(self.last_access, addr)

        # prefetch if predictions are strong enough
        preds = self.table.predictions_for(addr)
        if preds:
            prefetches.extend(preds)

        # update correlation from last access
        if self.last_access is not None:
            self.table.put(self.last_access, addr)

        self.last_access = addr
        return prefetches
