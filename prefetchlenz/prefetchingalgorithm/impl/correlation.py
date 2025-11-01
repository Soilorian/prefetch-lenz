"""
Correlation Prefetcher

Algorithm: Using a User-Level Memory Thread for Correlation Prefetching by Solihin et al.

This prefetcher learns address correlations from the access stream. It tracks
patterns where accessing address A is often followed by address B. When such
correlations are detected with sufficient confidence, the prefetcher issues
prefetches for the predicted addresses.

Key Components:
- CorrelationTable: Global table mapping trigger addresses to predicted addresses
- CorrelationTableEntry: Per-trigger entry storing multiple predictions with confidence
- PredictionWithConfidence: Represents a single prediction with its confidence counter
- CorrelationPrefetcher: Main prefetcher class implementing the algorithm

How it works:
1. On each access, record correlation from previous access to current address
2. On prefetch hits, reinforce the correlation between previous and current address
3. Check if current address has high-confidence predictions and issue prefetches
4. Confidence counters track prediction accuracy, with weak predictions decaying over time
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from prefetchlenz.cache.Cache import Cache
from prefetchlenz.cache.replacementpolicy.impl.lru import LruReplacementPolicy
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class PredictionWithConfidence:
    """Represents a predicted address with an associated confidence counter.

    Attributes:
        prediction: The predicted memory address.
        confidence: Confidence counter for this prediction.
    """

    prediction: int
    confidence: int

    @classmethod
    def create(cls, prediction: int) -> "PredictionWithConfidence":
        """Factory method for creating a prediction with initial confidence of 1."""
        return PredictionWithConfidence(prediction=prediction, confidence=1)


@dataclass
class CorrelationTableEntry:
    """Holds correlation predictions for a single trigger address.

    Each entry tracks up to `prediction_limit` predictions with confidence.

    Attributes:
        trigger: The trigger address that this entry corresponds to.
        predictions: List of predictions with their confidence counters.
        prediction_limit: Maximum number of predictions to store.
        discard_threshold: Minimum confidence below which predictions are discarded.
        max_confidence: Maximum confidence value to prevent runaway growth.
    """

    trigger: int
    predictions: List[PredictionWithConfidence]
    prediction_limit: int = 4
    discard_threshold: int = 0
    max_confidence: int = 3  # Cap confidence to prevent runaway growth.

    def update(self, prediction: int) -> None:
        """Update entry with a new observed prediction.

        If the prediction already exists, increment its confidence.
        Otherwise, add it if space is available. If the table is full,
        decay the weakest entry and replace it if confidence is below threshold.
        If confidence > discard_threshold, only decay (don't replace).
        """
        for pred_conf in self.predictions:
            if pred_conf.prediction == prediction:
                pred_conf.confidence = min(
                    pred_conf.confidence + 1, self.max_confidence
                )
                return

        if len(self.predictions) < self.prediction_limit:
            self.predictions.append(PredictionWithConfidence.create(prediction))
            return

        # Table is full: find weakest entry
        weakest = min(self.predictions, key=lambda p: p.confidence)
        # Check confidence before decay
        old_confidence = weakest.confidence
        # Decay weakest
        weakest.confidence = max(0, weakest.confidence - 1)
        # Replace only if confidence was > threshold before decay and dropped to <= threshold after decay
        # OR if confidence was already <= threshold before decay
        if (
            old_confidence > self.discard_threshold
            and weakest.confidence <= self.discard_threshold
        ):
            # Entry was above threshold, but decayed below threshold - replace it
            self.predictions.remove(weakest)
            self.predictions.append(PredictionWithConfidence.create(prediction))
        elif old_confidence <= self.discard_threshold:
            # Entry was already at or below threshold - replace it
            self.predictions.remove(weakest)
            self.predictions.append(PredictionWithConfidence.create(prediction))
        # Otherwise (confidence still > threshold after decay), we only decayed (don't replace)


class CorrelationTable:
    """Global correlation table mapping triggers to predictions with confidence.

    Uses an LRU replacement policy to manage limited entries.

    Attributes:
        selection_threshold: Minimum confidence required for issuing a prefetch.
        storage: Cache storage for correlation table entries.
    """

    selection_threshold = 2
    storage: Cache = Cache(
        num_ways=64,
        num_sets=1,
        replacement_policy_cls=LruReplacementPolicy,
    )

    def get(self, trigger: int) -> Optional[CorrelationTableEntry]:
        """Return the correlation entry for the given trigger address."""
        return self.storage.get(trigger)

    def put(self, trigger: int, prediction: int) -> None:
        """Record or update the correlation between trigger and prediction."""
        entry = self.storage.get(trigger)
        if entry is None:
            entry = CorrelationTableEntry(trigger=trigger, predictions=[])
            self.storage.put(trigger, entry)
        entry.update(prediction)

    def predictions_for(self, trigger: int) -> List[int]:
        """Return predictions for the trigger that meet the confidence threshold."""
        entry = self.storage.get(trigger)
        if entry is None:
            return []
        return [
            pred_conf.prediction
            for pred_conf in entry.predictions
            if pred_conf.confidence >= self.selection_threshold
        ]


class CorrelationPrefetcher(PrefetchAlgorithm):
    """Correlation-based prefetcher.

    Observes address stream, learns correlations (trigger -> predicted addresses),
    and issues prefetches when confidence is high enough.
    """

    def __init__(self):
        self.table = CorrelationTable()
        self.last_access: Optional[int] = None

    def init(self) -> None:
        self.last_access = None

    def close(self) -> None:
        self.last_access = None

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """Process memory access and return prefetch addresses.

        On prefetch hits, we reinforce the correlation between the previous
        and current address. We always check for high-confidence predictions
        and update the correlation table with the last access pattern.
        """
        current_address = access.address
        prefetches: List[int] = []

        if prefetch_hit and self.last_access is not None:
            self.table.put(self.last_access, current_address)

        predictions = self.table.predictions_for(current_address)
        if predictions:
            prefetches.extend(predictions)

        if self.last_access is not None:
            self.table.put(self.last_access, current_address)

        self.last_access = current_address
        return prefetches
