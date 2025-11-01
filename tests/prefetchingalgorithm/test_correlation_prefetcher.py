from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.correlation import (
    CorrelationPrefetcher,
    CorrelationTableEntry,
    PredictionWithConfidence,
)


def make_access(addr: int):
    return MemoryAccess(address=addr, pc=0)


def test_simple_correlation_learning():
    p = CorrelationPrefetcher()
    p.init()

    # Train with pattern: 1 -> 2 -> 3 -> 2 -> 3 (correlation 2 -> 3)
    accesses = [1, 2, 3, 2, 3]
    for a in accesses:
        p.progress(make_access(a), prefetch_hit=False)

    preds = p.table.predictions_for(2)
    assert 3 in preds, "Should have learned correlation 2 -> 3"


def test_prefetch_issued():
    p = CorrelationPrefetcher()
    p.init()

    # Train: 4 -> 5 multiple times to boost confidence
    for _ in range(3):
        p.progress(make_access(4), prefetch_hit=False)
        p.progress(make_access(5), prefetch_hit=False)

    # Now accessing 4 should issue prefetch for 5
    preds = p.progress(make_access(4), prefetch_hit=False)
    assert 5 in preds, "Prefetcher should predict 5 after 4"


def test_confidence_decay_and_replacement():
    p = CorrelationPrefetcher()
    p.init()

    # Fill predictions for trigger 10
    p.table.put(10, 11)
    p.table.put(10, 12)
    p.table.put(10, 13)
    p.table.put(10, 14)

    # Add new prediction 15 -> should cause weakest to decay/replace
    p.table.put(10, 15)

    preds = [pw.prediction for pw in p.table.get(10).predictions]
    assert len(preds) <= 4
    assert 15 in preds, "New prediction should have been added by replacement"


def test_prefetch_hit_reinforcement():
    """Test that prefetch hits reinforce correlations."""
    p = CorrelationPrefetcher()
    p.init()

    # First access
    p.progress(make_access(100), prefetch_hit=False)

    # Simulate a prefetch hit: accessing 101 after prefetching it
    p.progress(make_access(101), prefetch_hit=True)

    # Verify that correlation 100 -> 101 was reinforced
    entry = p.table.get(100)
    assert entry is not None
    predictions = [pw.prediction for pw in entry.predictions if pw.prediction == 101]
    assert len(predictions) > 0, "Prefetch hit should have reinforced correlation"


def test_first_access_no_correlation():
    """Test that first access doesn't try to record correlation."""
    p = CorrelationPrefetcher()
    p.init()

    # First access should not crash and should not have last_access
    assert p.last_access is None
    preds = p.progress(make_access(200), prefetch_hit=False)

    # Should not issue prefetches on first access
    assert len(preds) == 0
    # Should set last_access
    assert p.last_access == 200


def test_confidence_saturation():
    """Test that confidence saturates at max_confidence."""
    entry = CorrelationTableEntry(trigger=300, predictions=[])

    # Add prediction and increment to max
    entry.update(301)
    entry.update(301)
    entry.update(301)

    # Find the prediction
    pred = next(pw for pw in entry.predictions if pw.prediction == 301)
    assert pred.confidence == 3, "Confidence should saturate at max_confidence (3)"

    # One more increment should not exceed max
    entry.update(301)
    assert pred.confidence == 3, "Confidence should not exceed max_confidence"


def test_decay_without_replacement():
    """Test decay of weakest entry when confidence > discard_threshold."""
    entry = CorrelationTableEntry(trigger=400, predictions=[], discard_threshold=0)

    # Fill predictions with confidence >= 1
    for i in range(5, 9):
        entry.predictions.append(PredictionWithConfidence(prediction=i, confidence=2))

    # Update with new prediction - should decay weakest but not replace
    entry.update(10)

    # All entries should still exist (none removed)
    assert len(entry.predictions) == 4, "No entry removed when confidence > threshold"
    # Weakest should have been decayed
    confidences = [pw.confidence for pw in entry.predictions]
    assert 1 in confidences, "Weakest entry should have been decayed"


def test_predictions_for_threshold_filtering():
    """Test that predictions_for filters by confidence threshold."""
    from prefetchlenz.prefetchingalgorithm.impl.correlation import CorrelationTable

    entry = CorrelationTableEntry(trigger=500, predictions=[])

    # Add predictions with different confidences
    entry.predictions.append(PredictionWithConfidence(prediction=501, confidence=1))
    entry.predictions.append(PredictionWithConfidence(prediction=502, confidence=2))
    entry.predictions.append(PredictionWithConfidence(prediction=503, confidence=3))

    table = CorrelationTable()
    table.storage.put(500, entry)

    # Only predictions with confidence >= 2 should be returned
    preds = table.predictions_for(500)
    assert 501 not in preds, "Low confidence prediction should be filtered"
    assert 502 in preds, "Threshold confidence prediction should be included"
    assert 503 in preds, "High confidence prediction should be included"


def test_init_and_close():
    """Test that init and close properly reset state."""
    p = CorrelationPrefetcher()
    p.init()

    # Set state
    p.progress(make_access(600), prefetch_hit=False)
    assert p.last_access == 600

    # Close should reset
    p.close()
    assert p.last_access is None

    # Init should also reset
    p.progress(make_access(700), prefetch_hit=False)
    p.init()
    assert p.last_access is None


def test_empty_predictions_for():
    """Test predictions_for with trigger that has no entry."""
    p = CorrelationPrefetcher()
    p.init()

    # Query non-existent trigger
    preds = p.table.predictions_for(9999)
    assert preds == [], "Should return empty list for non-existent trigger"


def test_multiple_predictions_same_trigger():
    """Test that multiple predictions can exist for same trigger."""
    p = CorrelationPrefetcher()
    p.init()

    # Learn multiple correlations for same trigger
    p.progress(make_access(1000), prefetch_hit=False)
    p.progress(make_access(1001), prefetch_hit=False)
    p.progress(make_access(1000), prefetch_hit=False)
    p.progress(make_access(1002), prefetch_hit=False)
    p.progress(make_access(1000), prefetch_hit=False)
    p.progress(make_access(1003), prefetch_hit=False)

    # Should have multiple predictions for trigger 1000
    entry = p.table.get(1000)
    assert entry is not None
    assert len(entry.predictions) > 1, "Should track multiple predictions per trigger"
