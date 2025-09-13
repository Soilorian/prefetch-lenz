from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.correlation import CorrelationPrefetcher


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
