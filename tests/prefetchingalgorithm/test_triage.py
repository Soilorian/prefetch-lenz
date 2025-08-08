import pytest

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.triage import TriagePrefetcher
from prefetchlenz.util.size import Size


def test_no_prefetch_before_training():
    """
    Should never predict on the very first access of a PC.
    """
    triage = TriagePrefetcher(init_size=Size(4))  # allow up to 4 entries
    triage.init()

    # first ever access for PC=1 → no mapping → no prediction
    assert triage.progress(MemoryAccess(address=0x10, pc=1), prefetch_hit=False) == []

    # still no prediction on the second access if different PC
    assert triage.progress(MemoryAccess(address=0x20, pc=2), prefetch_hit=False) == []


def test_simple_training_then_predict():
    """
    After seeing A→B for PC=1, a new access at A (for same PC) predicts B.
    """
    triage = TriagePrefetcher(init_size=Size(4))
    triage.init()

    # Train PC=1: A->B
    triage.progress(MemoryAccess(address=0xA, pc=1), prefetch_hit=False)
    triage.progress(MemoryAccess(address=0xB, pc=1), prefetch_hit=False)

    # Now PC=1 sees A again → should predict B
    preds = triage.progress(MemoryAccess(address=0xA, pc=1), prefetch_hit=False)
    assert preds == [0xB]


def test_hawkeye_eviction_lowest_score():
    """
    With a capacity of 2 entries, inserting a 3rd mapping evicts
    the one with the lowest 'score'.
    """
    # Capacity = 2 entries
    triage = TriagePrefetcher(init_size=Size(2))
    triage.init()

    # Train PC=1: A->B  (will get one correct prediction, score→1)
    triage.progress(MemoryAccess(address=0xA, pc=1), prefetch_hit=False)
    triage.progress(MemoryAccess(address=0xB, pc=1), prefetch_hit=False)
    # Cause the prefetch to be issued (A→B), and mark it useful
    triage.progress(MemoryAccess(address=0xA, pc=1), prefetch_hit=False)
    triage.progress(MemoryAccess(address=0xB, pc=1), prefetch_hit=False)

    # Train PC=2: C->D  (no subsequent prefetch, score stays 0)
    triage.progress(MemoryAccess(address=0xD, pc=2), prefetch_hit=False)
    triage.progress(MemoryAccess(address=0xE, pc=2), prefetch_hit=False)

    # At this point we have two entries: {A→B (score=1), D→E (score=0)}
    assert 0xA in triage.cache and 0xD in triage.cache
    assert triage.cache.get(0xA).confidence == 1
    assert triage.cache.get(0xD).confidence == 0

    # Insert a third mapping PC=3: E->F → triggers eviction of lowest-score (C→D)
    triage.progress(MemoryAccess(address=0xE, pc=3), prefetch_hit=False)
    triage.progress(MemoryAccess(address=0xF, pc=3), prefetch_hit=False)

    assert 0xA not in triage.cache


def test_dynamic_grow_and_shrink():
    """
    Using a tiny resize_epoch, verify that the table grows when
    many useful prefetches occur and shrinks when none do.
    """
    # resize_epoch=4, grow_thresh=0.5, shrink_thresh=0.25
    triage = TriagePrefetcher(
        init_size=Size(4),
        min_size=Size(2),
        max_size=Size(8),
        resize_epoch=4,
        grow_thresh=0.8,
        shrink_thresh=0.2,
    )
    triage.init()

    # Prepare one mapping PC=1: 1->2
    triage.progress(MemoryAccess(address=0x1, pc=1), prefetch_hit=False)
    triage.progress(MemoryAccess(address=0x1, pc=1), prefetch_hit=False)
    triage.progress(MemoryAccess(address=0x1, pc=1), prefetch_hit=True)
    triage.progress(MemoryAccess(address=0x1, pc=1), prefetch_hit=False)

    # 4 accesses, all yield useful prefetches → ratio = 4/4 = 1.0 > 0.5 → grow
    for i in range(4):
        # revisit 1 to predict 2, and mark prefetch_hit=False to count useful
        triage.progress(MemoryAccess(address=0x1, pc=1), prefetch_hit=True)

    assert triage.num_ways == 2

    # Now do 4 accesses with prefetch_hit=True → ratio=0/4=0 < 0.25 → shrink
    for i in range(4):
        triage.progress(MemoryAccess(address=0x2 + i, pc=1), prefetch_hit=False)

    assert triage.num_ways == 1
