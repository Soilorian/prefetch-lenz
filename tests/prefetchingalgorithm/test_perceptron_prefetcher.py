import pytest

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.perceptron import PerceptronPrefetcher


def test_basic_stride_predictions():
    p = PerceptronPrefetcher()
    p.init()

    pc = 0x10
    base = 1000
    accesses = [MemoryAccess(address=base + i * 64, pc=pc) for i in range(5)]

    preds = []
    for acc in accesses:
        preds.extend(p.progress(acc, prefetch_hit=False))

    # Should have made some predictions
    assert len(preds) > 0
    assert all(isinstance(addr, int) for addr in preds)


def test_prefetch_table_training():
    p = PerceptronPrefetcher()
    p.init()

    pc = 0x20
    acc1 = MemoryAccess(address=2000, pc=pc)
    acc2 = MemoryAccess(address=2064, pc=pc)  # stride=64

    p.progress(acc1, prefetch_hit=False)
    preds = p.progress(acc2, prefetch_hit=False)

    # There should be candidates in the table
    for cand in preds:
        assert cand in p.prefetch_table or cand in p.reject_table


def test_reject_then_later_used():
    p = PerceptronPrefetcher()
    p.init()

    pc = 0x30
    acc1 = MemoryAccess(address=3000, pc=pc)
    acc2 = MemoryAccess(address=3064, pc=pc)

    p.progress(acc1, prefetch_hit=False)
    preds = p.progress(acc2, prefetch_hit=False)

    # Force rejection by setting high TAU
    p.TAU_LO = 100
    acc3 = MemoryAccess(address=3128, pc=pc)
    p.progress(acc3, prefetch_hit=False)

    # Now the rejected address should move to reject_table
    assert any(entry.address for entry in p.reject_table.values())
