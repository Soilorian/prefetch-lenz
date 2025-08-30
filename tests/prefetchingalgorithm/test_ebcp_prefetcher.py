from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.ebcp import EbcpPrefetcher


def make_access(addr: int):
    """Helper to build MemoryAccess with aligned addr."""
    return MemoryAccess(address=addr, pc=0, cpu=0)


def test_epoch_creation_and_prediction():
    p = EbcpPrefetcher(line_size=64, quiescence_len=2, ws_capacity=8)
    p.init()

    # Train with repeating epochs: misses 0x1000,0x2000 then quiesce
    seq = [
        0x1000,
        0x2000,
        0x2000,
        0x2000,  # epoch 1
        0x3000,
        0x4000,
        0x4000,
        0x4000,
    ]  # epoch 2

    preds = []
    for addr in seq:
        preds = p.progress(make_access(addr), prefetch_hit=False)

    # After second epoch, correlation table should have linked epoch1→epoch2
    entry = p.table.get(0x1000)
    assert entry is not None, "First epoch head should be in table"
    assert len(entry.succ1) > 0, "Should have learned successor misses"


def test_issue_prefetch_on_new_epoch():
    p = EbcpPrefetcher(line_size=64, quiescence_len=1, ws_capacity=2)
    p.init()

    # Train epoch1 -> epoch2
    p.progress(make_access(0x1000), False)
    p.progress(make_access(0x2000), False)
    # quiesce
    p.progress(make_access(0x2000), False)
    p.progress(make_access(0x2000), False)

    p.progress(make_access(0x3000), False)
    p.progress(make_access(0x4000), False)
    # quiesce
    p.progress(make_access(0x4000), False)
    p.progress(make_access(0x4000), False)

    # Start epoch3 should issue prefetches learned from epoch1→epoch2
    preds = p.progress(make_access(0x1000), False)
    assert isinstance(preds, list)
    assert any(addr in preds for addr in [0x3000, 0x4000])


def test_usefulness_feedback_counts():
    p = EbcpPrefetcher(line_size=64, quiescence_len=1, ws_capacity=8)
    p.init()

    # Fake a prefetch hit
    p.progress(make_access(0x1000), False)
    p.progress(make_access(0x1000), False)
    p.progress(make_access(0x2000), False)
    p.progress(make_access(0x2000), False)
    p.progress(make_access(0x3000), False)
    p.progress(make_access(0x3000), False)
    p.progress(make_access(0x1000), False)
    p.progress(make_access(0xBEEF), prefetch_hit=True)

    assert p._useful == 1, "Useful counter should increment on prefetch hit"
