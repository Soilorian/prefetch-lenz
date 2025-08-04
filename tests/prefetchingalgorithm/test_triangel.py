import pytest

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.triangel import TriangelPrefetcher
from prefetchlenz.util.size import Size


@pytest.fixture
def triangel():
    # tiny table so we can force eviction/resizing
    p = TriangelPrefetcher(
        init_size=Size(2),
        min_size=Size(1),
        max_size=Size(4),
        resize_epoch=4,
        grow_thresh=0.5,
        shrink_thresh=0.25,
    )
    p.init()
    # force HistorySampler and SecondChanceSampler to always signal reuse/pattern
    p.history.sample = lambda pc, addr, ts: (True, True)
    p.second_chance.sample = lambda pc, tgt, ts: True
    return p


def test_no_predict_before_training(triangel):
    # First access of PC=1: no prev → no prediction
    assert triangel.progress(MemoryAccess(address=0x10, pc=1), prefetch_hit=False) == []
    # Still none on first access of a new PC
    assert triangel.progress(MemoryAccess(address=0x20, pc=2), prefetch_hit=False) == []


def test_simple_training_and_prediction(triangel):
    # Train PC=1: A->B
    triangel.progress(MemoryAccess(address=0xA, pc=1), prefetch_hit=False)
    triangel.progress(MemoryAccess(address=0xB, pc=1), prefetch_hit=False)
    # Next see A: should predict B
    preds = triangel.progress(MemoryAccess(address=0xA, pc=1), prefetch_hit=False)
    assert preds == [0xB]


def test_lru_eviction_on_capacity(triangel):
    # With capacity=2 entries, insert 3 PCs → evict lowest‐score
    # Train PC1: A->B (will get score++ on prediction)
    triangel.progress(MemoryAccess(0xA, pc=1), prefetch_hit=False)
    triangel.progress(MemoryAccess(0xB, pc=1), prefetch_hit=False)
    triangel.progress(MemoryAccess(0xA, pc=1), prefetch_hit=False)  # score of A→B = 1

    # Train PC2: C->D (no prediction yet, so score=0)
    triangel.progress(MemoryAccess(0xC, pc=2), prefetch_hit=False)
    triangel.progress(MemoryAccess(0xD, pc=2), prefetch_hit=False)

    # At this point table keys should be {A, C}
    assert set(triangel.table.keys()) == {0xA, 0xC}

    # Train PC3: E->F → forces eviction of C (score=0 < score of A)
    triangel.progress(MemoryAccess(0xE, pc=3), prefetch_hit=False)
    triangel.progress(MemoryAccess(0xF, pc=3), prefetch_hit=False)

    keys = set(triangel.table.keys())
    assert 0xC not in keys
    assert keys == {0xA, 0xE}


def test_dynamic_grow_and_shrink(triangel):
    # Initially size = 2 entries
    assert int(triangel.size) == 2

    # Drive 4 accesses that all produce a useful prefetch → ratio = 4/4 =1 > grow_thresh
    # Need a single mapping: A->C
    triangel.progress(MemoryAccess(0xA, pc=1), prefetch_hit=False)
    triangel.progress(MemoryAccess(0xC, pc=1), prefetch_hit=False)
    for _ in range(4):
        triangel.progress(MemoryAccess(0xA, pc=1), prefetch_hit=False)

    assert int(triangel.size) > 2, "Table should have grown"

    grown = int(triangel.size)

    # Now 4 accesses all hit prefetched lines → ratio=0/4 < shrink_thresh → shrink
    for _ in range(4):
        triangel.progress(MemoryAccess(0xA, pc=1), prefetch_hit=True)

    assert int(triangel.size) < grown, "Table should have shrunk"


def test_metadata_reuse_buffer(triangel):
    # MRB size is 256 by default; we test that probe/insert work
    addr = 0xABC
    assert not triangel.mrb.probe(addr)
    triangel.mrb.insert(addr)
    assert triangel.mrb.probe(addr)


def test_set_dueller_basic(triangel):
    # Best partition starts at 0 ways
    init_part = triangel.dueller.best_partition()
    assert isinstance(init_part, int)

    # Recording data hits should bias toward lower ways
    triangel.dueller.record_data_hit(init_part)
    assert triangel.dueller.best_partition() >= init_part

    # Recording meta hits should bias toward higher ways
    triangel.dueller.record_meta_hit(init_part)
    assert triangel.dueller.best_partition() >= init_part
