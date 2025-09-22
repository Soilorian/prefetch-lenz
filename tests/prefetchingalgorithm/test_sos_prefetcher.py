from prefetchlenz.prefetchingalgorithm.access.storeorderedmemoryaccess import (
    StoreOrderedMemoryAccess,
)
from prefetchlenz.prefetchingalgorithm.impl.storeorderedstreamer import (
    StoreOrderedStreamer,
)

LINE = 64


def make_store(addr: int, tid: int = 0) -> StoreOrderedMemoryAccess:
    return StoreOrderedMemoryAccess(address=addr, pc=0, tid=tid, isWrite=True)


def make_load(addr: int, tid: int = 0) -> StoreOrderedMemoryAccess:
    return StoreOrderedMemoryAccess(address=addr, pc=0, tid=tid, isWrite=False)


def test_producer_recording_and_consumer_prefetch():
    sos = StoreOrderedStreamer(line_size=LINE, prefetch_degree=3)
    sos.init()

    # Producer writes 4 sequential lines
    for i in range(4):
        sos.progress(make_store(i * LINE, tid=1), prefetch_hit=False)

    # Consumer reads line 0 → should prefetch lines 1,2,3
    preds = sos.progress(make_load(0 * LINE, tid=2), prefetch_hit=False)
    assert preds == [1 * LINE, 2 * LINE, 3 * LINE]


def test_deduplication_of_consecutive_stores():
    sos = StoreOrderedStreamer(line_size=LINE, dedup_consecutive=True)
    sos.init()

    sos.progress(make_store(100 * LINE, tid=1), prefetch_hit=False)
    sos.progress(make_store(100 * LINE, tid=1), prefetch_hit=False)  # duplicate

    stream = list(sos.producer_streams[1])
    assert stream == [100], "Consecutive duplicate should not be added"


def test_consumer_triggers_only_if_line_in_stream():
    sos = StoreOrderedStreamer(line_size=LINE, prefetch_degree=2)
    sos.init()

    # No producer stores yet → no prefetch
    preds = sos.progress(make_load(10 * LINE, tid=1), prefetch_hit=False)
    assert preds == []

    # Now record a producer store
    sos.progress(make_store(10 * LINE, tid=2), prefetch_hit=False)
    sos.progress(make_store(11 * LINE, tid=2), prefetch_hit=False)

    # Consumer reads line 10 → should prefetch line 11
    preds = sos.progress(make_load(10 * LINE, tid=1), prefetch_hit=False)
    assert preds == [11 * LINE]


def test_outstanding_prefetch_tracking():
    sos = StoreOrderedStreamer(line_size=LINE, prefetch_degree=2)
    sos.init()

    sos.progress(make_store(0 * LINE, tid=1), prefetch_hit=False)
    sos.progress(make_store(1 * LINE, tid=1), prefetch_hit=False)
    sos.progress(make_store(2 * LINE, tid=1), prefetch_hit=False)

    # Consumer read at line 0 → prefetch 1,2
    preds1 = sos.progress(make_load(0 * LINE, tid=2), prefetch_hit=False)
    assert set(preds1) == {1 * LINE, 2 * LINE}

    # A second read at line 0 should not re-issue prefetches (still outstanding)
    preds2 = sos.progress(make_load(0 * LINE, tid=2), prefetch_hit=False)
    assert preds2 == []

    # Simulate completion of line 1 prefetch
    sos.prefetch_completed(1 * LINE)
    preds3 = sos.progress(make_load(0 * LINE, tid=2), prefetch_hit=False)
    assert preds3 == [1 * LINE] or preds3 == [2 * LINE]


def test_stream_window_limit():
    sos = StoreOrderedStreamer(line_size=LINE, stream_window=4, prefetch_degree=2)
    sos.init()

    tid = 1
    # Add more than 4 stores → old ones are evicted from deque
    for i in range(10):
        sos.progress(make_store(i * LINE, tid=tid), prefetch_hit=False)

    stream = list(sos.producer_streams[tid])
    assert len(stream) <= 4, "Stream should not exceed window size"
    assert stream == [6, 7, 8, 9], "Stream should contain latest entries only"
