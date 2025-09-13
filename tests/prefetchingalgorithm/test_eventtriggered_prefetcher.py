from prefetchlenz.prefetchingalgorithm.access.eventtriggeredmemoryaccess import (
    EventTriggeredMemoryAccess,
)
from prefetchlenz.prefetchingalgorithm.impl.eventtriggered import (
    EventTriggeredPrefetcher,
    FilterSpec,
)


def make_access(pc, addr, latency=50, pressure=2.0):
    return EventTriggeredMemoryAccess(
        pc=pc, address=addr, accessLatency=latency, bandwidthPressure=pressure
    )


def test_stride_prefetching():
    p = EventTriggeredPrefetcher()
    p.init()
    base_pc = 0x10

    p.filters.add_filter(FilterSpec(pcs=(base_pc,)))

    # Feed sequential addresses with stride = 64
    addrs = [0x1000, 0x1040, 0x1080]
    for addr in addrs:
        preds = p.progress(make_access(base_pc, addr), prefetch_hit=False)

    # Should learn stride and issue predictions
    preds = p.progress(make_access(base_pc, 0x10C0), prefetch_hit=False)
    assert any(0x10C0 < pred < 0x1140 for pred in preds)


def test_pressure_blocks_prefetching():
    p = EventTriggeredPrefetcher()
    p.init()
    p.filters.add_filter(FilterSpec(pcs=(0x20,)))

    # Access with high bandwidthPressure → scheduler blocks
    access = make_access(0x20, 0x2000, latency=100, pressure=50.0)
    preds = p.progress(access, prefetch_hit=False)
    assert preds == []


def test_latency_gate_allows_prefetch():
    p = EventTriggeredPrefetcher()
    p.init()
    p.filters.add_filter(FilterSpec(pcs=(0x30,)))
    p.ppu.latency_min = 20

    # Low latency → block
    a1 = make_access(0x30, 0x3000, latency=5)
    preds1 = p.progress(a1, prefetch_hit=False)
    assert preds1 == []

    # High latency + stride → allow
    a2 = make_access(0x30, 0x3040, latency=100)
    p.progress(a2, prefetch_hit=False)
    a3 = make_access(0x30, 0x3080, latency=100)
    preds3 = p.progress(a3, prefetch_hit=False)
    assert preds3 != []
