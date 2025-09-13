from prefetchlenz.prefetchingalgorithm.impl.ghb import GlobalHistoryBufferPrefetcher
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

# ---------- Tests ----------


def test_init_and_close_clears():
    p = GlobalHistoryBufferPrefetcher(ghb_size=8, search_depth=4, degree=2)
    p.init()
    # insert a couple of accesses
    a1 = MemoryAccess(pc=0x10, address=0x1000)
    p.progress(a1, prefetch_hit=False)
    a2 = MemoryAccess(pc=0x11, address=0x2000)
    p.progress(a2, prefetch_hit=False)

    # close should clear internal structures
    p.close()
    # GHB empty, index empty, no outstanding
    assert len(p.ghb._uid_to_index) == 0
    assert len(p.index) == 0
    assert len(p.outstanding) == 0


def test_simple_stride_prefetch():
    p = GlobalHistoryBufferPrefetcher(ghb_size=16, search_depth=4, degree=2)
    p.init()

    pc = 0x20
    base = 0x1000
    stride = 0x40

    # feed three sequential accesses forming a clear stride
    p.progress(MemoryAccess(pc=pc, address=base), prefetch_hit=False)
    p.progress(MemoryAccess(pc=pc, address=base + stride), prefetch_hit=False)
    # third access should allow detection: previous entries exist
    preds = p.progress(
        MemoryAccess(pc=pc, address=base + 2 * stride), prefetch_hit=False
    )

    # Expect degree=2 prefetches: next addresses base+3*stride and base+4*stride
    expected = [hex((base + 2 * stride) + stride * i) for i in (1, 2)]
    assert preds, "Expected prefetches to be issued"
    got_hex = [hex(x) for x in preds]
    assert got_hex == expected, f"Expected {expected} got {got_hex}"


def test_outstanding_credit_and_delta_credit():
    # set late_time_threshold=0 to mark credited prefetch hits as late immediately
    p = GlobalHistoryBufferPrefetcher(
        ghb_size=32, search_depth=4, degree=2, late_time_threshold=0
    )
    p.init()

    pc = 0x30
    base = 0x3000
    stride = 0x40

    # produce prefetches by creating stride pattern
    p.progress(MemoryAccess(pc=pc, address=base), prefetch_hit=False)
    p.progress(MemoryAccess(pc=pc, address=base + stride), prefetch_hit=False)
    preds = p.progress(
        MemoryAccess(pc=pc, address=base + 2 * stride), prefetch_hit=False
    )
    assert preds, "prefetches should be produced"

    # simulate a prefetch hit on the first issued target
    hit_addr = preds[0]
    # call progress with prefetch_hit=True and access to the hit address
    p.progress(MemoryAccess(pc=pc, address=hit_addr), prefetch_hit=True)

    # check that a delta credit exists for the origin PC (some credit should be recorded)
    top = p.corr.top_delta(pc)
    assert (
        top is not None
    ), "Delta should have been credited for the PC after prefetch hit"


def test_ghb_eviction_and_chain_truncation():
    # small GHB forces evictions quickly
    p = GlobalHistoryBufferPrefetcher(ghb_size=2, search_depth=4, degree=1)
    p.init()

    pc = 0x40
    a1 = MemoryAccess(pc=pc, address=0x1000)
    uid1_preds = p.progress(a1, prefetch_hit=False)  # inserts entry 1
    # capture uid of the head after first insert
    head1 = p.index.get_head(pc)
    assert head1 is not None

    # insert second entry (fills GHB)
    p.progress(MemoryAccess(pc=pc, address=0x1100), prefetch_hit=False)
    head2 = p.index.get_head(pc)
    assert head2 is not None and head2 != head1

    # insert third entry causing eviction of the oldest entry (uid1)
    p.progress(MemoryAccess(pc=pc, address=0x1200), prefetch_hit=False)
    # now uid1 should have been evicted from GHB
    assert p.ghb.get_entry(head1) is None, "Oldest GHB entry should be evicted"


def test_index_table_eviction_lru_behavior():
    # tiny index table to force eviction of earlier pcs
    p = GlobalHistoryBufferPrefetcher(ghb_size=8, line_size=64, index_capacity=1)
    p.init()

    # insert a head for pc A then pc B -> A should be evicted due to capacity=1
    p.progress(MemoryAccess(pc=0xA, address=0x1000), prefetch_hit=False)
    p.progress(MemoryAccess(pc=0xB, address=0x2000), prefetch_hit=False)

    # pc A head should no longer be present
    assert p.index.get_head(0xA) is None


def test_throttling_by_outstanding_limit():
    # outstanding_limit small to trigger throttle
    p = GlobalHistoryBufferPrefetcher(ghb_size=16, degree=2, outstanding_limit=2)
    p.init()

    pc = 0x50
    base = 0x5000
    stride = 0x40

    # produce some prefetches until outstanding grows
    p.progress(MemoryAccess(pc=pc, address=base), prefetch_hit=False)
    preds = p.progress(MemoryAccess(pc=pc, address=base + stride), prefetch_hit=False)
    assert preds, "initial prefetches expected"

    # issue more accesses that would normally cause more prefetches
    # but outstanding limit should prevent further issuance once reached
    # call repeatedly; once outstanding reach limit further calls should return []
    blocked = False
    for i in range(10):
        new_preds = p.progress(
            MemoryAccess(pc, base + (3 + i) * stride), prefetch_hit=False
        )
        if not new_preds:
            blocked = True
            break
    assert (
        blocked
    ), "Throttling should block further prefetches once outstanding limit reached"
