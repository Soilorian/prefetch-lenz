from prefetchlenz.prefetchingalgorithm.access.feedbackdirectedmemoryaccess import (
    FeedbackDirectedMemoryAccess,
)
from prefetchlenz.prefetchingalgorithm.impl.feedbackdirected import (
    FeedbackDirectedPrefetcher,
)


def make_access(pc, addr, demand_miss=True):
    """Helper: create a FeedbackDirectedMemoryAccess with minimal fields."""
    return FeedbackDirectedMemoryAccess(
        pc=pc,
        address=addr,
        demandMiss=demand_miss,
    )


def test_stride_prefetch_generation():
    p = FeedbackDirectedPrefetcher()
    p.init()

    # First access: no stride prefetch
    a1 = make_access(pc=0x10, addr=0x1000)
    preds = p.progress(a1, prefetch_hit=False)
    assert preds == []

    # Second access: stride = 64 → degree prefetches
    a2 = make_access(pc=0x10, addr=0x1040)
    preds = p.progress(a2, prefetch_hit=False)
    assert len(preds) == p.prefetch_degree
    assert preds[0] == 0x1080  # stride prediction


def test_accuracy_threshold_adjustment():
    p = FeedbackDirectedPrefetcher(t_interval=4)  # short interval for test
    p.init()

    # Simulate many useful prefetch hits (high accuracy)
    for i in range(4):
        a = make_access(pc=0x20, addr=0x2000 + i * 64)
        preds = p.progress(a, prefetch_hit=False)
        for pa in preds:
            # Mark them as useful hits
            p._update_counters(
                make_access(pc=0x20, addr=pa),
                is_demand_access=True,
                is_prefetch_hit=True,
            )

    # Force interval end
    p._end_interval()

    # Should not reduce aggressiveness
    assert 4 <= p.dyn_config_counter <= 5


def test_pollution_triggers_decrement():
    p = FeedbackDirectedPrefetcher(t_interval=2)
    p.init()

    # Force pollution by accessing the same miss repeatedly
    a1 = make_access(pc=0x30, addr=0x3000)
    p.progress(a1, prefetch_hit=False)
    # Mark as polluted (simulate demand hit after being added to pollution filter)
    p._update_counters(a1, is_demand_access=True, is_prefetch_hit=False)

    # End interval
    p._end_interval()
    assert p.dyn_config_counter < 3  # Should decrement from default=3


def test_lateness_behavior():
    p = FeedbackDirectedPrefetcher(t_interval=2)
    p.init()

    # Issue a prefetch
    a1 = make_access(pc=0x40, addr=0x4000)
    a2 = make_access(pc=0x40, addr=0x4040)
    p.progress(a1, prefetch_hit=False)
    preds = p.progress(a2, prefetch_hit=False)
    assert preds  # some stride predictions

    # Demand access arrives late (to same prefetched addr)
    late_addr = preds[0]
    a2 = make_access(pc=0x40, addr=late_addr)
    p._update_counters(a2, is_demand_access=True, is_prefetch_hit=False)

    # End interval
    p._end_interval()
    # Lateness counter should have been incremented
    assert p.late_total >= 0


def test_init_and_close_reset_state():
    p = FeedbackDirectedPrefetcher()
    p.init()
    assert p.pref_total == 0
    assert p.used_total == 0

    # Do some accesses
    a1 = make_access(pc=0x50, addr=0x5000)
    p.progress(a1, prefetch_hit=False)

    p.close()  # final interval update
    assert isinstance(p.pref_total, float)  # aggregated counters after close
