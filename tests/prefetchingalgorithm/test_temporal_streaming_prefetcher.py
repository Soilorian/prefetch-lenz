# test_temporal_streaming_prefetcher.py
import pytest

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.temporalmemorystreaming import (
    TemporalMemoryStreamingPrefetcher,
)

LINE = 64


def mk_access(addr: int) -> MemoryAccess:
    return MemoryAccess(address=addr, pc=0)


def test_cmob_record_and_replay_basic():
    t = TemporalMemoryStreamingPrefetcher(cmob_capacity=16, prefetch_degree=3)
    t.init()

    # record a sequence of 6 misses: A,B,C,D,E,F
    seq = [0, 64, 128, 192, 256, 320]
    for a in seq:
        t.progress(mk_access(a), prefetch_hit=False)

    # a later access to C should replay successors D,E,F
    preds = t.progress(mk_access(128), prefetch_hit=False)
    assert preds == [192, 256, 320]


def test_no_replay_if_sequence_evicted():
    t = TemporalMemoryStreamingPrefetcher(cmob_capacity=4, prefetch_degree=2)
    t.init()

    # add 6 entries so first two are evicted (capacity 4)
    adds = [0, 64, 128, 192, 256, 320]
    for a in adds:
        t.progress(mk_access(a), prefetch_hit=False)

    # original 0 was at seq 0, now evicted -> no replay
    preds = t.progress(mk_access(0), prefetch_hit=False)
    assert preds == []


def test_outstanding_prevents_duplicate_prefetches():
    t = TemporalMemoryStreamingPrefetcher(
        cmob_capacity=16, prefetch_degree=3, dedup_outstanding=True
    )
    t.init()

    seq = [0, 64, 128, 192, 256]
    for a in seq:
        t.progress(mk_access(a), prefetch_hit=False)

    # trigger replay for 0 -> prefetch 64,128,192
    p1 = t.progress(mk_access(0), prefetch_hit=False)
    assert p1 == [64, 128, 192]

    # replay again immediately -> no prefetch because outstanding tracked
    p2 = t.progress(mk_access(0), prefetch_hit=False)
    assert p2 == []

    # simulate completion of prefetch 64
    t.prefetch_completed(64)
    # now a replay can re-issue prefetch for 64 (if others still outstanding they may be skipped)
    p3 = t.progress(mk_access(0), prefetch_hit=False)
    assert 64 in p3 or p3 == []


def test_directory_updates_to_most_recent_occurrence():
    t = TemporalMemoryStreamingPrefetcher(cmob_capacity=10, prefetch_degree=2)
    t.init()

    # A appears twice in CMOB; directory should point to most recent so replay picks successors after the last occurrence
    t.progress(mk_access(0), prefetch_hit=False)  # seq 0
    t.progress(mk_access(64), prefetch_hit=False)  # seq 1
    t.progress(mk_access(0), prefetch_hit=False)  # seq 2 (most recent 0)
    t.progress(mk_access(128), prefetch_hit=False)  # seq 3
    # Now access '0' should replay successors following seq 2 -> should return 128
    preds = t.progress(mk_access(0), prefetch_hit=False)
    assert 128 in preds
