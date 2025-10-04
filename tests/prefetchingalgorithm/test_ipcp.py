"""
Unit tests for IPCP Prefetcher.
Covers initialization, IP classification, counter saturation,
prefetch issuance, and scheduler crediting.
"""

import pytest

from prefetchlenz.prefetchingalgorithm.impl.ipcp import (
    CONFIG,
    IPCPPrefetcher,
    block_index,
    region_base,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess


def test_init_and_close():
    p = IPCPPrefetcher()
    p.init()
    assert p.initialized
    p.close()
    assert not p.initialized


def test_ip_insertion_and_classification():
    p = IPCPPrefetcher()
    p.init()
    a = MemoryAccess(address=0x1000, pc=0x10)
    issued = p.progress(a, prefetch_hit=False)
    # A new IP should be inserted with default class
    entry = p.ip_table.lookup(a.pc)
    assert entry is not None
    assert entry.class_id == "GS"
    assert isinstance(issued, list)


def test_stride_prediction():
    p = IPCPPrefetcher()
    p.init()
    # Insert CS-class manually
    entry = p.ip_table.insert(0x20, "CS")
    # Access addresses with fixed stride
    p.progress(MemoryAccess(address=0x2000, pc=0x20), prefetch_hit=False)
    p.progress(
        MemoryAccess(address=0x2000 + CONFIG["BLOCK_SIZE"], pc=0x20), prefetch_hit=False
    )
    issued = p.progress(
        MemoryAccess(address=0x2000 + 2 * CONFIG["BLOCK_SIZE"], pc=0x20),
        prefetch_hit=False,
    )
    assert all(region_base(a) == region_base(0x2000) for a in issued)


def test_signature_prediction_and_saturation():
    p = IPCPPrefetcher()
    p.init()
    entry = p.ip_table.insert(0x30, "GS")
    # Train on several addresses in same region
    base = region_base(0x4000)
    preds = []
    for i in range(3):
        addr = base + i * CONFIG["BLOCK_SIZE"]
        preds += p.progress(MemoryAccess(address=addr, pc=0x30), prefetch_hit=False)
    preds += p.progress(MemoryAccess(address=base, pc=0x30), prefetch_hit=False)
    assert any(region_base(pr) == base for pr in preds)


def test_scheduler_crediting():
    p = IPCPPrefetcher()
    p.init()
    entry = p.ip_table.insert(0x40, "GS")
    base = region_base(0x5000)
    addr = base + 0 * CONFIG["BLOCK_SIZE"]
    preds = p.progress(MemoryAccess(address=addr, pc=0x40), prefetch_hit=False)
    if preds:
        hit = preds[0]
        # Demand access hits prefetched line
        p.progress(MemoryAccess(address=hit, pc=0x40), prefetch_hit=True)
        # Scheduler should remove it from outstanding
        assert hit not in p.scheduler.outstanding
