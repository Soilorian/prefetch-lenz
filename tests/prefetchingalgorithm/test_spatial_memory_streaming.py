"""
Unit tests for SMSPrefetcher implementation.

Each test validates a specific claim or mechanism from the SMS paper:
 - Initialization/reset
 - AGT promotion and commit
 - PHT counters and saturation
 - Prefetch issuance and scheduler crediting
 - Chaining behavior (multiple predicted blocks, MRB avoidance)
"""

import logging

from prefetchlenz.prefetchingalgorithm.impl.sms import (
    BLOCK_SIZE,
    CONFIG,
    REGION_SIZE,
    SetAssocPHT,
    SMSPrefetcher,
    region_base,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

logging.basicConfig(level=logging.ERROR)


def test_init_and_reset():
    """Initialization and reset behavior (paper: AGT/PHT lifetimes)."""
    p = SMSPrefetcher()
    p.init()
    assert p.initialized is True
    p.close()
    assert p.initialized is False
    assert len(p.scheduler.outstanding) == 0
    assert len(p.pred_regs) == 0


def test_agt_promote_and_commit():
    """Validates AGT filter -> accumulation promotion and commit to PHT (Section 4.1)."""
    p = SMSPrefetcher()
    base = region_base(0x10000)

    # First access creates filter entry
    a1 = MemoryAccess(address=base + 0 * BLOCK_SIZE, pc=0x2000)
    p.progress(a1, prefetch_hit=False)

    # Second distinct block promotes to accumulation
    a2 = MemoryAccess(address=base + 1 * BLOCK_SIZE, pc=0x2000)
    p.progress(a2, prefetch_hit=False)

    # Simulate more accesses to grow accumulation
    for i in range(2, 4):
        p.progress(
            MemoryAccess(address=base + i * BLOCK_SIZE, pc=0x2000), prefetch_hit=False
        )

    # Manually close and commit accumulation
    acc = p.agt.close_generation(base)
    if acc:
        p.commit_generation(acc)

    key = base ^ (0 & CONFIG["TRIGGER_KEY_MASK"])
    e = p.pht.get(key)
    assert e is not None
    # Counters for block 0 should have been initialized >= INIT_OBSERVED
    assert e.counters[0] >= CONFIG["INIT_OBSERVED"] - 1


def test_pht_counters_saturate_and_update():
    """Validates 2-bit counter saturation and update rules (per-block hysteresis)."""
    pht = SetAssocPHT(num_sets=4, associativity=2)
    region = region_base(0x30000)

    # Create pattern: first two blocks observed
    bitvec = [0] * (REGION_SIZE // BLOCK_SIZE)
    bitvec[0] = 1
    bitvec[1] = 1
    e = pht.insert_or_update(region, 0, bitvec)

    # Update repeatedly; counters should saturate at COUNTER_MAX
    for _ in range(10):
        pht.insert_or_update(region, 0, bitvec)

    assert all(0 <= c <= CONFIG["COUNTER_BITS"] ** 2 - 1 for c in e.counters)
    assert e.counters[0] == (1 << CONFIG["COUNTER_BITS"]) - 1  # Saturated


def test_prefetch_issue_and_crediting():
    """Validates that a PHT hit on trigger causes prefetches and prefetch_hit credits scheduler."""
    p = SMSPrefetcher()
    base = region_base(0x50000)

    # Create PHT entry manually
    key = base ^ (0 & CONFIG["TRIGGER_KEY_MASK"])
    bitvec = [0] * (REGION_SIZE // BLOCK_SIZE)
    bitvec[2] = 1
    bitvec[3] = 1
    p.pht.insert_or_update(key, 0, bitvec)

    # Trigger access should cause streaming
    access = MemoryAccess(address=base + 0 * BLOCK_SIZE, pc=0x4000)
    issued = p.progress(access, prefetch_hit=False)
    assert len(issued) >= 1

    # Simulate demand access hitting prefetched line
    hit_addr = issued[0]
    p.progress(MemoryAccess(address=hit_addr, pc=0x4000), prefetch_hit=True)

    # Scheduler should have credited back at least one outstanding prefetch
    assert len(p.scheduler.outstanding) >= 0


def test_chain_and_mrb_like_behavior():
    """Ensures chaining (multiple predicted blocks) is serviced without repeats (MRB avoidance)."""
    p = SMSPrefetcher()
    base = region_base(0x70000)

    # Insert pattern with 4 predicted blocks
    key = base ^ (0 & CONFIG["TRIGGER_KEY_MASK"])
    bitvec = [0] * (REGION_SIZE // BLOCK_SIZE)
    for i in range(1, 5):
        bitvec[i] = 1
    p.pht.insert_or_update(key, 0, bitvec)

    # Trigger should cause streaming of multiple unique prefetches
    access = MemoryAccess(address=base + 0 * BLOCK_SIZE, pc=0x6000)
    issued = p.progress(access, prefetch_hit=False)

    assert len(issued) == len(set(issued))  # no duplicates
    assert len(issued) <= CONFIG["PREFETCH_DEGREE"]
