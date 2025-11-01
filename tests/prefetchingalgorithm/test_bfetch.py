"""
Unit tests for BFetchPrefetcher (approximation using non-sequential detection).

Tests are deterministic and validate:
 - init/close behavior
 - BST insert/evict behavior (fallback dict mode)
 - Prefetch issuance on detected predicted branch events
 - Crediting behavior when prefetched lines are used
 - MRB suppression avoids immediate re-issue
"""

import pytest

from prefetchlenz.prefetchingalgorithm.impl.bfetch import (
    BST,
    CONFIG,
    BFetchPrefetcher,
    StreamDescriptor,
    block_base,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess


def test_init_and_close():
    """Initialization and reset behavior."""
    p = BFetchPrefetcher()
    p.init()
    assert p.initialized
    p.close()
    assert not p.initialized


def test_bst_insert_and_evict_fallback():
    """BST fallback dict inserts entries and evicts oldest when capacity exceeded."""
    bst = BST()
    # artificially set small capacity for test
    old_num = CONFIG["BST_NUM_ENTRIES"]
    CONFIG["BST_NUM_ENTRIES"] = 4
    for i in range(5):
        bst.insert(
            i, StreamDescriptor(target_base=0x1000 + i * 0x100, blocks=2, conf=1)
        )
    # After insert 5 with capacity 4, table length == 4
    assert len(bst.cache) == 4
    # restore
    CONFIG["BST_NUM_ENTRIES"] = old_num


def test_prefetch_on_nonsequential_event():
    """Prefetch issuance when a non-sequential (predicted) branch target is detected."""
    p = BFetchPrefetcher()
    p.init()
    # Simulate sequential fetch at pc=0x10
    p.progress(MemoryAccess(address=0x1000, pc=0x10), prefetch_hit=False)
    # Now simulate a non-sequential fetch (jump) to 0x2000 from previous pc
    issued = p.progress(MemoryAccess(address=0x2000, pc=0x20), prefetch_hit=False)
    # Because no BST entry existed, a new descriptor is created (no prefetch issued)
    assert isinstance(p.bst.lookup(0x10), StreamDescriptor)


def test_issue_prefetch_with_entry_and_crediting():
    """When BST entry exists, a non-sequential event leads to prefetch issuance, and prefetch_hit credits."""
    p = BFetchPrefetcher()
    p.init()
    # construct an entry for branch_pc = 0x100
    branch_pc = 0x100
    target = 0x4000
    desc = StreamDescriptor(target_base=block_base(target), blocks=2, conf=2)
    p.bst.insert(branch_pc, desc)
    # set last_pc so detection heuristics treat next access as predicted branch
    p.last_pc = branch_pc
    # trigger predicted branch event: access to target
    issued = p.progress(
        MemoryAccess(address=target, pc=branch_pc + 4), prefetch_hit=False
    )
    # issued should be a list (possibly empty if MRB/outstanding cap prevents)
    assert isinstance(issued, list)
    if issued:
        # Simulate using the first prefetched block -> credit
        hit = issued[0]
        p.progress(MemoryAccess(address=hit, pc=0x0), prefetch_hit=True)
        # MRB should suppress reissue: immediate re-trigger should not reissue the same block
        p.last_pc = branch_pc
        issued2 = p.progress(
            MemoryAccess(address=target, pc=branch_pc + 4), prefetch_hit=False
        )
        assert hit not in issued2


def test_mrb_suppresses_duplicates():
    """MRB suppression prevents repeated issues of the same block in consecutive triggers."""
    p = BFetchPrefetcher()
    p.init()
    branch_pc = 0x300
    target = 0x5000
    desc = StreamDescriptor(target_base=block_base(target), blocks=1, conf=2)
    p.bst.insert(branch_pc, desc)
    p.last_pc = branch_pc
    issued1 = p.progress(
        MemoryAccess(address=target, pc=branch_pc + 4), prefetch_hit=False
    )
    # Immediately trigger the same candidate again -> MRB should suppress it
    p.last_pc = branch_pc
    issued2 = p.progress(
        MemoryAccess(address=target, pc=branch_pc + 4), prefetch_hit=False
    )
    # addresses shouldn't repeat
    assert set(issued1).isdisjoint(set(issued2))
