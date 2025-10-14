"""
Unit tests for BingoPrefetcher implementation.

Tests validate:
- Initialization and cleanup
- Filter table operations (insert, lookup, LRU eviction)
- Accumulation table promotion
- Pattern History Table lookup and insertion
- Pattern rotation logic
- Prefetch generation from learned patterns
- Block eviction triggering pattern commit
- End-to-end spatial pattern learning
"""

import logging

import pytest

from prefetchlenz.prefetchingalgorithm.impl.bingo import (
    BLOCK_SIZE,
    PATTERN_LEN,
    REGION_SIZE,
    AccumulationTable,
    AccumulationTableData,
    BingoPrefetcher,
    FilterTable,
    FilterTableData,
    PatternHistoryTable,
    region_number,
    region_offset,
    rotate_pattern,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

# Set logging level for tests
logging.basicConfig(level=logging.WARNING)


def test_init_and_close():
    """Test initialization and cleanup."""
    p = BingoPrefetcher()
    assert not p.initialized

    p.init()
    assert p.initialized

    p.close()
    assert not p.initialized


def test_region_functions():
    """Test region number and offset calculation."""
    # Block 0 should be in region 0, offset 0
    assert region_number(0) == 0
    assert region_offset(0) == 0

    # Block 31 should be in region 0, offset 31
    assert region_number(31) == 0
    assert region_offset(31) == 31

    # Block 32 should be in region 1, offset 0
    assert region_number(32) == 1
    assert region_offset(32) == 0

    # Block 100 should be in region 3, offset 4
    assert region_number(100) == 3
    assert region_offset(100) == 4


def test_rotate_pattern():
    """Test pattern rotation."""
    pattern = [True, False, True, False]

    # Rotate left by 1
    rotated = rotate_pattern(pattern, 1)
    assert rotated == [False, True, False, True]

    # Rotate right by 1 (equivalent to left by 3)
    rotated = rotate_pattern(pattern, -1)
    assert rotated == [False, True, False, True]

    # Rotate by full length returns original
    rotated = rotate_pattern(pattern, 4)
    assert rotated == pattern

    # Empty pattern
    assert rotate_pattern([], 5) == []


def test_filter_table_operations():
    """Test filter table insert, find, and erase."""
    ft = FilterTable(size=4)

    # Insert entries
    ft.insert(region_num=0, pc=0x1000, offset=5)
    ft.insert(region_num=1, pc=0x2000, offset=10)

    # Find entries
    entry = ft.find(0)
    assert entry is not None
    assert entry.pc == 0x1000
    assert entry.offset == 5

    entry = ft.find(1)
    assert entry is not None
    assert entry.pc == 0x2000
    assert entry.offset == 10

    # Non-existent entry
    assert ft.find(999) is None

    # Erase entry
    erased = ft.erase(0)
    assert erased is not None
    assert erased.pc == 0x1000

    # Entry should be gone
    assert ft.find(0) is None


def test_filter_table_lru_eviction():
    """Test filter table LRU eviction."""
    ft = FilterTable(size=2)

    # Fill table
    ft.insert(0, 0x1000, 0)
    ft.insert(1, 0x2000, 1)

    # Access first entry to make it MRU
    ft.find(0)

    # Insert third entry - should evict entry 1 (LRU)
    ft.insert(2, 0x3000, 2)

    # Entry 1 should be evicted
    assert ft.find(1) is None

    # Entries 0 and 2 should exist
    assert ft.find(0) is not None
    assert ft.find(2) is not None


def test_accumulation_table_operations():
    """Test accumulation table operations."""
    at = AccumulationTable(size=4, pattern_len=32)

    # Create filter data
    filter_data = FilterTableData(pc=0x1000, offset=5)

    # Insert from filter
    evicted = at.insert(filter_data, region_num=0)
    assert evicted is None

    # Find entry
    entry = at.find(0)
    assert entry is not None
    assert entry.pc == 0x1000
    assert entry.offset == 5
    assert entry.pattern[5] is True
    assert sum(entry.pattern) == 1  # Only one bit set

    # Set additional pattern bits
    success = at.set_pattern(0, 10)
    assert success is True

    entry = at.find(0)
    assert entry.pattern[10] is True
    assert sum(entry.pattern) == 2

    # Try to set pattern for non-existent entry
    success = at.set_pattern(999, 0)
    assert success is False


def test_accumulation_table_eviction():
    """Test accumulation table eviction behavior."""
    at = AccumulationTable(size=2, pattern_len=32)

    # Fill table
    filter_data1 = FilterTableData(pc=0x1000, offset=0)
    filter_data2 = FilterTableData(pc=0x2000, offset=1)

    at.insert(filter_data1, region_num=0)
    at.insert(filter_data2, region_num=1)

    # Access first entry
    at.find(0)

    # Insert third entry - should evict region 1 (LRU)
    filter_data3 = FilterTableData(pc=0x3000, offset=2)
    evicted = at.insert(filter_data3, region_num=2)

    assert evicted is not None
    assert evicted.pc == 0x2000

    # Verify entries
    assert at.find(0) is not None
    assert at.find(1) is None  # Evicted
    assert at.find(2) is not None


def test_pattern_history_table_operations():
    """Test PHT insert and find."""
    pht = PatternHistoryTable(
        size=256,
        pattern_len=32,
        addr_width=16,
        pc_width=16,
        num_ways=4
    )

    # Create pattern
    pattern = [False] * 32
    pattern[0] = True
    pattern[5] = True
    pattern[10] = True

    # Insert pattern
    pht.insert(pc=0x1000, address=0, pattern=pattern)

    # Find pattern with same offset
    found = pht.find(pc=0x1000, address=0)
    assert found is not None
    assert found[0] is True
    assert found[5] is True
    assert found[10] is True

    # Find with different offset should rotate
    found = pht.find(pc=0x1000, address=1)
    assert found is not None
    # Pattern should be rotated by 1
    assert found[1] is True  # Was at 0
    assert found[6] is True  # Was at 5
    assert found[11] is True  # Was at 10


def test_pattern_history_table_different_indexing():
    """Test PHT with different PC/Address widths."""
    # PC+Address table
    pht_pc_addr = PatternHistoryTable(256, 32, addr_width=16, pc_width=16)

    # PC only table
    pht_pc = PatternHistoryTable(256, 32, addr_width=0, pc_width=16)

    # Address only table
    pht_addr = PatternHistoryTable(256, 32, addr_width=16, pc_width=0)

    pattern = [True] + [False] * 31

    # Insert same pattern
    pht_pc_addr.insert(0x1000, 0x5000, pattern)
    pht_pc.insert(0x1000, 0x5000, pattern)
    pht_addr.insert(0x1000, 0x5000, pattern)

    # PC+Address should only match exact PC and address
    assert pht_pc_addr.find(0x1000, 0x5000) is not None
    assert pht_pc_addr.find(0x1000, 0x6000) is None  # Different address
    assert pht_pc_addr.find(0x2000, 0x5000) is None  # Different PC

    # PC only should match same PC, any address
    assert pht_pc.find(0x1000, 0x5000) is not None
    assert pht_pc.find(0x1000, 0x6000) is not None  # Same PC
    assert pht_pc.find(0x2000, 0x5000) is None  # Different PC

    # Address only should match same address, any PC
    assert pht_addr.find(0x1000, 0x5000) is not None
    assert pht_addr.find(0x2000, 0x5000) is not None  # Same address
    assert pht_addr.find(0x1000, 0x6000) is None  # Different address


def test_bingo_filter_to_accumulation_promotion():
    """Test promotion from filter table to accumulation table."""
    p = BingoPrefetcher()
    p.init()

    region_base = 0
    block0 = region_base * PATTERN_LEN
    block1 = region_base * PATTERN_LEN + 1

    # First access creates filter entry
    addr0 = block0 * BLOCK_SIZE
    access0 = MemoryAccess(address=addr0, pc=0x1000)
    prefetches = p.progress(access0, prefetch_hit=False)

    # Should create filter entry
    filter_entry = p.filter_table.find(region_base)
    assert filter_entry is not None
    assert filter_entry.offset == 0

    # Second access to different offset promotes to accumulation
    addr1 = block1 * BLOCK_SIZE
    access1 = MemoryAccess(address=addr1, pc=0x1000)
    prefetches = p.progress(access1, prefetch_hit=False)

    # Filter entry should be gone
    assert p.filter_table.find(region_base) is None

    # Accumulation entry should exist
    accum_entry = p.accumulation_table.find(region_base)
    assert accum_entry is not None
    assert accum_entry.pattern[0] is True  # First access
    assert accum_entry.pattern[1] is True  # Second access


def test_bingo_pattern_learning_and_prefetch():
    """Test end-to-end pattern learning and prefetch generation."""
    p = BingoPrefetcher()
    p.init()

    # Define region 0
    region0 = 0
    blocks = [0, 5, 10, 15]  # Spatial pattern

    # Train pattern by accessing blocks in region 0
    for block_offset in blocks:
        block = region0 * PATTERN_LEN + block_offset
        addr = block * BLOCK_SIZE
        access = MemoryAccess(address=addr, pc=0x1000)
        p.progress(access, prefetch_hit=False)

    # Manually commit the pattern to PHT
    accum_entry = p.accumulation_table.find(region0)
    if accum_entry:
        p._commit_pattern(accum_entry, region0)

    # Now access region 1 with same PC - should trigger prefetch
    region1 = 1
    trigger_block = region1 * PATTERN_LEN
    trigger_addr = trigger_block * BLOCK_SIZE
    access = MemoryAccess(address=trigger_addr, pc=0x1000)

    prefetches = p.progress(access, prefetch_hit=False)

    # Should generate prefetches based on learned pattern
    assert len(prefetches) > 0

    # Convert prefetches back to block numbers
    pf_blocks = [pf // BLOCK_SIZE for pf in prefetches]
    pf_offsets = [region_offset(blk) for blk in pf_blocks]

    # Should include some of the learned pattern offsets
    # (may not be all due to PREFETCH_DEGREE limit)
    assert any(off in blocks for off in pf_offsets)


def test_bingo_eviction_commits_pattern():
    """Test that block eviction commits pattern to PHT."""
    p = BingoPrefetcher()
    p.init()

    region0 = 0
    blocks = [0, 3, 7]

    # Build pattern in accumulation table
    for block_offset in blocks:
        block = region0 * PATTERN_LEN + block_offset
        addr = block * BLOCK_SIZE
        access = MemoryAccess(address=addr, pc=0x2000)
        p.progress(access, prefetch_hit=False)

    # Verify accumulation entry exists
    accum_entry = p.accumulation_table.find(region0)
    assert accum_entry is not None

    # Trigger eviction
    evict_block = region0 * PATTERN_LEN + 5
    p.eviction(evict_block)

    # Accumulation entry should be gone
    assert p.accumulation_table.find(region0) is None

    # Pattern should be in PHT - test by accessing new region with same PC
    region1 = 1
    trigger_block = region1 * PATTERN_LEN
    trigger_addr = trigger_block * BLOCK_SIZE
    access = MemoryAccess(address=trigger_addr, pc=0x2000)

    prefetches = p.progress(access, prefetch_hit=False)

    # Should generate prefetches from committed pattern
    assert len(prefetches) > 0


def test_bingo_multi_region_tracking():
    """Test tracking multiple regions simultaneously."""
    p = BingoPrefetcher()
    p.init()

    # Create patterns in multiple regions
    regions = [0, 1, 2]
    patterns = {
        0: [0, 1, 2],
        1: [5, 10, 15],
        2: [8, 16, 24]
    }

    for region in regions:
        for block_offset in patterns[region]:
            block = region * PATTERN_LEN + block_offset
            addr = block * BLOCK_SIZE
            pc = 0x1000 + region * 0x100  # Different PC per region
            access = MemoryAccess(address=addr, pc=pc)
            p.progress(access, prefetch_hit=False)

    # Verify all regions have accumulation entries
    for region in regions:
        accum = p.accumulation_table.find(region)
        assert accum is not None
        # Check pattern bits
        for offset in patterns[region]:
            assert accum.pattern[offset] is True


def test_bingo_close_commits_all_patterns():
    """Test that close() commits all remaining patterns."""
    p = BingoPrefetcher()
    p.init()

    # Create pattern in region 0
    region0 = 0
    blocks = [0, 4, 8, 12]

    for block_offset in blocks:
        block = region0 * PATTERN_LEN + block_offset
        addr = block * BLOCK_SIZE
        access = MemoryAccess(address=addr, pc=0x3000)
        p.progress(access, prefetch_hit=False)

    # Verify accumulation entry exists before close
    assert p.accumulation_table.find(region0) is not None

    # Close should commit pattern
    p.close()

    # Accumulation table should be cleared
    assert len(p.accumulation_table.cache.entries) == 0
    assert not p.initialized


def test_bingo_prefetch_degree_limit():
    """Test that prefetch degree is respected."""
    p = BingoPrefetcher()
    p.init()

    # Create pattern with many blocks
    region0 = 0
    blocks = list(range(1, 20))  # 19 blocks

    for block_offset in blocks:
        block = region0 * PATTERN_LEN + block_offset
        addr = block * BLOCK_SIZE
        access = MemoryAccess(address=addr, pc=0x4000)
        p.progress(access, prefetch_hit=False)

    # Commit pattern
    accum = p.accumulation_table.find(region0)
    if accum:
        p._commit_pattern(accum, region0)

    # Trigger prefetch
    region1 = 1
    trigger_block = region1 * PATTERN_LEN
    trigger_addr = trigger_block * BLOCK_SIZE
    access = MemoryAccess(address=trigger_addr, pc=0x4000)

    prefetches = p.progress(access, prefetch_hit=False)

    # Should be limited by PREFETCH_DEGREE
    from prefetchlenz.prefetchingalgorithm.impl.bingo import CONFIG
    assert len(prefetches) <= CONFIG["PREFETCH_DEGREE"]


def test_bingo_no_prefetch_without_pattern():
    """Test that no prefetches are generated without learned pattern."""
    p = BingoPrefetcher()
    p.init()

    # Access region without any prior training
    region = 10
    block = region * PATTERN_LEN
    addr = block * BLOCK_SIZE
    access = MemoryAccess(address=addr, pc=0x9999)

    prefetches = p.progress(access, prefetch_hit=False)

    # Should not generate prefetches
    assert len(prefetches) == 0


def test_bingo_same_offset_updates_accumulation():
    """Test that accessing same offset in region updates accumulation."""
    p = BingoPrefetcher()
    p.init()

    region0 = 0
    block0 = region0 * PATTERN_LEN + 5

    # First access
    addr = block0 * BLOCK_SIZE
    access = MemoryAccess(address=addr, pc=0x5000)
    p.progress(access, prefetch_hit=False)

    # Creates filter entry
    assert p.filter_table.find(region0) is not None

    # Access different offset to promote to accumulation
    block1 = region0 * PATTERN_LEN + 10
    addr1 = block1 * BLOCK_SIZE
    access1 = MemoryAccess(address=addr1, pc=0x5000)
    p.progress(access1, prefetch_hit=False)

    # Now in accumulation table
    assert p.accumulation_table.find(region0) is not None

    # Access another offset
    block2 = region0 * PATTERN_LEN + 15
    addr2 = block2 * BLOCK_SIZE
    access2 = MemoryAccess(address=addr2, pc=0x5000)
    p.progress(access2, prefetch_hit=False)

    # Check pattern has all three bits set
    accum = p.accumulation_table.find(region0)
    assert accum.pattern[5] is True
    assert accum.pattern[10] is True
    assert accum.pattern[15] is True
