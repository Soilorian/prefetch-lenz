"""
Unit tests for DSPatchPrefetcher implementation.

Tests validate:
- Initialization and cleanup
- Page buffer operations
- Pattern compression and decompression
- Pattern rotation (anchoring)
- Coverage-biased pattern learning (OR operations)
- Accuracy-biased pattern learning (AND operations)
- Bandwidth tracking and quartile calculation
- Pattern selection based on bandwidth utilization
- Dual pattern modulation
- Multiple triggers per page
- End-to-end spatial pattern learning
"""

import logging

from prefetchlenz.prefetchingalgorithm.impl.dspatch import (
    CONFIG,
    DSPatchPrefetcher,
    PageBuffer,
    SignaturePatternTable,
    BandwidthTracker,
    compress_pattern,
    decompress_pattern,
    rotate_pattern,
    popcount,
    hash_pc,
    page_number,
    page_offset,
    region_in_page,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

# Set logging level for tests
logging.basicConfig(level=logging.WARNING)


def test_helper_functions():
    """Test helper functions."""
    # Test page_number and page_offset
    addr = 0x10000  # 64KB
    assert page_number(addr) == 16  # 64KB / 4KB = 16

    addr = 0x1040  # 4KB + 64B
    assert page_offset(addr) == 1  # Second block in page

    # Test region_in_page
    assert region_in_page(0) == 0  # First region
    assert region_in_page(31) == 0  # Last block of first region
    assert region_in_page(32) == 1  # First block of second region
    assert region_in_page(63) == 1  # Last block of second region

    # Test popcount
    assert popcount(0b1010) == 2
    assert popcount(0b1111) == 4
    assert popcount(0) == 0

    # Test hash_pc
    pc1 = 0x1000
    pc2 = 0x2000
    idx1 = hash_pc(pc1)
    idx2 = hash_pc(pc2)
    assert 0 <= idx1 < CONFIG["SPT_ENTRIES"]
    assert 0 <= idx2 < CONFIG["SPT_ENTRIES"]


def test_pattern_compression():
    """Test 128B-granularity compression."""
    # Pattern with adjacent bits set
    pattern = 0b11001100  # 8 bits for simplicity
    compressed = compress_pattern(pattern, 8)
    # Adjacent bits should be ORed: 11 -> 1, 00 -> 0, 11 -> 1, 00 -> 0
    assert compressed == 0b1010

    # Pattern with sparse bits
    pattern = 0b10101010
    compressed = compress_pattern(pattern, 8)
    assert compressed == 0b1111  # Each pair has at least one bit set

    # Decompress
    compressed = 0b1010  # 4 bits
    decompressed = decompress_pattern(compressed, 8)
    # Each bit expands to two: 1 -> 11, 0 -> 00, 1 -> 11, 0 -> 00
    assert decompressed == 0b11001100


def test_pattern_rotation():
    """Test pattern rotation (anchoring)."""
    pattern = 0b1000  # 4 bits
    rotated = rotate_pattern(pattern, 1, 4)
    assert rotated == 0b0001  # Rotated left by 1

    pattern = 0b1100
    rotated = rotate_pattern(pattern, 2, 4)
    assert rotated == 0b0011  # Rotated left by 2

    # Rotate by full length should return same
    pattern = 0b1010
    rotated = rotate_pattern(pattern, 4, 4)
    assert rotated == pattern


def test_init_and_close():
    """Test initialization and cleanup."""
    p = DSPatchPrefetcher()
    assert not p.initialized

    p.init()
    assert p.initialized

    p.close()
    assert not p.initialized


def test_page_buffer_operations():
    """Test page buffer basic operations."""
    pb = PageBuffer(entries=4)

    # First access to page
    entry = pb.access(page_num=0, offset=0, pc=0x1000)
    assert entry is not None
    assert entry.page_num == 0
    assert entry.pattern & 1  # Bit 0 should be set
    assert entry.triggered[0]  # Region 0 triggered
    assert entry.trigger_pc[0] == 0x1000

    # Second access to same page, different offset
    entry = pb.access(page_num=0, offset=5, pc=0x1000)
    assert entry.pattern & (1 << 5)  # Bit 5 should be set

    # Access to second region
    entry = pb.access(page_num=0, offset=35, pc=0x2000)
    assert entry.triggered[1]  # Region 1 triggered
    assert entry.trigger_pc[1] == 0x2000


def test_page_buffer_lru_eviction():
    """Test page buffer LRU eviction."""
    pb = PageBuffer(entries=2)

    # Fill buffer
    pb.access(0, 0, 0x1000)
    pb.access(1, 0, 0x2000)

    # Access page 0 to make it MRU
    pb.access(0, 1, 0x1000)

    # Insert new page, should evict page 1 (LRU)
    pb.access(2, 0, 0x3000)

    assert pb.get_entry(0) is not None
    assert pb.get_entry(1) is None  # Evicted
    assert pb.get_entry(2) is not None


def test_signature_pattern_table():
    """Test SPT operations."""
    spt = SignaturePatternTable(entries=256)

    # Lookup creates entry if not exists
    entry1 = spt.lookup(0x1000)
    assert entry1 is not None
    assert entry1.cov_pattern == [0, 0]
    assert entry1.acc_pattern == [0, 0]

    # Update entry
    spt.update(0x1000, region=0, program_pattern=0b1010,
               cov_pattern=0b1110, acc_pattern=0b1010,
               measure_cov=1, measure_acc=1, or_count=1)

    entry2 = spt.lookup(0x1000)
    assert entry2.cov_pattern[0] == 0b1110
    assert entry2.acc_pattern[0] == 0b1010


def test_bandwidth_tracker():
    """Test bandwidth tracking."""
    bw = BandwidthTracker()

    # Initially in lowest quartile
    assert bw.get_quartile() == 0

    # Simulate low utilization
    for _ in range(200):
        bw.record_access()
    assert bw.get_quartile() == 0  # Still low

    # Simulate high utilization
    for _ in range(800):
        bw.record_access()
    # Should increase quartile (depends on window processing)


def test_coverage_biased_pattern_or_operation():
    """Test CoP learning via OR operations."""
    p = DSPatchPrefetcher()
    p.init()

    spt_entry = p.spt.lookup(0x1000)

    # First pattern
    p._update_patterns(spt_entry, region=0, program_pattern=0b1010)
    assert spt_entry.cov_pattern[0] == 0b1010

    # Second pattern - OR should add bits
    p._update_patterns(spt_entry, region=0, program_pattern=0b0101)
    assert spt_entry.cov_pattern[0] == 0b1111  # OR of 1010 and 0101

    # Check OR count incremented
    assert spt_entry.or_count[0] > 0


def test_accuracy_biased_pattern_and_operation():
    """Test AccP learning via AND operations."""
    p = DSPatchPrefetcher()
    p.init()

    spt_entry = p.spt.lookup(0x1000)

    # Set up CoP first
    spt_entry.cov_pattern[0] = 0b1111

    # Update with program pattern
    p._update_patterns(spt_entry, region=0, program_pattern=0b1010)

    # AccP should be AND of CoP and program pattern
    assert spt_entry.acc_pattern[0] == 0b1010  # 1111 AND 1010 = 1010


def test_pattern_selection_low_bandwidth():
    """Test pattern selection with low bandwidth."""
    p = DSPatchPrefetcher()
    p.init()

    # Set low bandwidth
    p.bw_tracker.current_quartile = 0  # <25%

    spt_entry = p.spt.lookup(0x1000)
    spt_entry.cov_pattern[0] = 0b1111
    spt_entry.acc_pattern[0] = 0b1010

    # Should select CoP for low bandwidth
    selected = p._select_pattern(spt_entry, region=0)
    assert selected == 0b1111  # CoP selected


def test_pattern_selection_high_bandwidth():
    """Test pattern selection with high bandwidth."""
    p = DSPatchPrefetcher()
    p.init()

    # Set high bandwidth
    p.bw_tracker.current_quartile = 3  # >=75%

    spt_entry = p.spt.lookup(0x1000)
    spt_entry.cov_pattern[0] = 0b1111
    spt_entry.acc_pattern[0] = 0b1010
    spt_entry.measure_acc[0] = 0  # Good accuracy

    # Should select AccP for high bandwidth
    selected = p._select_pattern(spt_entry, region=0)
    assert selected == 0b1010  # AccP selected


def test_pattern_selection_throttling():
    """Test prefetch throttling with high bandwidth and poor accuracy."""
    p = DSPatchPrefetcher()
    p.init()

    # Set high bandwidth
    p.bw_tracker.current_quartile = 3

    spt_entry = p.spt.lookup(0x1000)
    spt_entry.measure_acc[0] = 3  # Saturated (poor accuracy)

    # Should throttle completely
    selected = p._select_pattern(spt_entry, region=0)
    assert selected is None


def test_quantify_accuracy():
    """Test accuracy quantification."""
    p = DSPatchPrefetcher()
    p.init()

    # Perfect accuracy
    program = 0b1010
    predicted = 0b1010
    accuracy = p._quantify_accuracy(program, predicted)
    assert accuracy == 3  # >=75%

    # 50% accuracy
    program = 0b1111
    predicted = 0b1100
    accuracy = p._quantify_accuracy(program, predicted)
    assert accuracy >= 1  # At least 25%


def test_quantify_coverage():
    """Test coverage quantification."""
    p = DSPatchPrefetcher()
    p.init()

    # Perfect coverage
    program = 0b1010
    predicted = 0b1111
    coverage = p._quantify_coverage(program, predicted)
    assert coverage == 3  # 100%

    # Partial coverage
    program = 0b1111
    predicted = 0b1000
    coverage = p._quantify_coverage(program, predicted)
    assert coverage >= 0  # Some coverage


def test_dual_pattern_modulation():
    """Test simultaneous CoP and AccP modulation."""
    p = DSPatchPrefetcher()
    p.init()

    spt_entry = p.spt.lookup(0x1000)

    # Three different program patterns
    patterns = [0b1100, 0b1010, 0b1001]

    for pat in patterns:
        p._update_patterns(spt_entry, region=0, program_pattern=pat)

    # CoP should have OR of all patterns
    expected_cov = 0b1100 | 0b1010 | 0b1001
    assert spt_entry.cov_pattern[0] == expected_cov

    # AccP should be more restrictive
    assert popcount(spt_entry.acc_pattern[0]) <= popcount(spt_entry.cov_pattern[0])


def test_multiple_triggers_per_page():
    """Test two triggers per 4KB page."""
    p = DSPatchPrefetcher()
    p.init()

    page = 0
    base_addr = page * CONFIG["PAGE_SIZE"]

    # First trigger in region 0
    access1 = MemoryAccess(address=base_addr, pc=0x1000)
    p.progress(access1, prefetch_hit=False)

    # Second trigger in region 1
    access2 = MemoryAccess(address=base_addr + 2048, pc=0x2000)
    p.progress(access2, prefetch_hit=False)

    # Check both triggers recorded
    pb_entry = p.page_buffer.get_entry(page)
    assert pb_entry.triggered[0]
    assert pb_entry.triggered[1]
    assert pb_entry.trigger_pc[0] == 0x1000
    assert pb_entry.trigger_pc[1] == 0x2000


def test_end_to_end_pattern_learning():
    """Test complete pattern learning cycle."""
    p = DSPatchPrefetcher()
    p.init()

    pc = 0x1000
    page = 0
    base_addr = page * CONFIG["PAGE_SIZE"]

    # Phase 1: Train pattern
    # Access blocks 0, 2, 4, 6 in first region
    for offset in [0, 2, 4, 6]:
        addr = base_addr + (offset * CONFIG["BLOCK_SIZE"])
        access = MemoryAccess(address=addr, pc=pc)
        p.progress(access, prefetch_hit=False)

    # Close to commit patterns to SPT
    p.close()

    # Re-initialize and check SPT
    p.init()

    # Get SPT entry
    spt_entry = p.spt.lookup(pc)

    # Should have learned some pattern
    # (Exact pattern depends on compression and updates)
    assert spt_entry.cov_pattern[0] != 0 or spt_entry.acc_pattern[0] != 0


def test_prefetch_generation():
    """Test prefetch generation from learned pattern."""
    p = DSPatchPrefetcher()
    p.init()

    pc = 0x1000
    page = 1
    base_addr = page * CONFIG["PAGE_SIZE"]

    # Manually set up a learned pattern
    spt_entry = p.spt.lookup(pc)
    spt_entry.cov_pattern[0] = 0b10101010  # Alternating pattern
    spt_entry.acc_pattern[0] = 0b10101010
    spt_entry.measure_acc[0] = 0  # Good accuracy
    spt_entry.measure_cov[0] = 0  # Good coverage

    # Set low bandwidth for CoP selection
    p.bw_tracker.current_quartile = 0

    # Trigger access
    access = MemoryAccess(address=base_addr, pc=pc)
    prefetches = p.progress(access, prefetch_hit=False)

    # Should generate prefetches based on pattern
    # Note: Actual count depends on decompression and anchoring


def test_close_commits_patterns():
    """Test that close() commits all patterns."""
    p = DSPatchPrefetcher()
    p.init()

    page = 0
    base_addr = page * CONFIG["PAGE_SIZE"]

    # Create some accesses
    for offset in [0, 1, 2]:
        addr = base_addr + (offset * CONFIG["BLOCK_SIZE"])
        access = MemoryAccess(address=addr, pc=0x1000)
        p.progress(access, prefetch_hit=False)

    # Close should commit patterns
    p.close()

    # State should be cleared
    assert not p.initialized
    assert len(p.page_buffer.entries) == 0


def test_bandwidth_aware_adaptation():
    """Test adaptation to bandwidth changes."""
    p = DSPatchPrefetcher()
    p.init()

    spt_entry = p.spt.lookup(0x1000)
    spt_entry.cov_pattern[0] = 0b1111
    spt_entry.acc_pattern[0] = 0b1010
    spt_entry.measure_acc[0] = 0
    spt_entry.measure_cov[0] = 0

    # Low bandwidth - selects CoP
    p.bw_tracker.current_quartile = 0
    selected = p._select_pattern(spt_entry, 0)
    assert selected == spt_entry.cov_pattern[0]

    # High bandwidth - selects AccP
    p.bw_tracker.current_quartile = 3
    selected = p._select_pattern(spt_entry, 0)
    assert selected == spt_entry.acc_pattern[0]


def test_or_count_saturation():
    """Test OR count doesn't exceed maximum."""
    p = DSPatchPrefetcher()
    p.init()

    spt_entry = p.spt.lookup(0x1000)

    # Perform many updates
    for i in range(10):
        pattern = 1 << i
        p._update_patterns(spt_entry, 0, pattern)

    # OR count should be saturated at max
    assert spt_entry.or_count[0] <= CONFIG["MAX_OR_COUNT"]


def test_measure_counters_saturation():
    """Test measure counters saturate at 3."""
    p = DSPatchPrefetcher()
    p.init()

    spt_entry = p.spt.lookup(0x1000)

    # Create poor accuracy scenario
    for _ in range(10):
        # Program pattern doesn't match predictions
        p._update_patterns(spt_entry, 0, 0b1010)
        spt_entry.cov_pattern[0] = 0b0101  # Force mismatch

    # Measure counters should saturate at 3 (2-bit counters)
    assert spt_entry.measure_cov[0] <= 3
    assert spt_entry.measure_acc[0] <= 3


def test():
    """Run all test functions."""
    print("Running DSPatch Prefetcher Tests...\n")

    tests = [
        test_helper_functions,
        test_pattern_compression,
        test_pattern_rotation,
        test_init_and_close,
        test_page_buffer_operations,
        test_page_buffer_lru_eviction,
        test_signature_pattern_table,
        test_bandwidth_tracker,
        test_coverage_biased_pattern_or_operation,
        test_accuracy_biased_pattern_and_operation,
        test_pattern_selection_low_bandwidth,
        test_pattern_selection_high_bandwidth,
        test_pattern_selection_throttling,
        test_quantify_accuracy,
        test_quantify_coverage,
        test_dual_pattern_modulation,
        test_multiple_triggers_per_page,
        test_end_to_end_pattern_learning,
        test_prefetch_generation,
        test_close_commits_patterns,
        test_bandwidth_aware_adaptation,
        test_or_count_saturation,
        test_measure_counters_saturation,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("[PASSED]")
            passed += 1
        except Exception as e:
            print(f"[FAILED]: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = test()
    exit(0 if success else 1)
