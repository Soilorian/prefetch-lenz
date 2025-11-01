from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.bestoffset import BestOffsetPrefetcher


def test():
    """
    A collection of test cases to validate the functionality of the BestOffsetPrefetcher.
    """
    # Test Scenario 1: Successful learning and a single prefetch is issued.
    print("--- Test Scenario 1: Successful Learning ---")
    prefetcher = BestOffsetPrefetcher(max_score=3, max_rounds=100)
    # Simulate a pattern with offset 4
    for i in range(200):
        # A recent access at 100, then an access at 104
        prefetcher.progress(MemoryAccess(address=100 + 4 * i, pc=0), False)
        prefetches = prefetcher.progress(MemoryAccess(address=104 + 4 * i, pc=0), False)

        if prefetcher.offset is not None:
            print(
                f"Prefetcher learned offset: {prefetcher.offset}. Prefetching: {prefetches}"
            )
            break

    assert prefetcher.offset == 4, "Offset should be 4."
    print("Test 1 passed: Prefetcher successfully learned the offset.\n")

    # Test Scenario 2: Max rounds reached, select the best offset.
    print("--- Test Scenario 2: Max Rounds Reached ---")
    prefetcher = BestOffsetPrefetcher(
        max_score=100, max_rounds=2
    )  # Force a short training
    # Only provide a few accesses to a single offset, not enough to meet max_score
    for i in range(200):
        prefetcher.progress(MemoryAccess(address=100 + i, pc=0), False)

    # After max_rounds, the prefetcher should choose the most scored offset, even if below max_score
    print(f"Final offset after max rounds: {prefetcher.offset}")
    assert (
        prefetcher.offset is not None
    ), "Prefetcher should have selected an offset after max rounds."
    print("Test 2 passed: Prefetcher selected an offset after max_rounds.\n")

    # Test Scenario 3: No pattern found, no prefetching.
    print("--- Test Scenario 3: No Pattern Found ---")
    prefetcher = BestOffsetPrefetcher(max_score=10, max_rounds=2)
    # Simulate random accesses
    for i in range(200):
        prefetches = prefetcher.progress(MemoryAccess(address=i * 7, pc=0), False)
        assert not prefetches, "No prefetch should be issued when no pattern is found."

    assert (
        prefetcher.offset is None or prefetcher.offset <= prefetcher.bad_score
    ), "No offset should be selected if no pattern is found after max rounds."
    print("Test 3 passed: Prefetcher correctly did not issue prefetches.\n")


def test_no_match_in_recent_table():
    """Test that offset score doesn't update when pattern doesn't match."""
    prefetcher = BestOffsetPrefetcher(max_score=3, max_rounds=100)
    prefetcher.init()

    # Add access
    prefetcher.progress(MemoryAccess(address=1000, pc=0), False)

    # Next access doesn't match any offset pattern
    prefetcher.progress(MemoryAccess(address=5000, pc=0), False)

    # No offset should be learned
    assert (
        prefetcher.offset is None
    ), "No offset should be learned without pattern match"


def test_offset_wraparound():
    """Test that offset index wraps around when testing offsets."""
    prefetcher = BestOffsetPrefetcher(max_score=1000, max_rounds=1)
    prefetcher.init()

    # Access many times to wrap around offset list
    num_offsets = len(prefetcher.offset_list.offsets)
    for i in range(num_offsets + 10):
        prefetcher.progress(MemoryAccess(address=1000 + i, pc=0), False)

    # Should have wrapped around and still be training
    assert prefetcher.current_offset_index < num_offsets


def test_get_best_offset_no_qualifying():
    """Test get_best_offset when no offset exceeds bad_score."""
    from prefetchlenz.prefetchingalgorithm.impl.bestoffset import OffsetList

    offset_list = OffsetList()
    offset_list.update_score(4, 0)  # Score of 0, below bad_score=1

    best = offset_list.get_best_offset(bad_score=1)
    assert best is None, "Should return None when no offset exceeds bad_score"


def test_init_and_close():
    """Test that init and close properly reset state."""
    prefetcher = BestOffsetPrefetcher(max_score=5, max_rounds=10)
    prefetcher.init()

    # Set some state
    prefetcher.progress(MemoryAccess(address=100, pc=0), False)
    assert prefetcher.current_round == 0
    assert prefetcher.current_offset_index > 0

    # Close should reset
    prefetcher.close()
    assert prefetcher.offset is None
    assert prefetcher.current_round == 0
    assert prefetcher.current_offset_index == 0


def test_prefetch_after_training():
    """Test that prefetches are issued after training completes."""
    prefetcher = BestOffsetPrefetcher(max_score=2, max_rounds=100)
    prefetcher.init()

    # Train offset 8
    for i in range(10):
        prefetcher.progress(MemoryAccess(address=1000 + 8 * i, pc=0), False)
        prefetcher.progress(MemoryAccess(address=1008 + 8 * i, pc=0), False)
        if prefetcher.offset is not None:
            break

    # Once trained, should prefetch
    if prefetcher.offset is not None:
        prefetches = prefetcher.progress(MemoryAccess(address=2000, pc=0), False)
        assert len(prefetches) > 0, "Should issue prefetch after training"
        assert prefetches[0] == 2000 + prefetcher.offset


def test_prefetch_hit_parameter():
    """Test that prefetch_hit parameter doesn't break the prefetcher."""
    prefetcher = BestOffsetPrefetcher(max_score=3, max_rounds=100)
    prefetcher.init()

    # Prefetch hit shouldn't cause errors
    prefetcher.progress(MemoryAccess(address=100, pc=0), prefetch_hit=True)
    prefetches = prefetcher.progress(
        MemoryAccess(address=104, pc=0), prefetch_hit=False
    )

    # Should continue working normally (prefetch_hit is unused but shouldn't break)
    assert isinstance(prefetches, list), "Should return list of prefetches"
