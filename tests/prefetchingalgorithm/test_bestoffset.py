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
        prefetcher.progress(MemoryAccess(cpu=0, address=100 + 4 * i, pc=0), False)
        prefetches = prefetcher.progress(
            MemoryAccess(cpu=0, address=104 + 4 * i, pc=0), False
        )

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
        prefetcher.progress(MemoryAccess(cpu=0, address=100 + i, pc=0), False)

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
        prefetches = prefetcher.progress(
            MemoryAccess(cpu=0, address=i * 7, pc=0), False
        )
        assert not prefetches, "No prefetch should be issued when no pattern is found."

    assert (
        prefetcher.offset is None or prefetcher.offset <= prefetcher.bad_score
    ), "No offset should be selected if no pattern is found after max rounds."
    print("Test 3 passed: Prefetcher correctly did not issue prefetches.\n")
