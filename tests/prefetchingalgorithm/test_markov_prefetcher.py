import logging

from prefetchlenz.prefetchingalgorithm.impl.markovpredictor import MarkovPredictor
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

logger = logging.getLogger("test.markov")


def test_markov_basic_prediction():
    predictor = MarkovPredictor()
    predictor.init()

    # Simulated stream: 100 -> 200 -> 300 (repeat)
    accesses = [MemoryAccess(100, 1), MemoryAccess(200, 1), MemoryAccess(300, 1)] * 3
    predictions = []

    for acc in accesses:
        preds = predictor.progress(acc, prefetch_hit=False)
        predictions.append((acc.address, preds))

    # By the end, predictor should learn 100->200, 200->300
    preds_after_100 = [p for addr, p in predictions if addr == 100 and p]
    preds_after_200 = [p for addr, p in predictions if addr == 200 and p]

    assert any(200 in p for p in preds_after_100), "Should predict 200 after 100"
    assert any(300 in p for p in preds_after_200), "Should predict 300 after 200"


def test_first_order_prediction():
    p = MarkovPredictor()
    p.init()

    # Stream: 100 -> 200 -> 100 -> 200 -> 100 -> 200 ...
    seq = [100, 200, 100, 200, 100, 200]
    preds_seen = []
    for a in seq:
        preds = p.progress(MemoryAccess(address=a, pc=0))
        preds_seen.append((a, preds))

    # After first repetition, when processing the second occurrence of 100 we should predict 200
    found = False
    for addr, preds in preds_seen:
        if addr == 100 and preds:
            if 200 in preds:
                found = True
                break
    assert found, "Should predict 200 after seeing 100 repeatedly"


def test_second_order_prediction():
    p = MarkovPredictor()
    p.init()

    # Pattern: 10,20,30 repeated: we want (10,20) -> 30
    seq = [10, 20, 30, 10, 20, 30, 10, 20, 30]
    outputs = []
    for a in seq:
        preds = p.progress(MemoryAccess(address=a, pc=0))
        outputs.append((a, preds))

    # Find a point where current state is (10,20) and predictor predicts 30
    # Current state (10,20) occurs at accesses with addr==20 (after we pushed 20 into buffer)
    found = False
    # Walk outputs: when addr==20, curr_state=(10,20). Check preds for 30.
    for addr, preds in outputs:
        if addr == 20 and preds:
            if 30 in preds:
                found = True
                break

    assert found, "Second-order predictor should predict 30 after state (10,20)"


def test_no_prediction_initially():
    p = MarkovPredictor()
    p.init()
    preds = p.progress(MemoryAccess(address=999, pc=0))
    assert preds == [], "No prediction on very first access"
