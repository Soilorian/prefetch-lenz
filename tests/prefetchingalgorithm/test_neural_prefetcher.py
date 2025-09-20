# test_hierarchical_prefetcher.py
from prefetchlenz.prefetchingalgorithm.impl.markovpredictor import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.neural import HierarchicalNeuralPrefetcher


def test_simple_sequence_learns_prediction():
    p = HierarchicalNeuralPrefetcher(vocab_cap=64, combine_alpha=0.9, top_k=1)
    p.init()

    pc = 0x10
    # create a small repeating pattern: 100 -> 110 -> 120 -> 130 ...
    seq = [100, 110, 120, 130, 140, 150] * 4

    preds_at = []
    for a in seq:
        preds = p.progress(MemoryAccess(address=a, pc=pc), prefetch_hit=False)
        preds_at.append((a, preds))

    # After several repetitions the model should have learned the +10 delta
    # Look for a point where when address==120, it predicts 130 (i.e., +10)
    found = False
    for addr, preds in preds_at:
        if addr == 120 and preds:
            if 130 in preds:
                found = True
                break
    assert found, "Predictor should learn +10 delta and predict next address 130"


def test_alternate_pcs_have_local_models():
    p = HierarchicalNeuralPrefetcher(vocab_cap=64, combine_alpha=0.9, top_k=1)
    p.init()

    pc1 = 1
    pc2 = 2
    # pc1 sequence: 100,110,120 repeat
    for a in [100, 110, 120, 100, 110, 120]:
        p.progress(MemoryAccess(address=a, pc=pc1), prefetch_hit=False)
    # pc2 sequence: 200,260,320 (stride +60)
    for a in [200, 260, 320, 200, 260, 320]:
        p.progress(MemoryAccess(address=a, pc=pc2), prefetch_hit=False)

    # Now when we at pc1 address 110 we expect prediction around 120
    preds1 = p.progress(MemoryAccess(address=110, pc=pc1), prefetch_hit=False)
    assert any(
        abs(pred - 110 - 10) <= 0 for pred in preds1
    ), "pc1 should predict +10 stride"

    # For pc2 at 260 expect +60
    preds2 = p.progress(MemoryAccess(address=260, pc=pc2), prefetch_hit=False)
    assert any(
        abs(pred - 260 - 60) <= 0 for pred in preds2
    ), "pc2 should predict +60 stride"


def test_vocab_capacity_limits_new_deltas():
    p = HierarchicalNeuralPrefetcher(vocab_cap=2, combine_alpha=0.5, top_k=1)
    p.init()
    pc = 0x11
    # generate 3 distinct deltas; vocab cap=2 so third delta won't be added
    p.progress(MemoryAccess(address=100, pc=pc), prefetch_hit=False)  # no delta
    p.progress(MemoryAccess(address=110, pc=pc), prefetch_hit=False)  # delta +10
    p.progress(MemoryAccess(address=200, pc=pc), prefetch_hit=False)  # delta +90
    p.progress(
        MemoryAccess(address=305, pc=pc), prefetch_hit=False
    )  # delta +105 -> not added due to cap

    # check vocab size <= cap
    assert len(p.vocab.id_to_delta) <= 2
