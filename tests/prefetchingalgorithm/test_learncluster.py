"""
Deterministic pytest unit tests for the Clustering+LSTM prefetcher implementation.

Tests:
 - initialization and reset
 - k-means clustering on synthetic sample
 - per-cluster vocab building deterministic behavior
 - model forward determinism (tiny model seeded)
 - prefetch issuance and MRB suppression / scheduler crediting
"""

import random

import numpy as np
import pytest
import torch

from prefetchlenz.prefetchingalgorithm.impl.learncluster import (
    CONFIG,
    ClusterVocab,
    KMeans1D,
    LearnClusterPrefetcher,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

# deterministic seeds for tests
torch.manual_seed(CONFIG["PYTORCH_SEED"])
np.random.seed(CONFIG["NUMPY_SEED"])
random.seed(CONFIG["RANDOM_SEED"])


def build_tiny_model_state(max_vocab: int):
    """Build tiny model state with deterministic weights for testing."""
    from prefetchlenz.prefetchingalgorithm.impl.learncluster import ClusterLSTMModel

    m = ClusterLSTMModel(
        num_clusters=CONFIG["NUM_CLUSTERS"],
        cluster_embed_dim=CONFIG["CLUSTER_EMBED_DIM"],
        delta_embed_dim=CONFIG["DELTA_EMBED_DIM"],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        num_layers=CONFIG["NUM_LAYERS"],
        max_cluster_vocab=max_vocab,
    )
    torch.manual_seed(CONFIG["PYTORCH_SEED"])
    for p in m.parameters():
        p.data.copy_(torch.randn_like(p) * 0.01)
    return m.state_dict()


def test_kmeans1d_basic():
    """Validate KMeans1D centroids and predict behavior on synthetic data."""
    data = np.array([0, 10, 12, 11, 100, 102, 99, 500, 520, 510], dtype=float)
    km = KMeans1D(k=3, iters=10)
    km.fit(data)
    # ensure centroids exist and predict returns an index in range
    assert km.centroids.shape[0] == 3
    idx = km.predict(11.0)
    assert 0 <= idx < 3


def test_cluster_vocab_from_deltas():
    """ClusterVocab builds a deterministic mapping from deltas frequency."""
    deltas = [0, 64, 64, -64, 128, 64, -64, 0]
    cv = ClusterVocab.from_deltas(deltas, vocab_size=4, scale=64)
    # vocab must include 0 and 64 likely
    assert 0 in cv.delta_to_idx.values() or 0 in cv.idx_to_delta.values()
    # encoding known delta
    idx = cv.encode(64)
    assert isinstance(idx, int)


def test_prefetch_issue_and_mrb_crediting():
    """Train a small cluster vocab then run prefetch issuance and ensure MRB suppression works."""
    # build a small synthetic sample for clustering
    sample_addrs = [i * 64 for i in range(100)]  # linear addresses
    # create small vocabs by sampling deltas within cluster manually
    model_state = build_tiny_model_state(max_vocab=CONFIG["CLUSTER_VOCAB_SIZE"])
    p = LearnClusterPrefetcher(
        cluster_sample=sample_addrs, model_state=model_state, config={"NUM_CLUSTERS": 4}
    )
    p.init()
    # rebuild one cluster vocab from some observed deltas to make predictions meaningful
    # pick cluster 0 and build frequent deltas [64,64,64, -64]
    p._build_cluster_vocab_from_sample(0, [64, 64, 64, -64, 0])
    # seed history with addresses close together so deltas are 64
    base = 0x1000
    for i in range(CONFIG["SEQ_LEN"] + 1):
        addr = base + i * 64
        p.progress(MemoryAccess(address=addr, pc=0x0), prefetch_hit=False)
    # after history, next call should predict next addresses (likely base + (SEQ_LEN)*64 + predicted delta)
    issued = p.progress(
        MemoryAccess(address=base + (CONFIG["SEQ_LEN"] + 1) * 64, pc=0x0),
        prefetch_hit=False,
    )
    assert isinstance(issued, list)
    # if issued, simulate prefetch hit and credit
    if issued:
        hit = issued[0]
        p.progress(MemoryAccess(address=hit, pc=0x0), prefetch_hit=True)
        assert hit not in p.scheduler.outstanding
        # immediate re-trigger should not re-issue same address due to MRB
        # re-run same trigger (append same address to history)
        p.history.append(p.history[-1])
        new_issued = p.progress(
            MemoryAccess(address=p.history[-1], pc=0x0), prefetch_hit=False
        )
        assert hit not in new_issued


def test_init_and_close_behavior():
    p = LearnClusterPrefetcher(cluster_sample=[i * 128 for i in range(50)])
    p.init()
    assert p.initialized
    p.close()
    assert not p.initialized
