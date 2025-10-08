"""
Clustering + LSTM prefetcher (Hashemi et al. style) — PyTorch inference implementation.

Implements the "Clustering + LSTM" variant:
 - cluster address space (k-means) into regions
 - build per-cluster delta vocab (quantized) and a shared LSTM encoder that takes
   cluster id + delta sequence and predicts top-K deltas within the cluster
 - inference-only prefetcher with MRB + Scheduler and deterministic behavior

Top-level adapter class: LearnClusterPrefetcher
 - init()
 - progress(access: MemoryAccess, prefetch_hit: bool) -> List[int]
 - close()

Approximations / notes (documented also in README):
 - Uses offline clustering (k-means) performed on a small synthetic sample or provided sample.
 - Per-cluster vocab is built by quantizing deltas observed within cluster; unknown deltas map to fallback index 0.
 - Model is a single shared LSTM taking delta indices and cluster ID embedding (implementation detail).
 - Offline training is not included; a state_dict can be supplied for inference.
 - All numeric sizes are configurable in CONFIG.

Requires:
 - PyTorch installed (torch)
 - Repository provides MemoryAccess dataclass in prefetchlenz.prefetchingalgorithm.impl.types
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.impl.learncluster")

# -------------------------
# Configuration (all tunable)
# -------------------------
CONFIG: Dict[str, Any] = {
    # clustering
    "NUM_CLUSTERS": 8,  # k in k-means (paper explores many clusterings)
    "CLUSTER_SAMPLE_SIZE": 1024,  # how many deltas to sample to run k-means on
    "KMEANS_ITERS": 20,
    # per-cluster vocab / quantization
    "CLUSTER_VOCAB_SIZE": 64,  # per-cluster output vocab (small for tests)
    "DELTA_BUCKET_SCALE": 64,  # quantization scale (bytes)
    # model
    "CLUSTER_EMBED_DIM": 8,
    "DELTA_EMBED_DIM": 12,
    "HIDDEN_DIM": 48,
    "NUM_LAYERS": 1,
    "TOPK": 4,  # top-K predicted deltas in cluster
    "SEQ_LEN": 4,  # sequence length of deltas fed (inference window)
    "DEVICE": "cpu",
    # scheduling / MRB
    "PREFETCH_DEGREE": 4,
    "MAX_OUTSTANDING": 32,
    "MRB_SIZE": 64,
    # deterministic seeds
    "PYTORCH_SEED": 1337,
    "NUMPY_SEED": 1337,
    "RANDOM_SEED": 1337,
}

# deterministic seeds for reproducibility
torch.manual_seed(CONFIG["PYTORCH_SEED"])
np.random.seed(CONFIG["NUMPY_SEED"])
random.seed(CONFIG["RANDOM_SEED"])


# -------------------------
# Utility helpers
# -------------------------
def region_block(addr: int, block_size: int = 64) -> int:
    """Return block-aligned address base (cache-block granularity)."""
    return (addr // block_size) * block_size


def delta(a_next: int, a_prev: int) -> int:
    return int(a_next - a_prev)


# -------------------------
# KMeans clustering (1D values: absolute address / coarse region)
# -------------------------
class KMeans1D:
    """Very small deterministic 1D k-means implementation on numpy arrays."""

    def __init__(self, k: int, iters: int = 20):
        self.k = k
        self.iters = iters
        self.centroids: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray):
        """Fit k-means to 1D data (expects shape (N,))."""
        if data.size == 0:
            self.centroids = np.linspace(0, 1, self.k)
            return
        # deterministic init: choose k quantiles
        qs = np.linspace(0, 100, self.k + 2)[1:-1]
        centroids = np.percentile(data, qs).astype(float)
        for _ in range(self.iters):
            # assign
            dists = np.abs(data[:, None] - centroids[None, :])  # (N, k)
            idx = np.argmin(dists, axis=1)
            new_cent = np.array(
                [
                    data[idx == j].mean() if np.any(idx == j) else centroids[j]
                    for j in range(self.k)
                ]
            )
            if np.allclose(new_cent, centroids):
                break
            centroids = new_cent
        self.centroids = centroids

    def predict(self, x: float) -> int:
        """Return index of nearest centroid for value x."""
        assert self.centroids is not None, "KMeans not fitted"
        return int(np.argmin(np.abs(self.centroids - x)))


# -------------------------
# Per-cluster vocab & quantizer
# -------------------------
@dataclass
class ClusterVocab:
    """Per-cluster mapping index <-> quantized delta."""

    idx_to_delta: Dict[int, int]
    delta_to_idx: Dict[int, int]

    @classmethod
    def from_deltas(cls, deltas: List[int], vocab_size: int, scale: int):
        """Build quantized vocab by taking most frequent quantized deltas up to vocab_size."""
        # quantize
        q = [
            (d // scale) * scale if d >= 0 else -(((-d) // scale) * scale)
            for d in deltas
        ]
        # frequency
        freq = {}
        for val in q:
            freq[val] = freq.get(val, 0) + 1
        # sort by frequency
        items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        top = [item[0] for item in items[:vocab_size]]
        if 0 not in top:
            top = [0] + [v for v in top if v != 0]
            top = top[:vocab_size]
        idx_to_delta = {i: d for i, d in enumerate(top)}
        delta_to_idx = {d: i for i, d in idx_to_delta.items()}
        return cls(idx_to_delta=idx_to_delta, delta_to_idx=delta_to_idx)

    def encode(self, d: int) -> int:
        q = (
            (d // CONFIG["DELTA_BUCKET_SCALE"]) * CONFIG["DELTA_BUCKET_SCALE"]
            if d >= 0
            else -(
                ((-d) // CONFIG["DELTA_BUCKET_SCALE"]) * CONFIG["DELTA_BUCKET_SCALE"]
            )
        )
        return self.delta_to_idx.get(q, 0)

    def decode(self, idx: int) -> int:
        return self.idx_to_delta[idx]


# -------------------------
# Shared LSTM model that conditions on cluster id and delta sequence
# -------------------------
class ClusterLSTMModel(nn.Module):
    """
    Shared LSTM with:
      - cluster embedding
      - per-delta embedding (indices are per-cluster; but we reuse single embedding of size CLUSTER_VOCAB_SIZE)
      - LSTM that concatenates cluster embed with delta embed at each timestep
      - final linear layer mapping to per-cluster vocab logits (we implement a shared output and mask)
    Note: for simplicity we use a single output head of size max_cluster_vocab and mask unused indices at inference.
    """

    def __init__(
        self,
        num_clusters: int,
        cluster_embed_dim: int,
        delta_embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        max_cluster_vocab: int,
    ):
        super().__init__()
        self.cluster_embed = nn.Embedding(num_clusters, cluster_embed_dim)
        self.delta_embed = nn.Embedding(max_cluster_vocab, delta_embed_dim)
        self.lstm = nn.LSTM(
            input_size=cluster_embed_dim + delta_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, max_cluster_vocab)

    def forward(
        self,
        cluster_ids: torch.LongTensor,
        seq_indices: torch.LongTensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        cluster_ids: (batch,) cluster id per sequence
        seq_indices: (batch, seq_len) deltas indices (these are per-cluster indices mapped into global index space)
        """
        batch = seq_indices.size(0)
        seq_len = seq_indices.size(1)
        c_emb = (
            self.cluster_embed(cluster_ids).unsqueeze(1).repeat(1, seq_len, 1)
        )  # (batch, seq_len, cdim)
        d_emb = self.delta_embed(seq_indices)  # (batch, seq_len, ddim)
        inp = torch.cat([c_emb, d_emb], dim=2)
        out, hidden = self.lstm(inp, hidden)
        final = out[:, -1, :]
        logits = self.fc(final)
        return logits, hidden


# -------------------------
# MRB & Scheduler (same pattern as other modules)
# -------------------------
class MRB:
    def __init__(self, size: int = CONFIG["MRB_SIZE"]):
        self.size = size
        self.buf: List[int] = []

    def insert(self, a: int):
        if a in self.buf:
            self.buf.remove(a)
            self.buf.append(a)
            return
        if len(self.buf) >= self.size:
            self.buf.pop(0)
        self.buf.append(a)

    def contains(self, a: int) -> bool:
        return a in self.buf

    def clear(self):
        self.buf.clear()


class Scheduler:
    def __init__(
        self,
        max_outstanding: int = CONFIG["MAX_OUTSTANDING"],
        mrbsz: int = CONFIG["MRB_SIZE"],
    ):
        self.max_outstanding = max_outstanding
        self.outstanding: List[int] = []
        self.mrb = MRB(size=mrbsz)

    def issue(self, candidates: List[int], degree: int) -> List[int]:
        issued = []
        for a in candidates:
            if len(issued) >= degree:
                break
            if a in self.outstanding:
                continue
            if self.mrb.contains(a):
                continue
            if len(self.outstanding) >= self.max_outstanding:
                break
            self.outstanding.append(a)
            self.mrb.insert(a)
            issued.append(a)
            logger.info("Scheduler issued %x", a)
        return issued

    def credit(self, addr: int):
        if addr in self.outstanding:
            self.outstanding.remove(addr)
        self.mrb.insert(addr)

    def clear(self):
        self.outstanding.clear()
        self.mrb.clear()


# -------------------------
# Top-level adapter
# -------------------------
class LearnClusterPrefetcher:
    """
    Top-level adapter implementing the clustering + LSTM prefetcher.

    Methods:
      - init()
      - progress(access: MemoryAccess, prefetch_hit: bool) -> List[int]
      - close()

    Behavior:
      - Maintains last N addresses to form delta sequence (SEQ_LEN)
      - On miss-trigger (progress called with prefetch_hit False), quantize deltas, map addresses to cluster via kmeans on block bases,
        encode per-cluster indices, run model forward, obtain top-K per-cluster predictions, map them back to absolute addresses,
        and issue prefetches via Scheduler.
      - On prefetch_hit True: credit scheduler and keep suppression via MRB.

    Configurable params are in CONFIG.
    """

    def __init__(
        self,
        cluster_sample: Optional[List[int]] = None,
        model_state: Optional[Dict] = None,
        cache_factory: Optional[Any] = None,
        config: Optional[Dict] = None,
    ):
        # merge config overrides
        if config:
            for k, v in config.items():
                if k in CONFIG:
                    CONFIG[k] = v

        self.device = torch.device(CONFIG["DEVICE"])
        # clustering model
        self.kmeans = KMeans1D(k=CONFIG["NUM_CLUSTERS"], iters=CONFIG["KMEANS_ITERS"])
        self.cluster_vocabs: List[ClusterVocab] = []
        self.max_cluster_vocab = CONFIG["CLUSTER_VOCAB_SIZE"]
        # sample for clustering: expects list of addresses (block bases); if none, use empty sample and fit trivial centroids
        sample = np.array(cluster_sample or [], dtype=float)
        if sample.size > 0:
            self.kmeans.fit(sample)
        else:
            # trivial centroids (0..k-1)
            self.kmeans.centroids = np.linspace(0, 1, CONFIG["NUM_CLUSTERS"])

        # build empty vocabs initially
        for _ in range(CONFIG["NUM_CLUSTERS"]):
            # default vocab containing zero delta only
            cv = ClusterVocab.from_deltas(
                [0], CONFIG["CLUSTER_VOCAB_SIZE"], CONFIG["DELTA_BUCKET_SCALE"]
            )
            self.cluster_vocabs.append(cv)

        # model
        self.model = ClusterLSTMModel(
            num_clusters=CONFIG["NUM_CLUSTERS"],
            cluster_embed_dim=CONFIG["CLUSTER_EMBED_DIM"],
            delta_embed_dim=CONFIG["DELTA_EMBED_DIM"],
            hidden_dim=CONFIG["HIDDEN_DIM"],
            num_layers=CONFIG["NUM_LAYERS"],
            max_cluster_vocab=self.max_cluster_vocab,
        ).to(self.device)
        if model_state:
            # Filter checkpoint to remove any tensors with mismatched shapes.
            safe_state = {}
            for k, v in model_state.items():
                if k in self.model.state_dict():
                    target_shape = tuple(self.model.state_dict()[k].shape)
                    if tuple(v.shape) == target_shape:
                        safe_state[k] = v
                    else:
                        logger.warning(
                            "Skipping key %s due to shape mismatch: checkpoint %s vs model %s",
                            k,
                            tuple(v.shape),
                            target_shape,
                        )
                else:
                    logger.debug("Skipping unexpected key %s", k)
            load_res = self.model.load_state_dict(safe_state, strict=False)
            logger.info(
                "Loaded filtered model_state: kept %d / %d tensors",
                len(safe_state),
                len(model_state),
            )
        self.model.eval()

        # scheduler
        self.scheduler = Scheduler(
            max_outstanding=CONFIG["MAX_OUTSTANDING"], mrbsz=CONFIG["MRB_SIZE"]
        )

        # history buffer for last SEQ_LEN addresses (block bases)
        self.history: List[int] = []

        # optional metadata store via cache_factory (not required)
        self.cache_factory = cache_factory
        if cache_factory is not None:
            try:
                self.meta = cache_factory(16, 4)
            except Exception:
                self.meta = {}
        else:
            self.meta = {}

        self.initialized = False

    def init(self):
        logger.info("LearnClusterPrefetcher.init")
        self.initialized = True

    def _addr_to_cluster(self, addr: int) -> int:
        b = region_block(addr)
        return self.kmeans.predict(float(b))

    def _build_cluster_vocab_from_sample(self, cluster_idx: int, deltas: List[int]):
        """Rebuild a single cluster vocab from observed deltas (called during warmup in tests)."""
        cv = ClusterVocab.from_deltas(
            deltas, CONFIG["CLUSTER_VOCAB_SIZE"], CONFIG["DELTA_BUCKET_SCALE"]
        )
        self.cluster_vocabs[cluster_idx] = cv

    def _encode_sequence(self, cluster_idx: int, deltas: List[int]) -> List[int]:
        """Encode deltas into per-cluster indices (pad/truncate to SEQ_LEN)."""
        cv = self.cluster_vocabs[cluster_idx]
        seq = [cv.encode(d) for d in deltas[-CONFIG["SEQ_LEN"] :]]
        # left-pad with zeros if short
        if len(seq) < CONFIG["SEQ_LEN"]:
            seq = [0] * (CONFIG["SEQ_LEN"] - len(seq)) + seq
        return seq

    def _predict_topk_in_cluster(
        self, cluster_idx: int, seq_indices: List[int]
    ) -> List[int]:
        """Run model forward and return top-K per-cluster decoded deltas."""
        self.model.eval()
        with torch.no_grad():
            cluster_ids = torch.LongTensor([cluster_idx]).to(self.device)
            seq = torch.LongTensor([seq_indices]).to(self.device)
            logits, _ = self.model(cluster_ids, seq)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            topk = np.argsort(-probs)[: CONFIG["TOPK"]].tolist()
        # decode using cluster vocab (indices are 0..max_vocab-1)
        cv = self.cluster_vocabs[cluster_idx]
        deltas = []
        for idx in topk:
            if idx in cv.idx_to_delta:
                deltas.append(cv.decode(idx))
        return deltas

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Handle one memory access (treated as miss-trigger). Behavior:
          - if prefetch_hit True: credit scheduler
          - else: append to history, when enough history available:
              * map current address's cluster
              * build delta sequence from history
              * encode into per-cluster indices
              * model predicts top-K deltas (within cluster)
              * map deltas -> absolute addresses, filter duplicates, ask scheduler to issue up to PREFETCH_DEGREE
        """
        if not self.initialized:
            self.init()

        addr = access.address
        issued: List[int] = []

        if prefetch_hit:
            # credit usage of prefetched address
            logger.debug("prefetch_hit for %x", addr)
            self.scheduler.credit(addr)
            # update history with this access too
            self.history.append(addr)
            if len(self.history) > CONFIG["SEQ_LEN"]:
                self.history = self.history[-CONFIG["SEQ_LEN"] :]
            return []

        # treat every call as a miss-trigger (paper triggers on misses)
        self.history.append(addr)
        if len(self.history) > CONFIG["SEQ_LEN"] + 1:
            # keep history limited to SEQ_LEN+1 so we can compute deltas
            self.history = self.history[-(CONFIG["SEQ_LEN"] + 1) :]

        # need at least two addresses to compute a delta sequence
        if len(self.history) < 2:
            return []

        # compute deltas relative to previous addresses (most recent last)
        deltas = []
        for i in range(1, len(self.history)):
            deltas.append(delta(self.history[i], self.history[i - 1]))

        # current cluster is cluster of the latest address
        cluster_idx = self._addr_to_cluster(self.history[-1])

        # encode sequence for that cluster
        seq_indices = self._encode_sequence(cluster_idx, deltas)

        # predict top-K deltas in that cluster
        pred_deltas = self._predict_topk_in_cluster(cluster_idx, seq_indices)

        # map deltas to absolute predicted addresses (predict next = latest addr + delta)
        candidates = []
        base = self.history[-1]
        for d in pred_deltas:
            p = int(base + d)
            if p not in candidates:
                candidates.append(p)

        # issue via scheduler (handles MRB/outstanding)
        issued = self.scheduler.issue(candidates, CONFIG["PREFETCH_DEGREE"])
        return issued

    def close(self):
        logger.info("LearnClusterPrefetcher.close")
        self.scheduler.clear()
        self.history.clear()
        self.initialized = False
