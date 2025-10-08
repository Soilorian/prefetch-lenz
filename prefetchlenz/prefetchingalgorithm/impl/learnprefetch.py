"""
Learning Memory Access Patterns prefetcher (PyTorch LSTM model).

Implements a runtime inference wrapper around an LSTM that predicts address deltas
(top-K) given a stream of misses. Designed to fit the repo's PrefetchAlgorithm API:
  class PrefetchAlgorithm: init(), progress(access: MemoryAccess, prefetch_hit: bool) -> List[int], close()

Design notes / mapping to Hashemi et al.:
 - We model deltas (Addr_{t+1} - Addr_t) as the prediction target (paper uses delta vocabulary).
 - We support a small embedding LSTM (configurable) and Top-K softmax outputs.
 - The paper trains large models offline; this implementation exposes a PyTorch model object
   and performs inference only. (Online training is intentionally omitted; see README for rationale.)
 - All numeric limits are configurable via CONFIG.
 - Uses an MRB + Scheduler to suppress immediate reissuance of prefetched addresses.

Requirements:
 - PyTorch must be installed (torch).
 - The repo must provide MemoryAccess dataclass in prefetchlenz.prefetchingalgorithm.impl.types
 - PrefetchAlgorithm interface should be available; this class implements the same API.

Approximations (documented in README):
 - Small model sizes (embedding/hidden dims) for unit tests; paper uses larger sizes.
 - Vocab construction: tests use a small synthetic delta->index mapping; in a real deployment
   you must build the vocabulary from traces (top-N deltas).
 - Offline training: not included (paper trains offline). We include code paths for loading
   a trained model state_dict (torch) if available.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.impl.learnprefetch")

# -------------------------
# Config (all tunable)
# -------------------------
CONFIG: Dict[str, int] = {
    # Model / vocab
    "DELTA_VOCAB_SIZE": 512,  # output vocabulary size (paper used up to ~50k; tests use small values)
    "EMBED_DIM": 16,
    "HIDDEN_DIM": 32,
    "NUM_LAYERS": 1,
    "TOPK": 4,  # top-K predictions to use for prefetch candidates
    # Prefetch scheduling / MRB
    "PREFETCH_DEGREE": 4,  # how many addresses to attempt to issue per trigger
    "MAX_OUTSTANDING": 32,
    "MRB_SIZE": 64,
    # Delta quantization default (for mapping predicted ids -> delta value)
    # In practice you build a discrete mapping from training traces. For tests we supply one.
    "DELTA_BUCKET_SCALE": 64,
    # Device
    "DEVICE": "cpu",
    # Deterministic seeds for tests & reproducibility
    "PYTORCH_SEED": 1234,
    "NUMPY_SEED": 1234,
    "RANDOM_SEED": 1234,
}

# Re-seed deterministically for consistent runs when imported in testing
torch.manual_seed(CONFIG["PYTORCH_SEED"])
np.random.seed(CONFIG["NUMPY_SEED"])
random.seed(CONFIG["RANDOM_SEED"])


# -------------------------
# Small utility dataclasses
# -------------------------
@dataclass
class Vocab:
    """
    Vocabulary mapping between quantized deltas and integer indices.
    For simplicity this class implements a mapping from an integer delta (quantized)
    -> index and back. In production this would be constructed from traces (top-N deltas).
    """

    idx_to_delta: Dict[int, int]
    delta_to_idx: Dict[int, int]

    @classmethod
    def from_list(cls, deltas: List[int]) -> "Vocab":
        idx_to_delta = {i: d for i, d in enumerate(deltas)}
        delta_to_idx = {d: i for i, d in idx_to_delta.items()}
        return cls(idx_to_delta=idx_to_delta, delta_to_idx=delta_to_idx)

    def encode(self, delta: int) -> int:
        """Encode quantized delta -> index. If unknown, return a fallback index (0)."""
        return self.delta_to_idx.get(delta, 0)

    def decode(self, idx: int) -> int:
        """Decode index -> quantized delta; raises KeyError if missing."""
        return self.idx_to_delta[idx]

    def topk_to_deltas(self, indices: List[int]) -> List[int]:
        """Map a list of indices (top-k) to quantized deltas."""
        return [self.decode(i) for i in indices if i in self.idx_to_delta]


# -------------------------
# Delta encoder (quantizer)
# -------------------------
class DeltaQuantizer:
    """
    Simple quantizer that maps raw address deltas into discrete buckets.
    This is a stand-in for the quantization pipeline used in the paper (vocab construction).
    For deterministic tests we keep it simple and symmetric.

    Methods:
      - quantize(delta) -> int   (returns quantized delta)
    """

    def __init__(self, scale: int = CONFIG["DELTA_BUCKET_SCALE"]):
        self.scale = int(scale)

    def quantize(self, delta: int) -> int:
        """Quantize delta by rounding to nearest multiple of scale (signed)."""
        if delta >= 0:
            return (delta // self.scale) * self.scale
        else:
            # keep sign preserve
            return -((-delta) // self.scale) * self.scale


# -------------------------
# PyTorch LSTM model wrapper
# -------------------------
class DeltaLSTMModel(nn.Module):
    """
    Small embedding + LSTM model that takes encoded delta indices as input and
    outputs logits over DELTA_VOCAB_SIZE classes.

    Architecture:
     - nn.Embedding(DELTA_VOCAB_SIZE, EMBED_DIM)
     - nn.LSTM(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
     - nn.Linear(HIDDEN_DIM, DELTA_VOCAB_SIZE)
    """

    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        x: LongTensor of shape (batch, seq_len) containing indices in [0, vocab_size)
        Returns logits of shape (batch, vocab_size) for the final timestep.
        """
        emb = self.embed(x)  # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(emb, hidden)  # out: (batch, seq_len, hidden_dim)
        final = out[:, -1, :]  # take last time step
        logits = self.fc(final)  # (batch, vocab_size)
        return logits, hidden


# -------------------------
# MRB + Scheduler
# -------------------------
class MRB:
    """Short-term suppression buffer to avoid immediate re-issue of same addresses."""

    def __init__(self, size: int = CONFIG["MRB_SIZE"]):
        self.size = size
        self.buf: List[int] = []

    def insert(self, addr: int) -> None:
        if addr in self.buf:
            self.buf.remove(addr)
            self.buf.append(addr)
            return
        if len(self.buf) >= self.size:
            ev = self.buf.pop(0)
            logger.debug("MRB evicted %x", ev)
        self.buf.append(addr)
        logger.debug("MRB insert %x", addr)

    def contains(self, addr: int) -> bool:
        return addr in self.buf

    def clear(self) -> None:
        self.buf.clear()


class Scheduler:
    """
    Issue prefetches subject to outstanding cap and MRB suppression.
    - outstanding: list of outstanding addresses (deterministic ordering).
    """

    def __init__(
        self,
        max_outstanding: int = CONFIG["MAX_OUTSTANDING"],
        mrbsz: int = CONFIG["MRB_SIZE"],
    ):
        self.max_outstanding = max_outstanding
        self.outstanding: List[int] = []
        self.mrb = MRB(size=mrbsz)

    def can_issue(self) -> bool:
        return len(self.outstanding) < self.max_outstanding

    def issue(self, candidates: List[int], degree: int) -> List[int]:
        """Issue up to degree addresses from candidates, subject to MRB/outstanding."""
        issued = []
        for addr in candidates:
            if len(issued) >= degree:
                break
            if addr in self.outstanding:
                continue
            if self.mrb.contains(addr):
                continue
            if not self.can_issue():
                break
            self.outstanding.append(addr)
            self.mrb.insert(addr)
            issued.append(addr)
            logger.info("Scheduler issued prefetch %x", addr)
        return issued

    def credit(self, addr: int) -> None:
        """Credit when a prefetched address is used by the program."""
        if addr in self.outstanding:
            self.outstanding.remove(addr)
            logger.debug("Scheduler credited outstanding for %x", addr)
        # keep it in MRB to avoid immediate re-issue
        self.mrb.insert(addr)

    def clear(self) -> None:
        self.outstanding.clear()
        self.mrb.clear()


# -------------------------
# Main Prefetcher adapter
# -------------------------
class LearnPrefetcher:
    """
    PrefetchAlgorithm adapter implementing inference-time Behavior from Hashemi et al.

    Public methods:
      - init()
      - progress(access: MemoryAccess, prefetch_hit: bool) -> List[int]
      - close()

    Key behavior:
      - Maintains short history (last address) to compute delta
      - Quantizes delta via DeltaQuantizer, encodes via Vocab (index)
      - Feeds a short sequence (we use last-N deltas; tests use seq_len=1 for simplicity)
      - Runs the LSTM forward, extracts top-K logits and maps indices -> deltas -> addresses
      - Issues prefetches through Scheduler (MRB suppression)
    """

    def __init__(
        self,
        vocab: Optional[Vocab] = None,
        model_state: Optional[Dict] = None,
        cache_factory: Optional[callable] = None,
        config: Optional[Dict] = None,
    ):
        """
        vocab: optional Vocab instance (idx->delta mapping). If None, a default small vocab is created.
        model_state: optional torch state_dict to load into the model.
        cache_factory: optional callable to create a Cache-like metadata store (not used heavily here).
        config: optional config overrides (merged into CONFIG).
        """
        # Merge config overrides
        if config:
            for k, v in config.items():
                if k in CONFIG:
                    CONFIG[k] = v

        self.device = torch.device(CONFIG["DEVICE"])
        self.delta_quant = DeltaQuantizer(scale=CONFIG["DELTA_BUCKET_SCALE"])
        # Build/accept vocabulary
        if vocab is None:
            # default small vocab: zero delta + +/- small multiples
            base_deltas = [0] + [
                i * CONFIG["DELTA_BUCKET_SCALE"]
                for i in range(1, CONFIG["DELTA_VOCAB_SIZE"])
            ]
            base_deltas = base_deltas[: CONFIG["DELTA_VOCAB_SIZE"]]
            self.vocab = Vocab.from_list(base_deltas)
        else:
            self.vocab = vocab

        # Build model
        self.model = DeltaLSTMModel(
            vocab_size=CONFIG["DELTA_VOCAB_SIZE"],
            embed_dim=CONFIG["EMBED_DIM"],
            hidden_dim=CONFIG["HIDDEN_DIM"],
            num_layers=CONFIG["NUM_LAYERS"],
        ).to(self.device)
        if model_state:
            # load provided trained weights
            self.model.load_state_dict(model_state)
            logger.info("Loaded model state_dict")
        # model in eval mode for inference
        self.model.eval()

        # Simple short history: for tests we'll use seq_len=1 (last delta). Could be extended.
        self.last_addr: Optional[int] = None

        # Scheduler
        self.scheduler = Scheduler(
            max_outstanding=CONFIG["MAX_OUTSTANDING"], mrbsz=CONFIG["MRB_SIZE"]
        )

        # deterministic seeds for inference
        torch.manual_seed(CONFIG["PYTORCH_SEED"])
        self.initialized = False

        # optional metadata store (not required in this implementation, placeholder for Cache usage)
        self.cache_factory = cache_factory
        if cache_factory is not None:
            try:
                # attempt to create a tiny metadata cache (not heavily used)
                self.meta_cache = cache_factory(16, 4)  # default small
            except Exception:
                self.meta_cache = {}
        else:
            self.meta_cache = {}

    def init(self) -> None:
        logger.info("LearnPrefetcher.init")
        self.initialized = True

    def _predict_deltas(self, seq_indices: List[int]) -> List[int]:
        """
        Run the LSTM forward for a batch of one sequence and return top-K predicted delta indices.
        seq_indices: list of indices (e.g., [i_last, ...]) shape (seq_len,)
        Returns: list of predicted delta indices (top-K ordered)
        """
        x = (
            torch.LongTensor(seq_indices).unsqueeze(0).to(self.device)
        )  # shape (1, seq_len)
        with torch.no_grad():
            logits, _ = self.model(x)  # logits shape (1, vocab_size)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            # top-K indices
            topk = np.argsort(-probs)[: CONFIG["TOPK"]].tolist()
            return topk

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Handle one memory access event.
        - If prefetch_hit: credit scheduler and update minimal metadata (no online training).
        - Else: treat as a miss-trigger (paper triggers on misses), compute delta from last_addr,
          quantize/encode, feed model, map top-k predictions -> addresses, filter via MRB, and issue.
        """
        if not self.initialized:
            self.init()

        addr = access.address
        issued: List[int] = []

        if prefetch_hit:
            # credit scheduler for used prefetched line
            logger.debug("prefetch_hit for addr %x", addr)
            self.scheduler.credit(addr)
            self.last_addr = addr
            return []

        # If we don't have a previous address, just record and return no prefetch
        if self.last_addr is None:
            self.last_addr = addr
            return []

        # compute delta (next - last)
        delta = addr - self.last_addr
        qdelta = self.delta_quant.quantize(delta)
        idx = self.vocab.delta_to_idx.get(qdelta, 0)

        # For simplicity use seq_len=1 (last delta); paper uses longer sequences — this is configurable.
        seq_indices = [idx]

        # Predict top-k delta indices
        try:
            topk_indices = self._predict_deltas(seq_indices)
        except Exception as e:
            logger.exception("Model prediction failed: %s", e)
            topk_indices = []

        # Map indices -> quantized deltas -> absolute addresses
        candidate_addrs: List[int] = []
        for i in topk_indices:
            try:
                pd = self.vocab.decode(i)
            except KeyError:
                continue
            # predicted next address = current addr + predicted delta
            paddr = addr + pd
            candidate_addrs.append(int(paddr))

        # Issue via scheduler (PREFETCH_DEGREE cap handled by scheduler.issue)
        issued = self.scheduler.issue(candidate_addrs, degree=CONFIG["PREFETCH_DEGREE"])

        # update last address for next delta computation
        self.last_addr = addr

        return issued

    def close(self) -> None:
        logger.info("LearnPrefetcher.close")
        self.scheduler.clear()
        self.last_addr = None
        self.initialized = False
