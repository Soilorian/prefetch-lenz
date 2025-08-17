import logging
import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # requires numpy

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.neural")


# -------------------------
# Small utilities & models
# -------------------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def linear(x: np.ndarray, W: np.ndarray, b: Optional[np.ndarray]) -> np.ndarray:
    return x.dot(W) + (b if b is not None else 0.0)


@dataclass
class SmallMLP:
    """Tiny 2-layer MLP with simple SGD step (used for both global and local models)."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    lr: float = 0.01

    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    W2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        rng = np.random.RandomState(1234)
        self.W1 = rng.randn(self.input_dim, self.hidden_dim).astype(np.float32) * (
            1.0 / math.sqrt(max(1, self.input_dim))
        )
        self.b1 = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.W2 = rng.randn(self.hidden_dim, self.output_dim).astype(np.float32) * (
            1.0 / math.sqrt(max(1, self.hidden_dim))
        )
        self.b2 = np.zeros((self.output_dim,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = relu(linear(x, self.W1, self.b1))
        out = linear(h, self.W2, self.b2)
        return out

    def step(self, x: np.ndarray, target: np.ndarray):
        """One SGD step on MSE(target, output)."""
        with self.lock:
            h_pre = linear(x, self.W1, self.b1)
            h = relu(h_pre)
            out = linear(h, self.W2, self.b2)

            diff = out - target
            grad_out = 2.0 * diff  # dLoss/dout

            grad_W2 = np.outer(h, grad_out)
            grad_b2 = grad_out

            grad_h = self.W2.dot(grad_out)
            relu_mask = (h_pre > 0).astype(np.float32)
            grad_h_pre = grad_h * relu_mask

            grad_W1 = np.outer(x, grad_h_pre)
            grad_b1 = grad_h_pre

            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1


# -------------------------
# Address encoder (embeddings)
# -------------------------
@dataclass
class AddressEncoder:
    """Map addresses <-> ids and store small trainable embeddings."""

    emb_dim: int = 32
    init_scale: float = 0.1
    max_candidates: int = 8192

    addr2id: Dict[int, int] = field(default_factory=dict)
    id2addr: Dict[int, int] = field(default_factory=dict)
    embeddings: np.ndarray = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self.embeddings = np.zeros((0, self.emb_dim), dtype=np.float32)

    def _add_address(self, addr: int) -> int:
        with self.lock:
            if addr in self.addr2id:
                return self.addr2id[addr]
            new_id = len(self.addr2id)
            if new_id >= self.max_candidates:
                slot = new_id % self.max_candidates
                old = self.id2addr.get(slot)
                if old is not None:
                    del self.addr2id[old]
                self.addr2id[addr] = slot
                self.id2addr[slot] = addr
                self.embeddings[slot] = (
                    np.random.randn(self.emb_dim).astype(np.float32) * self.init_scale
                )
                return slot
            new_emb = (
                np.random.randn(1, self.emb_dim).astype(np.float32) * self.init_scale
            )
            if self.embeddings.shape[0] == 0:
                self.embeddings = new_emb
            else:
                self.embeddings = np.vstack([self.embeddings, new_emb])
            self.addr2id[addr] = new_id
            self.id2addr[new_id] = addr
            return new_id

    def get_id(self, addr: int) -> int:
        if addr in self.addr2id:
            return self.addr2id[addr]
        return self._add_address(addr)

    def get_embedding(self, addr_or_id: Any) -> np.ndarray:
        if isinstance(addr_or_id, int) and addr_or_id in self.addr2id:
            idx = self.addr2id[addr_or_id]
        elif isinstance(addr_or_id, int) and addr_or_id in self.id2addr:
            idx = addr_or_id
        else:
            idx = self.get_id(addr_or_id)
        return self.embeddings[idx]

    def get_embeddings_matrix(self) -> np.ndarray:
        return self.embeddings

    def num_embeddings(self) -> int:
        return self.embeddings.shape[0]


# -------------------------
# Local models registry & history
# -------------------------
@dataclass
class LocalModels:
    """Registry of per-PC small MLPs (lazy creation)."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    lr: float = 0.01
    models: Dict[Any, SmallMLP] = field(default_factory=dict)
    max_locals: int = 4096

    def get(self, pc: Any) -> SmallMLP:
        if pc in self.models:
            return self.models[pc]
        if len(self.models) >= self.max_locals:
            self.models.pop(next(iter(self.models)))
        m = SmallMLP(self.input_dim, self.hidden_dim, self.output_dim, lr=self.lr)
        self.models[pc] = m
        return m


@dataclass
class HistoryUnit:
    """Per-PC recent-history buffer."""

    length: int
    _buf: Dict[Any, deque] = field(default_factory=lambda: defaultdict(deque))

    def observe(self, pc: Any, addr: int) -> None:
        dq = self._buf[pc]
        dq.append(addr)
        if len(dq) > self.length:
            dq.popleft()

    def peek(self, pc: Any) -> Tuple[int, ...]:
        dq = self._buf.get(pc)
        if not dq:
            return tuple()
        return tuple(dq)


# -------------------------
# Candidate pool (restrict scoring)
# -------------------------
@dataclass
class CandidatePool:
    max_size: int = 2048
    recent: deque = field(default_factory=deque)
    present: set = field(default_factory=set)

    def touch(self, addr: int):
        if addr in self.present:
            try:
                self.recent.remove(addr)
            except ValueError:
                pass
        else:
            self.present.add(addr)
        self.recent.append(addr)
        while len(self.recent) > self.max_size:
            old = self.recent.popleft()
            self.present.discard(old)


# -------------------------
# Neural hierarchical prefetcher (implements PrefetchAlgorithm)
# -------------------------
@dataclass
class NeuralPrefetchAlgorithm(PrefetchAlgorithm):
    """
    Neural hierarchical prefetcher implementing PrefetchAlgorithm interface.
    - online learning (global + per-PC local MLPs)
    - embedding-based scoring over recent candidate pool
    """

    emb_dim: int = 32
    hidden_dim: int = 64
    history_len: int = 4
    lr: float = 0.02
    candidate_pool_size: int = 2048
    top_k: int = 4
    tolerance: float = 0.05
    local_weight: float = 0.6
    prefetch_chain: int = 1

    # internal (initialized in init)
    encoder: AddressEncoder = field(init=False)
    global_model: SmallMLP = field(init=False)
    locals: LocalModels = field(init=False)
    history: HistoryUnit = field(init=False)
    candidates: CandidatePool = field(init=False)

    def init(self):
        """Initialize embeddings, models, history and candidate pool."""
        self.encoder = AddressEncoder(
            emb_dim=self.emb_dim, max_candidates=max(4096, self.candidate_pool_size * 2)
        )
        input_dim = self.history_len * self.emb_dim
        # global model
        self.global_model = SmallMLP(
            input_dim, self.hidden_dim, self.emb_dim, lr=self.lr
        )
        # local models registry (smaller hidden dim)
        local_hidden = max(8, self.hidden_dim // 4)
        self.locals = LocalModels(input_dim, local_hidden, self.emb_dim, lr=self.lr)
        self.history = HistoryUnit(length=self.history_len)
        self.candidates = CandidatePool(max_size=self.candidate_pool_size)
        logger.info(
            "NeuralPrefetchAlgorithm initialized: emb_dim=%d hidden=%d hist_len=%d",
            self.emb_dim,
            self.hidden_dim,
            self.history_len,
        )

    # -- small helpers --
    def _history_to_input(self, hist: Tuple[int, ...]) -> np.ndarray:
        vects = []
        pad_needed = self.history_len - len(hist)
        if pad_needed > 0:
            for _ in range(pad_needed):
                vects.append(np.zeros((self.emb_dim,), dtype=np.float32))
        for a in hist:
            emb = self.encoder.get_embedding(a)
            vects.append(emb)
        return np.concatenate(vects).astype(np.float32)

    def _predict_embedding(self, pc: Any, hist: Tuple[int, ...]) -> np.ndarray:
        x = self._history_to_input(hist)
        g = self.global_model.forward(x)
        l = self.locals.get(pc).forward(x)
        return self.local_weight * l + (1.0 - self.local_weight) * g

    def _score_candidates(self, pred_emb: np.ndarray) -> List[Tuple[int, float]]:
        emb_mat = self.encoder.get_embeddings_matrix()
        if emb_mat.shape[0] == 0:
            return []
        scores = emb_mat.dot(pred_emb)
        cand_list = list(self.candidates.recent)
        scored = []
        for addr in cand_list:
            aid = self.encoder.get_id(addr)
            s = float(scores[aid]) if aid < scores.shape[0] else 0.0
            scored.append((addr, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _select_top(self, scored: List[Tuple[int, float]]) -> List[int]:
        if not scored:
            return []
        best = scored[0][1]
        cutoff = best - abs(best) * self.tolerance
        result = []
        for addr, sc in scored:
            if sc >= cutoff:
                result.append(addr)
            if len(result) >= self.top_k:
                break
        return result

    def _extract(self, access: MemoryAccess) -> Tuple[int, int]:
        pc = getattr(access, "pc", None)
        if pc is None:
            pc = getattr(access, "PC", None)
        addr = getattr(access, "address", None)
        if addr is None:
            addr = getattr(access, "addr", None)
        if pc is None or addr is None:
            if isinstance(access, (tuple, list)) and len(access) >= 2:
                pc, addr = access[0], access[1]
        if pc is None or addr is None:
            raise ValueError("MemoryAccess missing pc/address")
        return int(pc), int(addr)

    # -- core API --
    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process a memory access and return predicted addresses to prefetch (List[int]).
        Updates models online (previous-history -> current address).
        """
        pc, addr = self._extract(access)

        # register address and candidate pool
        self.encoder.get_id(addr)
        self.candidates.touch(addr)

        # training pair: prev_hist -> addr
        prev_hist = self.history.peek(pc)
        if prev_hist:
            x_prev = self._history_to_input(prev_hist)
            target_emb = self.encoder.get_embedding(addr)
            # SGD step: global + local
            self.global_model.step(x_prev, target_emb)
            self.locals.get(pc).step(x_prev, target_emb)

        # optional extra boost on prefetch_hit (small extra step)
        if prefetch_hit and prev_hist:
            x_prev = self._history_to_input(prev_hist)
            target_emb = self.encoder.get_embedding(addr)
            old_lr_g = self.global_model.lr
            old_lr_l = self.locals.get(pc).lr
            self.global_model.lr = max(1e-6, self.global_model.lr * 1.5)
            self.global_model.step(x_prev, target_emb)
            self.global_model.lr = old_lr_g
            lm = self.locals.get(pc)
            lm.lr = max(1e-6, lm.lr * 1.5)
            lm.step(x_prev, target_emb)
            lm.lr = old_lr_l

        # observe address for future predictions
        self.history.observe(pc, addr)

        # predict for next access
        cur_hist = self.history.peek(pc)
        pred_emb = self._predict_embedding(pc, cur_hist)
        scored = self._score_candidates(pred_emb)
        top_addrs = self._select_top(scored)

        # optional chaining
        if self.prefetch_chain > 1 and top_addrs:
            preds = list(top_addrs)
            simulated = list(cur_hist)
            used = set(preds)
            hops = self.prefetch_chain - 1
            for _ in range(hops):
                simulated.append(preds[-1])
                if len(simulated) > self.history_len:
                    simulated = simulated[-self.history_len :]
                sim_emb = self._predict_embedding(pc, tuple(simulated))
                sim_scored = self._score_candidates(sim_emb)
                sim_top = self._select_top(sim_scored)
                if not sim_top:
                    break
                pick = sim_top[0]
                if pick in used:
                    break
                preds.append(pick)
                used.add(pick)
                if len(preds) >= self.top_k:
                    break
            return preds[: self.top_k]
        else:
            return top_addrs[: self.top_k]

    def close(self):
        """Cleanup (no-op for in-memory predictor)."""
        logger.info("NeuralPrefetchAlgorithm closed")
        return
