# hierarchical_prefetcher.py
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional

from prefetchlenz.prefetchingalgorithm.impl.markovpredictor import MemoryAccess

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.hierarchical")
logger.addHandler(logging.NullHandler())


class DeltaVocab:
    """
    Manage mapping delta -> id and id -> delta with a cap.
    New deltas are added until capacity; old ones remain (no eviction) for simplicity.
    """

    def __init__(self, cap: int = 1024):
        self.cap = int(cap)
        self.delta_to_id: Dict[int, int] = {}
        self.id_to_delta: List[int] = []

    def ensure(self, delta: int) -> Optional[int]:
        if delta in self.delta_to_id:
            return self.delta_to_id[delta]
        if len(self.id_to_delta) >= self.cap:
            # capacity reached; don't add new deltas
            return None
        idx = len(self.id_to_delta)
        self.delta_to_id[delta] = idx
        self.id_to_delta.append(delta)
        logger.debug("DeltaVocab: added delta=%d id=%d", delta, idx)
        return idx

    def get_id(self, delta: int) -> Optional[int]:
        return self.delta_to_id.get(delta)

    def get_delta(self, idx: int) -> int:
        return self.id_to_delta[idx]


class LocalModelBank:
    """
    Per-PC lightweight linear scorer over delta ids.
    Implemented as a dict: pc -> array of scores (sparse via dict).
    Updated online by incrementing the true delta score and decaying others slightly.
    """

    def __init__(self, vocab: DeltaVocab, decay: float = 0.99, learn_rate: float = 1.0):
        self.vocab = vocab
        self.decay = float(decay)
        self.lr = float(learn_rate)
        # pc -> dict(delta_id -> score)
        self.bank: Dict[int, Dict[int, float]] = defaultdict(dict)

    def score(self, pc: int) -> Dict[int, float]:
        return self.bank.get(pc, {})

    def update(self, pc: int, true_delta_id: Optional[int]):
        m = self.bank[pc]
        # decay all entries
        for k in list(m.keys()):
            m[k] *= self.decay
            # small pruning to keep it tidy
            if abs(m[k]) < 1e-6:
                del m[k]
        # boost true delta
        if true_delta_id is not None:
            m[true_delta_id] = m.get(true_delta_id, 0.0) + self.lr
        logger.debug(
            "LocalModel: pc=0x%X updated true_delta_id=%s top=%s",
            pc,
            str(true_delta_id),
            sorted(m.items(), key=lambda kv: -kv[1])[:3],
        )


class GlobalModel:
    """
    Global aggregator over delta ids. Similar to LocalModel but single instance.
    """

    def __init__(
        self, vocab: DeltaVocab, decay: float = 0.995, learn_rate: float = 1.0
    ):
        self.vocab = vocab
        self.decay = float(decay)
        self.lr = float(learn_rate)
        self.scores: Dict[int, float] = {}

    def score(self) -> Dict[int, float]:
        return self.scores

    def update(self, true_delta_id: Optional[int]):
        for k in list(self.scores.keys()):
            self.scores[k] *= self.decay
            if abs(self.scores[k]) < 1e-6:
                del self.scores[k]
        if true_delta_id is not None:
            self.scores[true_delta_id] = self.scores.get(true_delta_id, 0.0) + self.lr
        logger.debug(
            "GlobalModel: updated true_delta_id=%s top=%s",
            str(true_delta_id),
            sorted(self.scores.items(), key=lambda kv: -kv[1])[:3],
        )


def softmax_scores(score_map: Dict[int, float]) -> Dict[int, float]:
    if not score_map:
        return {}
    vals = list(score_map.values())
    mx = max(vals)
    exps = {k: math.exp(v - mx) for k, v in score_map.items()}
    s = sum(exps.values())
    if s == 0:
        return {k: 0.0 for k in exps}
    return {k: v / s for k, v in exps.items()}


class HierarchicalNeuralPrefetcher:
    """
    Lightweight, testable approximation of the "Hierarchical Neural Model of Data Prefetching".
    This is a small, online learner with two levels:
      - local per-PC scorer
      - global aggregator
    It models deltas (addr differences) as the prediction target.

    Parameters:
      - vocab_cap: max distinct deltas tracked
      - combine_alpha: weight given to local model when combining (0.0..1.0)
      - top_k: how many predicted deltas to turn into prefetch addresses
    """

    def __init__(
        self, vocab_cap: int = 1024, combine_alpha: float = 0.7, top_k: int = 2
    ):
        self.vocab = DeltaVocab(cap=vocab_cap)
        self.local_bank = LocalModelBank(self.vocab)
        self.global_model = GlobalModel(self.vocab)
        self.combine_alpha = float(combine_alpha)
        self.top_k = int(top_k)

        # last seen address per PC and global last address for delta computation
        self.last_addr_per_pc: Dict[int, int] = {}
        self.last_addr_global: Optional[int] = None

    def init(self):
        """Reset internal state."""
        self.vocab = DeltaVocab(cap=self.vocab.cap)
        self.local_bank = LocalModelBank(self.vocab)
        self.global_model = GlobalModel(self.vocab)
        self.last_addr_per_pc.clear()
        self.last_addr_global = None
        logger.info(
            "HierarchicalNeuralPrefetcher: initialized vocab_cap=%d", self.vocab.cap
        )

    def close(self):
        logger.info("HierarchicalNeuralPrefetcher: closed")

    def _compute_delta(self, addr: int, pc: int) -> Optional[int]:
        """
        Compute delta = addr - last_addr_for_pc if available, else use global last.
        Return delta id (add to vocab if new and room).
        """
        ref = self.last_addr_per_pc.get(pc, self.last_addr_global)
        if ref is None:
            return None
        delta = addr - ref
        return self.vocab.ensure(delta)

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """
        Process one access. Returns list of predicted addresses to prefetch.
        Steps:
          1. compute delta id for this access (based on pc-local or global history)
          2. predict next deltas using combined local+global scores (curr_state)
          3. update models with the observed true delta (prev_state -> curr)
          4. update last-address trackers
        """
        pc = int(access.pc)
        addr = int(access.address)

        # 1) compute delta of current access relative to previous ref
        curr_delta_id = self._compute_delta(addr, pc)

        # 2) Prediction: use current address (curr state) to predict next delta
        # Combine local(pc) and global scores
        local_scores = self.local_bank.score(pc)
        global_scores = self.global_model.score()
        # For prediction we combine normalized probabilities
        local_prob = softmax_scores(local_scores)
        global_prob = softmax_scores(global_scores)

        combined: Dict[int, float] = {}
        # union of keys
        for k in set(local_prob.keys()).union(global_prob.keys()):
            lp = local_prob.get(k, 0.0)
            gp = global_prob.get(k, 0.0)
            combined[k] = self.combine_alpha * lp + (1.0 - self.combine_alpha) * gp

        preds: List[int] = []
        if combined:
            # pick top-k by combined score
            items = sorted(combined.items(), key=lambda kv: -kv[1])
            for idx, score in items[: self.top_k]:
                delta = self.vocab.get_delta(idx)
                tgt = addr + delta
                preds.append(tgt)

        # 3) Update models using observed delta (supervised signal).
        # The update uses curr_delta_id as the "true" delta observed at this step.
        # We interpret update as: prev->curr, but our simple model only accumulates curr as evidence.
        if curr_delta_id is not None:
            # update global and local models to reinforce this delta for this PC
            self.global_model.update(curr_delta_id)
            self.local_bank.update(pc, curr_delta_id)

        # 4) update last-address trackers
        self.last_addr_per_pc[pc] = addr
        self.last_addr_global = addr

        if preds:
            logger.debug(
                "Prefetcher: pc=0x%X addr=0x%X preds=%s",
                pc,
                addr,
                [hex(p) for p in preds],
            )
        return preds
