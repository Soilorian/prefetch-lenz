import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, FrozenSet, Iterable, List, Optional, Tuple

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.hds")

# Assumed interface:
# class MemoryAccess:
#     pc: int
#     address: int

Symbol = Tuple[int, int]  # (pc, addr)
StateElem = Tuple[int, int]  # (stream_id, seen) where 0 <= seen < headLen
State = FrozenSet[StateElem]


@dataclass(frozen=True)
class HotStream:
    """A hot data stream v = [(pc,addr), ...] with prefix length headLen."""

    id: int
    seq: List[Symbol]

    @property
    def head(self) -> List[Symbol]:
        return self.seq[: self.headLen]  # type: ignore[attr-defined]

    @property
    def tail(self) -> List[Symbol]:
        return self.seq[self.headLen :]  # type: ignore[attr-defined]


class BurstyController:
    """
    Implements the paper's bursty tracing counters:
      - nCheck0 (mostly uninstrumented)
      - nInstr0 (instrumented burst length)
      - nAwake0 burst-periods awake, then nHibernate0 periods hibernating

    We expose:
      - should_profile(): whether to record references this step
      - tick(): advance one 'check' point (we treat each MemoryAccess as a check)
    """

    def __init__(self, nCheck0: int, nInstr0: int, nAwake0: int, nHibernate0: int):
        assert nCheck0 >= 1 and nInstr0 >= 1
        self.nCheck0 = nCheck0
        self.nInstr0 = nInstr0
        self.nAwake0 = nAwake0
        self.nHibernate0 = nHibernate0

        self._phase_awake = True
        self._checks_left = nCheck0
        self._instr_left = 0
        self._period_left = nAwake0

    @property
    def is_awake(self) -> bool:
        return self._phase_awake

    def _flip_phase(self):
        self._phase_awake = not self._phase_awake
        self._period_left = self.nAwake0 if self._phase_awake else self.nHibernate0
        # Reset counters for a new period
        self._checks_left = (
            self.nCheck0 if self._phase_awake else (self.nCheck0 + self.nInstr0 - 1)
        )
        self._instr_left = self.nInstr0 if self._phase_awake else 1

    def should_profile(self) -> bool:
        return self._phase_awake and self._instr_left > 0

    def tick(self):
        # One "check" passed
        if self._phase_awake:
            if self._instr_left > 0:
                self._instr_left -= 1
                if self._instr_left == 0:
                    # exit burst; continue checks until period ends
                    pass
            else:
                # checking-only portion
                pass
        else:
            # hibernating: 1-instruction 'instrumented' per period, effectively negligible
            if self._instr_left > 0:
                self._instr_left -= 1

        self._checks_left -= 1
        if self._checks_left == 0:
            # completed a burst-period
            self._period_left -= 1
            if self._period_left == 0:
                self._flip_phase()
            else:
                # start next period within same phase
                if self._phase_awake:
                    self._checks_left = self.nCheck0
                    self._instr_left = self.nInstr0
                else:
                    self._checks_left = self.nCheck0 + self.nInstr0 - 1
                    self._instr_left = 1


class StreamMiner:
    """
    Fast(ish) online approximation for hot stream discovery.
    We approximate the paper’s Sequitur-based detection with a bounded n-gram scan
    that counts non-overlapping occurrences and scores v by heat = len(v) * frequency.

    Controls:
      - min_len, max_len
      - heat_threshold H
      - min_unique (to exclude trivially repeating same-symbol runs)
      - max_profile_buffer: max references kept in one awake window
    """

    def __init__(
        self,
        min_len: int = 11,  # paper used >10
        max_len: int = 64,
        heat_threshold: int = 8,  # tunable
        min_unique: int = 2,
        max_profile_buffer: int = 50_000,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.heat_threshold = heat_threshold
        self.min_unique = min_unique
        self.max_profile_buffer = max_profile_buffer
        self.buffer: List[Symbol] = []

    def feed(self, sym: Symbol):
        if len(self.buffer) < self.max_profile_buffer:
            self.buffer.append(sym)

    def _nonoverlap_counts(
        self, seq: List[Symbol], L: int
    ) -> Dict[Tuple[Symbol, ...], int]:
        counts: Dict[Tuple[Symbol, ...], int] = defaultdict(int)
        i = 0
        n = len(seq)
        while i + L <= n:
            s = tuple(seq[i : i + L])
            counts[s] += 1
            i += L  # non-overlapping by construction for this pass
        return counts

    def analyze(self) -> List[List[Symbol]]:
        if not self.buffer:
            return []
        logger.info("HDS: analyzing %d profiled references", len(self.buffer))
        # Multi-length scan, accumulate best candidates by heat
        best: Dict[Tuple[Symbol, ...], int] = {}
        # A cheap heuristic: sample a stride of 1 and also a staggered start to capture shifts
        for start in (0, 1, 2):
            seq = self.buffer[start:]
            for L in range(self.min_len, min(self.max_len, len(seq)) + 1):
                counts = self._nonoverlap_counts(seq, L)
                for s, freq in counts.items():
                    if freq <= 1:
                        continue
                    if len(set(s)) < self.min_unique:
                        continue
                    heat = L * freq
                    if heat >= self.heat_threshold:
                        prev = best.get(s, 0)
                        if heat > prev:
                            best[s] = heat

        # De-subsumption: if a longer stream’s occurrences fully cover a shorter one, prefer longer
        # Simple filter: drop any stream that is a contiguous substring of another chosen stream.
        streams = sorted(best.items(), key=lambda x: (-x[1], -len(x[0])))
        chosen: List[Tuple[Tuple[Symbol, ...], int]] = []
        for s, heat in streams:
            if any(self._is_subsequence(s, t[0]) for t in chosen):
                continue
            chosen.append((s, heat))

        result = [list(s) for (s, _) in chosen]
        logger.info("HDS: selected %d hot streams", len(result))
        return result

    @staticmethod
    def _is_subsequence(a: Tuple[Symbol, ...], b: Tuple[Symbol, ...]) -> bool:
        # is 'a' a contiguous subsequence of 'b'?
        if len(a) > len(b):
            return False
        for i in range(len(b) - len(a) + 1):
            if b[i : i + len(a)] == list(a):
                return True
        return False

    def reset(self):
        self.buffer.clear()


class DFSM:
    """
    Single DFSM that matches prefixes of ALL hot streams simultaneously, as in the paper.
    States are frozensets of (stream_id, seen), where seen in [0, headLen-1].
    Transition on a Symbol -> next State, possibly with prefetch payloads of the matched streams’ tails.
    """

    def __init__(self, hot_streams: List[HotStream], headLen: int):
        self.headLen = headLen
        self.hot_streams = hot_streams
        for hs in self.hot_streams:
            object.__setattr__(hs, "headLen", headLen)  # attach for convenience

        # Build alphabet only from head symbols
        self.alphabet: List[Symbol] = sorted(
            {
                hs.head[i]
                for hs in hot_streams
                for i in range(min(len(hs.head), headLen))
            }
        )
        self.start: State = frozenset()
        self.transitions: Dict[Tuple[State, Symbol], State] = {}
        self.prefetches: Dict[State, List[List[Symbol]]] = (
            {}
        )  # state -> list of tails to prefetch

        self._build()

    def _build(self):
        logger.info(
            "HDS: building DFSM for %d streams (headLen=%d)",
            len(self.hot_streams),
            self.headLen,
        )
        work: Deque[State] = deque([self.start])
        seen_states = {self.start}

        def advance(state: State, a: Symbol) -> State:
            # d(s, a) = {[v, n+1] | [v, n] in s and a == v[n+1]} ∪ {[w,1] | a == w[1]}
            next_elems: List[StateElem] = []
            # advance partials
            for sid, n in state:
                hs = self.hot_streams[sid]
                if n < self.headLen and n < len(hs.head) and hs.head[n] == a:
                    next_elems.append((sid, n + 1))
            # start new matches
            for hs in self.hot_streams:
                if len(hs.head) > 0 and hs.head[0] == a:
                    next_elems.append((hs.id, 1))
            return frozenset(next_elems)

        while work:
            s = work.popleft()
            for a in self.alphabet:
                ns = advance(s, a)
                if ns and (s, a) not in self.transitions:
                    self.transitions[(s, a)] = ns
                    if ns not in seen_states:
                        seen_states.add(ns)
                        work.append(ns)

        # Annotate prefetch payloads for any state that completes a head
        for st in seen_states:
            payloads: List[List[Symbol]] = []
            for sid, n in st:
                if n >= self.headLen:
                    hs = self.hot_streams[sid]
                    if len(hs.tail) > 0:
                        payloads.append(hs.tail)
            if payloads:
                self.prefetches[st] = payloads

        logger.info(
            "HDS: DFSM has %d states, %d transitions, %d prefetching states",
            len(seen_states),
            len(self.transitions),
            len(self.prefetches),
        )

    def step(
        self, state: State, sym: Symbol
    ) -> Tuple[State, Optional[List[List[Symbol]]]]:
        ns = self.transitions.get((state, sym), frozenset())
        payload = self.prefetches.get(ns)
        return ns, payload


class HdsPrefetcher:
    """
    Dynamic Hot Data Stream Prefetcher (software), implementing:
      - Bursty profiling (awake/hibernate)
      - Stream mining (approximate heat-based selection)
      - Single DFSM prefix matcher
      - Prefetch tail issue on match completion

    Integration points:
      - progress(access, prefetch_hit): call per MemoryAccess
      - set_prefetch_callback(cb): cb(address:int) called once per tail symbol (dedup per access)
      - init() / close(): lifecycle
    """

    def __init__(
        self,
        headLen: int = 2,
        min_len: int = 11,
        max_len: int = 64,
        heat_threshold: int = 12,
        min_unique: int = 2,
        nCheck0: int = 11_940,
        nInstr0: int = 60,
        nAwake0: int = 50,
        nHibernate0: int = 2_450,
        max_hot_streams: int = 64,
    ):
        self.headLen = headLen
        self.controller = BurstyController(nCheck0, nInstr0, nAwake0, nHibernate0)
        self.miner = StreamMiner(min_len, max_len, heat_threshold, min_unique)
        self.dfsm: Optional[DFSM] = None
        self.state: State = frozenset()
        self.prefetch_cb = lambda addr: None
        self.max_hot_streams = max_hot_streams
        # Stats
        self._prefetches_issued = 0
        self._matches = 0

    def set_prefetch_callback(self, cb):
        self.prefetch_cb = cb

    def init(self):
        logger.info("HDS: init (headLen=%d)", self.headLen)
        self.controller = self.controller  # no-op; keep counters
        self.miner.reset()
        self.dfsm = None
        self.state = frozenset()
        self._prefetches_issued = 0
        self._matches = 0

    def _symbolize(self, access) -> Symbol:
        return (int(access.pc), int(access.key))

    def _rebuild_dfsm(self, streams: List[List[Symbol]]):
        # Cap number of hot streams, rank by length descending as a cheap proxy for coverage.
        ranked = sorted(streams, key=lambda s: (-len(s), s[0] if s else (0, 0)))[
            : self.max_hot_streams
        ]
        hot = [HotStream(id=i, seq=list(s)) for i, s in enumerate(ranked)]
        self.dfsm = DFSM(hot, self.headLen)
        self.state = frozenset()

    def _maybe_profile(self, sym: Symbol):
        if self.controller.should_profile():
            self.miner.feed(sym)

    def _maybe_analyze_and_optimize(self):
        # Trigger analysis exactly when we switch from awake->hibernate (period boundary) is handled by controller.
        # Here we simply rebuild when DFSM is None and we have data, or at any awake-end implied by buffer fullness.
        if self.dfsm is None and len(self.miner.buffer) > 0:
            streams = self.miner.analyze()
            if streams:
                self._rebuild_dfsm(streams)
            self.miner.reset()

    def _prefetch_payloads(self, payloads: Iterable[List[Symbol]]):
        seen_block: set = set()
        for seq in payloads:
            for _, addr in seq:
                if addr not in seen_block:
                    self.prefetch_cb(addr)
                    seen_block.add(addr)
                    self._prefetches_issued += 1

    def progress(self, access, prefetch_hit: bool = False):
        """
        Call on each MemoryAccess.
        """
        sym = self._symbolize(access)
        # 1) profiling
        self._maybe_profile(sym)

        # 2) matching/prefetching (during both phases; the paper keeps checks live in hibernation)
        if self.dfsm is not None:
            ns, payloads = self.dfsm.step(self.state, sym)
            if ns != self.state:
                # state advanced or reset
                self.state = ns
            if payloads:
                self._matches += 1
                self._prefetch_payloads(payloads)

        # 3) advance bursty counters and possibly rebuild DFSM when transitioning to hibernation
        self.controller.tick()

        # Rebuild DFSM when (a) we have data and (b) current DFSM is missing or stale.
        if self.controller.is_awake is False:
            # We just entered/are in hibernation: if we profiled anything recently and have no DFSM, build it.
            self._maybe_analyze_and_optimize()

        # Optional bookkeeping on prefetch hits (to adapt policies later)
        if prefetch_hit:
            # You can add adaptive tweaks here (e.g., raise headLen or heat threshold on too many false positives)
            pass

    def close(self):
        logger.info(
            "HDS: close — matches=%d, prefetches_issued=%d",
            self._matches,
            self._prefetches_issued,
        )
        self.dfsm = None
        self.state = frozenset()
