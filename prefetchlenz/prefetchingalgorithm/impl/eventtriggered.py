from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from prefetchlenz.prefetchingalgorithm.access.eventtriggeredmemoryaccess import (
    EventTriggeredMemoryAccess,
)
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.event_triggered")


# ----------------------------- EWMA / Metrics ---------------------------------


class Metric(Enum):
    """Supported EWMA metrics used as rule operands."""

    STRIDE = auto()  # Δaddr between consecutive events of a stream
    LATENCY = auto()  # demand load latency (cycles or ns)
    INTER_ARRIVAL = auto()  # time between consecutive events
    PRESSURE = auto()  # proxy for queue occupancy / outstanding reqs


@dataclass
class EwmaState:
    """Holds one EWMA register for a metric."""

    value: float = 0.0
    initialized: bool = False


def _key(access: EventTriggeredMemoryAccess, metric: Metric) -> Tuple[int, Metric]:
    return int(access.pc), metric


class EwmaCalculator:
    """
    General-purpose EWMA bank.

    - Maintains EWMA per (stream_key, metric).
    - Each entry stores a register E and uses α (0 < α ≤ 1).
    - Update rule: E <- α·X + (1-α)·E

    Stream key:
      By default we use the instruction PC of the event (access.pc).
      You can switch to any key extractor (e.g., filter ID, region).
    """

    def __init__(self, alpha: float = 0.3):
        assert 0.0 < alpha <= 1.0
        self.alpha = alpha
        self._bank: Dict[Tuple[int, Metric], EwmaState] = defaultdict(EwmaState)
        logger.info("EWMA: initialized alpha=%.3f", self.alpha)

    def clear(self):
        self._bank.clear()

    def update_sample(
        self, access: EventTriggeredMemoryAccess, metric: Metric, sample: float
    ) -> None:
        """
        Ingest one sample X for a metric of this stream.
        """
        k = (access.pc, metric)
        st = self._bank[k]
        if not st.initialized:
            st.value = float(sample)
            st.initialized = True
        else:
            st.value = self.alpha * float(sample) + (1.0 - self.alpha) * st.value
        logger.debug(
            "EWMA: pc=%#x metric=%s sample=%.3f -> E=%.3f",
            access.pc,
            metric.name,
            sample,
            st.value,
        )

    def get(
        self, access: EventTriggeredMemoryAccess, metric: Metric
    ) -> Optional[float]:
        """Return current EWMA value for stream metric, if initialized."""
        st = self._bank.get((access.pc, metric))
        return st.value if (st and st.initialized) else None

    def get_parameters(
        self, access: EventTriggeredMemoryAccess
    ) -> Dict[Metric, Optional[float]]:
        """
        Convenience: return a dict of all metrics for this stream.
        Missing metrics yield None.
        """
        return {m: self.get(access, m) for m in Metric}


# ------------------------------- Filters --------------------------------------


@dataclass(frozen=True)
class FilterSpec:
    """Programmable software filter spec."""

    pcs: Optional[Tuple[int, ...]] = None  # trigger PCs
    addr_ranges: Optional[Tuple[Tuple[int, int], ...]] = None  # [(lo, hi), ...]


class Filter:
    """
    Software filter. Passes an event if it matches the spec.
    """

    def __init__(self, spec: FilterSpec):
        self.spec = spec

    def pass_filter(self, access: EventTriggeredMemoryAccess) -> bool:
        pc_ok = True if not self.spec.pcs else int(access.pc) in self.spec.pcs
        addr_ok = True
        if self.spec.addr_ranges:
            a = int(access.address)
            addr_ok = any(lo <= a < hi for (lo, hi) in self.spec.addr_ranges)
        ok = pc_ok and addr_ok
        logger.debug(
            "Filter: pc=%#x addr=%#x -> %s",
            access.pc,
            access.address,
            "PASS" if ok else "BLOCK",
        )
        return ok


class AddrFilter:
    """
    Factory and holder for active filters. In a full system there may be many rules,
    each with its own filter.
    """

    def __init__(self, specs: Optional[List[FilterSpec]] = None):
        self.filters: List[Filter] = [Filter(s) for s in (specs or [])]

    def add_filter(self, spec: FilterSpec) -> None:
        self.filters.append(Filter(spec))

    def select(self, access: EventTriggeredMemoryAccess) -> List[Filter]:
        """Return all filters that pass this access."""
        return [f for f in self.filters if f.pass_filter(access)]


# --------------------------- Observation Queue --------------------------------


@dataclass
class ObsRecord:
    """
    One observation entry kept in history.
    """

    ts: float
    pc: int
    addr: int
    stride: int
    latency: Optional[int]


class ObservationQueue:
    """
    Per-stream bounded history used by rules.

    Structure:
      dict[stream_key] -> deque[ObsRecord] capped at capacity.

    Provides:
      - update(access, latency): computes stride and stores a record
      - history(access, k): returns the most recent k records for the stream
    """

    def __init__(self, capacity_per_stream: int = 16):
        self.capacity = max(1, capacity_per_stream)
        self._hist: Dict[int, Deque[ObsRecord]] = defaultdict(
            lambda: deque(maxlen=self.capacity)
        )
        self._last_addr: Dict[int, int] = {}
        self._last_ts: Dict[int, float] = {}

    def clear(self):
        self._hist.clear()
        self._last_addr.clear()
        self._last_ts.clear()

    def update(
        self, access: EventTriggeredMemoryAccess, latency: Optional[int]
    ) -> Tuple[int, Optional[float]]:
        pc = access.pc
        now = time.time()
        addr = access.address

        prev_addr = self._last_addr.get(pc)
        stride = (addr - prev_addr) if prev_addr is not None else 0

        prev_ts = self._last_ts.get(pc)
        inter_arrival = (now - prev_ts) if prev_ts is not None else None

        rec = ObsRecord(ts=now, pc=pc, addr=addr, stride=stride, latency=latency)
        self._hist[pc].append(rec)
        self._last_addr[pc] = addr
        self._last_ts[pc] = now

        logger.debug(
            "ObsQ: pc=%#x addr=%#x stride=%d lat=%s",
            pc,
            addr,
            stride,
            str(latency) if latency is not None else "NA",
        )
        return stride, inter_arrival

    def get_history(
        self, access: EventTriggeredMemoryAccess, k: int = 4
    ) -> List[ObsRecord]:
        pc = access.pc
        dq = self._hist.get(pc, deque())
        return list(dq)[-k:]


# -------------------------------- Scheduler -----------------------------------


class Scheduler:
    """
    Minimal scheduler with a simple rate limiter.

    Blocks prefetch issue when the EWMA PRESSURE metric exceeds a threshold.
    """

    def __init__(self, pressure_threshold: float = 8.0):
        self.pressure_threshold = pressure_threshold

    def allow(self, ewma_params: Dict[Metric, Optional[float]]) -> bool:
        pr = ewma_params.get(Metric.PRESSURE)
        ok = True if pr is None else pr < self.pressure_threshold
        logger.debug(
            "Sched: pressure=%s threshold=%.1f -> %s",
            "NA" if pr is None else f"{pr:.2f}",
            self.pressure_threshold,
            "ALLOW" if ok else "BLOCK",
        )
        return ok


# ------------------------ Programmable Prefetch Unit ---------------------------


class ProgrammablePrefetchUnit:
    """
    Executes programmable rules.

    Example rules implemented:
      1) Stride rule: if EWMA(STRIDE) exists, prefetch addr + N*stride for N in [1..degree]
      2) Latency gate: only if EWMA(LATENCY) ≥ latency_min
    """

    def __init__(self, degree: int = 2, latency_min: Optional[float] = None):
        self.degree = max(1, degree)
        self.latency_min = latency_min

    def prefetch(
        self,
        access: EventTriggeredMemoryAccess,
        ewma_parameters: Dict[Metric, Optional[float]],
        history: List[ObsRecord],
    ) -> List[int]:
        preds: List[int] = []

        # Optional latency gate
        lat = ewma_parameters.get(Metric.LATENCY)
        if self.latency_min is not None and (lat is None or lat < self.latency_min):
            logger.debug(
                "PPU: latency gate not met (lat=%s, min=%s)",
                str(lat),
                str(self.latency_min),
            )
            return preds

        # Stride rule
        stride = ewma_parameters.get(Metric.STRIDE)
        if stride is not None and stride != 0:
            base = int(access.address)
            for n in range(1, self.degree + 1):
                preds.append(base + int(round(n * stride)))

        logger.debug(
            "PPU: pc=%#x base=%#x stride=%s degree=%d -> preds=%d",
            access.pc,
            access.address,
            "NA" if stride is None else f"{stride:.2f}",
            self.degree,
            len(preds),
        )
        return preds


# -------------------------- Event-Triggered Prefetcher ------------------------


class EventTriggeredPrefetcher(PrefetchAlgorithm):
    """
    Event-Triggered Programmable Prefetcher.

    Pipeline per read-event:
      1) AddrFilter selects matching events.
      2) ObservationQueue updates history (stride, inter-arrival) and provides k-latest.
      3) EWMACalculator ingests samples for STRIDE, INTER_ARRIVAL, LATENCY, PRESSURE.
      4) Scheduler gates issue based on PRESSURE.
      5) PPU executes rules and yields prefetch addresses.

    Notes:
      - This is a software model. Latency and pressure samples must be supplied
        by the driver (e.g., from simulator stats).
    """

    def __init__(self):
        self.filters = AddrFilter()
        self.obsq = ObservationQueue(capacity_per_stream=16)
        self.ewma = EwmaCalculator(alpha=0.3)
        self.sched = Scheduler(pressure_threshold=8.0)
        self.ppu = ProgrammablePrefetchUnit(degree=2, latency_min=None)

        # External hooks the driver can set:
        self.sample_latency: Optional[int] = None  # last observed latency sample
        self.sample_pressure: Optional[float] = None  # last observed pressure sample

    def init(self):
        self.obsq.clear()
        self.ewma.clear()
        logger.info("ETP initialized")

    def close(self):
        self.obsq.clear()
        self.ewma.clear()
        logger.info("ETP closed")

    def progress(
        self, access: EventTriggeredMemoryAccess, prefetch_hit: bool
    ) -> List[int]:
        """
        Process one event. Return prefetch addresses.

        The caller should set:
          - self.sample_latency with the demand latency for this access (if available)
          - self.sample_pressure with current system pressure (if available)
        """
        # 1) Filter
        passing = self.filters.select(access)
        if not passing:
            return []

        # 2) Observation update
        stride, inter_arrival = self.obsq.update(access, latency=access.accessLatency)

        # 3) EWMA updates
        if stride is not None:
            self.ewma.update_sample(access, Metric.STRIDE, float(stride))
        if inter_arrival is not None:
            self.ewma.update_sample(access, Metric.INTER_ARRIVAL, float(inter_arrival))
        if access.accessLatency is not None:
            self.ewma.update_sample(access, Metric.LATENCY, float(access.accessLatency))
        if access.bandwidthPressure is not None:
            self.ewma.update_sample(
                access, Metric.PRESSURE, float(access.bandwidthPressure)
            )

        ewma_params = self.ewma.get_parameters(access)
        hist = self.obsq.get_history(access, k=4)

        # 4) Schedule gate
        if not self.sched.allow(ewma_params):
            return []

        # 5) PPU rules
        preds = self.ppu.prefetch(access, ewma_params, hist)
        logger.debug(
            "ETP: pc=%#x addr=%#x -> issued=%d", access.pc, access.address, len(preds)
        )
        return preds
