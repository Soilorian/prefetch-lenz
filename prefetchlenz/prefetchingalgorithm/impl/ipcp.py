"""
IPCP: Instruction Pointer Classifier–based Prefetcher
(see ISCA’20 paper "Instruction Pointer Classifier-based Spatial Prefetching")

Implements IPCP inside the PrefetchAlgorithm framework:
 - Uses IPTable to classify PCs into classes
 - Sub-prefetchers per class: CS, GS, CPLX
 - Scheduler caps outstanding and avoids duplicates
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess

logger = logging.getLogger("prefetchlenz.prefetchingalgorithm.impl.ipcp")

# -------------------------
# Config parameters
# -------------------------
CONFIG = {
    "REGION_SIZE": 2 * 1024,  # bytes
    "BLOCK_SIZE": 64,  # bytes
    "BLOCKS_PER_REGION": (2 * 1024) // 64,
    "IPTABLE_ENTRIES": 64,  # direct-mapped IP table
    "COUNTER_BITS": 2,
    "PREFETCH_DEGREE": 4,
    "MAX_OUTSTANDING": 16,
    "INIT_CONF": 2,  # init counter for observed
    "CONF_MAX": (1 << 2) - 1,
    "CONF_MIN": 0,
}


# -------------------------
# Helper
# -------------------------


def region_base(addr: int) -> int:
    return addr - (addr % CONFIG["REGION_SIZE"])


def block_index(addr: int) -> int:
    return (addr % CONFIG["REGION_SIZE"]) // CONFIG["BLOCK_SIZE"]


# -------------------------
# Data structures
# -------------------------


@dataclass
class IPEntry:
    ip: int
    class_id: str
    conf: int
    stride: Optional[int] = None
    last_addr: Optional[int] = None
    signature: Optional[List[int]] = None  # per-block counters


class IPTable:
    """Direct-mapped table mapping PC->IPEntry"""

    def __init__(self, entries: int = CONFIG["IPTABLE_ENTRIES"]):
        self.entries = entries
        self.table: Dict[int, IPEntry] = {}

    def _index(self, ip: int) -> int:
        return ip % self.entries

    def lookup(self, ip: int) -> Optional[IPEntry]:
        return self.table.get(self._index(ip))

    def insert(self, ip: int, class_id: str) -> IPEntry:
        idx = self._index(ip)
        entry = IPEntry(
            ip=ip,
            class_id=class_id,
            conf=CONFIG["INIT_CONF"],
            stride=None,
            last_addr=None,
            signature=[CONFIG["INIT_CONF"]] * CONFIG["BLOCKS_PER_REGION"],
        )
        self.table[idx] = entry
        logger.debug(f"Inserted IPEntry {entry}")
        return entry


# -------------------------
# Sub-prefetchers
# -------------------------


class CSPrefetcher:
    """Constant Stride Prefetcher for CS-class IPs"""

    def update(self, entry: IPEntry, access_addr: int):
        if entry.last_addr is not None:
            stride = (access_addr - entry.last_addr) // CONFIG["BLOCK_SIZE"]
            if entry.stride == stride:
                # stable stride, do nothing
                pass
            else:
                entry.stride = stride
        entry.last_addr = access_addr

    def predict(self, entry: IPEntry, access_addr: int) -> List[int]:
        if entry.stride is None:
            return []
        base = access_addr + entry.stride * CONFIG["BLOCK_SIZE"]
        preds = [
            base + i * entry.stride * CONFIG["BLOCK_SIZE"]
            for i in range(1, CONFIG["PREFETCH_DEGREE"] + 1)
        ]
        return preds


class GSPrefetcher:
    """Region Signature Prefetcher for GS-class IPs"""

    def update(self, entry: IPEntry, addr: int):
        idx = block_index(addr)
        sig = entry.signature
        if sig is None:
            sig = [CONFIG["INIT_CONF"]] * CONFIG["BLOCKS_PER_REGION"]
            entry.signature = sig
        sig[idx] = min(CONFIG["CONF_MAX"], sig[idx] + 1)

    def predict(self, entry: IPEntry, addr: int) -> List[int]:
        base = region_base(addr)
        sig = entry.signature or [0] * CONFIG["BLOCKS_PER_REGION"]
        preds = []
        for i, c in enumerate(sig):
            if c >= CONFIG["INIT_CONF"]:
                blk_addr = base + i * CONFIG["BLOCK_SIZE"]
                preds.append(blk_addr)
        return preds[: CONFIG["PREFETCH_DEGREE"]]


class CPLXPrefetcher:
    """Hybrid prefetcher mixing stride and signature"""

    def __init__(self):
        self.cs = CSPrefetcher()
        self.gs = GSPrefetcher()

    def update(self, entry: IPEntry, addr: int):
        self.cs.update(entry, addr)
        self.gs.update(entry, addr)

    def predict(self, entry: IPEntry, addr: int) -> List[int]:
        return (self.cs.predict(entry, addr) + self.gs.predict(entry, addr))[
            : CONFIG["PREFETCH_DEGREE"]
        ]


# -------------------------
# Scheduler
# -------------------------


class Scheduler:
    def __init__(self):
        self.outstanding: set[int] = set()
        self.recent: set[int] = set()

    def issue(self, addrs: List[int]) -> List[int]:
        res = []
        for a in addrs:
            if a in self.recent:  # MRB suppression
                continue
            if len(self.outstanding) >= CONFIG["MAX_OUTSTANDING"]:
                break
            if a not in self.outstanding:
                self.outstanding.add(a)
                res.append(a)
        # Record all new issues in recent
        self.recent.update(res)
        return res

    def credit(self, addr: int):
        self.outstanding.discard(addr)
        self.recent.add(addr)  # prevent immediate re-issue

    def new_cycle(self):
        # expire recent entries once per cycle
        self.recent.clear()


# -------------------------
# IPCP Prefetcher
# -------------------------


class IPCPPrefetcher:
    """Main IPCP Prefetcher entrypoint"""

    def __init__(self):
        self.ip_table = IPTable()
        self.cs = CSPrefetcher()
        self.gs = GSPrefetcher()
        self.cplx = CPLXPrefetcher()
        self.scheduler = Scheduler()
        self.initialized = False

    def init(self):
        self.initialized = True
        logger.info("IPCP initialized")

    def close(self):
        self.initialized = False
        logger.info("IPCP closed")

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        ip = access.pc
        addr = access.address

        self.scheduler.new_cycle()
        entry = self.ip_table.lookup(ip)
        if entry is None:
            # classify new IP heuristically (default: GS)
            entry = self.ip_table.insert(ip, "GS")

        # Update predictor
        if entry.class_id == "CS":
            self.cs.update(entry, addr)
            candidates = self.cs.predict(entry, addr)
        elif entry.class_id == "GS":
            self.gs.update(entry, addr)
            candidates = self.gs.predict(entry, addr)
        else:  # CPLX
            self.cplx.update(entry, addr)
            candidates = self.cplx.predict(entry, addr)

        # Feedback
        if prefetch_hit:
            entry.conf = min(CONFIG["CONF_MAX"], entry.conf + 1)
            self.scheduler.credit(addr)
        else:
            entry.conf = max(CONFIG["CONF_MIN"], entry.conf - 1)

        # Filter predictions within region
        base = region_base(addr)
        candidates = [c for c in candidates if region_base(c) == base]

        return self.scheduler.issue(candidates)
