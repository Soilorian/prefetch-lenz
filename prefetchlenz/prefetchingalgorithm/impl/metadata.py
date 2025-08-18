import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.metadata")


class MetadataEntry:
    """
    Represents one metadata entry for a prefetcher.
    Each entry tracks (key -> metadata dict).
    """

    def __init__(self, key: int, data: Optional[Dict[str, Any]] = None):
        self.key = key
        self.data = data or {}
        self.use_count = 0

    def touch(self):
        self.use_count += 1


class MetadataCache:
    """
    Set-associative metadata cache with LRU replacement.
    Keys = e.g., PC or history hash.
    Values = MetadataEntry objects.
    """

    def __init__(self, num_sets: int = 64, num_ways: int = 8):
        self.num_sets = num_sets
        self.num_ways = num_ways
        self.sets = [OrderedDict() for _ in range(num_sets)]

    def _index(self, key: int) -> int:
        return key % self.num_sets

    def lookup(self, key: int) -> Optional[MetadataEntry]:
        idx = self._index(key)
        set_ = self.sets[idx]
        if key in set_:
            entry = set_[key]
            entry.touch()
            set_.move_to_end(key)
            logger.debug("Metadata hit: key=%s", key)
            return entry
        logger.debug("Metadata miss: key=%s", key)
        return None

    def insert(self, key: int, data: Optional[Dict[str, Any]] = None) -> MetadataEntry:
        idx = self._index(key)
        set_ = self.sets[idx]

        if key in set_:
            entry = set_[key]
            entry.data.update(data or {})
            entry.touch()
            set_.move_to_end(key)
            logger.debug("Metadata update: key=%s", key)
            return entry

        if len(set_) >= self.num_ways:
            evicted_key, evicted_entry = set_.popitem(last=False)
            logger.debug(
                "Metadata evict: key=%s use_count=%d",
                evicted_key,
                evicted_entry.use_count,
            )

        entry = MetadataEntry(key, data)
        set_[key] = entry
        logger.debug("Metadata insert: key=%s", key)
        return entry

    def update(self, key: int, data: Dict[str, Any]):
        entry = self.lookup(key)
        if entry:
            entry.data.update(data)
        else:
            self.insert(key, data)

    def remove(self, key: int):
        idx = self._index(key)
        set_ = self.sets[idx]
        if key in set_:
            del set_[key]
            logger.debug("Metadata remove: key=%s", key)

    def flush(self):
        self.sets = [OrderedDict() for _ in range(self.num_sets)]
        logger.debug("Metadata cache flushed")


class MetadataPrefetcher(PrefetchAlgorithm):
    """
    Prefetcher implementing Efficient Metadata Management.
    Stores correlation metadata in a set-associative metadata cache,
    and uses it to issue prefetches for likely future addresses.
    """

    def __init__(self, num_sets: int = 64, num_ways: int = 8):
        self.cache = MetadataCache(num_sets, num_ways)

    def init(self):
        self.cache.flush()
        logger.info("MetadataPrefetcher initialized")

    def progress(self, access: MemoryAccess, prefetch_hit: bool):
        pc = int(access.pc)
        addr = int(access.address)

        # 1. Lookup or create metadata entry for this PC
        entry = self.cache.lookup(pc)
        if not entry:
            entry = self.cache.insert(pc, {"history": [], "prefetches": []})

        # 2. Record this address in history
        history: List[int] = entry.data.setdefault("history", [])
        history.append(addr)
        if len(history) > 4:  # cap history length
            history.pop(0)

        # 3. If we have at least 2 history points, record correlation
        if len(history) >= 2:
            prev = history[-2]
            nexts: List[int] = entry.data.setdefault("prefetches", [])
            if addr not in nexts:
                nexts.append(addr)
                logger.debug(
                    "Recorded correlation: pc=%s prev=0x%x -> next=0x%x", pc, prev, addr
                )

        # 4. Issue prefetches for recorded targets
        targets = entry.data.get("prefetches", [])
        for t in targets:
            if t != addr:  # avoid prefetching current address
                self._issue_prefetch(t)

        # 5. Optionally adapt based on prefetch hits
        if prefetch_hit:
            entry.touch()

    def _issue_prefetch(self, address: int):
        logger.debug("Prefetch issued for address=0x%x", address)
        # Hook into your system’s prefetch enqueue here.

    def close(self):
        logger.info("MetadataPrefetcher closed")
        self.cache.flush()
