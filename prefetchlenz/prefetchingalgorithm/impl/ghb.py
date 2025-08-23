import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# change logger name if you prefer a different module path
logger = logging.getLogger("prefetchLenz.prefetchingalgorithm.impl.ghb")


@dataclass
class GHBEntry:
    uid: int
    addr: int
    pc: int
    prev_uid: Optional[int]  # link to previous GHB entry UID for same PC
    ts: int  # insertion sequence number (for debugging)


class GlobalHistoryBufferPrefetcher:
    """
    Global History Buffer (GHB) prefetcher implementation.
    - ghb_size: number of GHB entries (circular buffer)
    - search_depth: how many previous entries in the PC chain to inspect
    - degree: how many prefetches to issue (addr + delta * i)
    - prefer_credited: if True, prefer delta candidates that were credited by prefetch hits
    """

    def __init__(
        self,
        ghb_size: int = 1024,
        search_depth: int = 8,
        degree: int = 4,
        prefer_credited: bool = True,
        outstanding_limit: int = 8192,
    ):
        self.ghb_size = max(16, ghb_size)
        self.search_depth = max(1, search_depth)
        self.degree = max(1, degree)
        self.prefer_credited = prefer_credited
        self.outstanding_limit = max(1, outstanding_limit)

        # circular buffer of GHBEntry or None
        self.ghb: List[Optional[GHBEntry]] = [None] * self.ghb_size
        # mapping from uid -> index in ghb array (only for currently valid entries)
        self.uid_to_index: Dict[int, int] = {}
        # index table: pc -> head_uid (most recent UID for that PC)
        self.index_table: Dict[int, int] = {}

        # uid generator and insertion sequence
        self._next_uid = 1
        self._insert_count = 0
        self._ghb_ptr = 0  # next index to overwrite

        # outstanding prefetch mapping: target_addr -> (origin_pc, delta)
        self.outstanding: Dict[int, Tuple[int, int]] = {}
        self.outstanding_q: deque = deque()

        # per-PC credited-delta counters (when a prefetch hit is observed)
        self.delta_credit: Dict[int, Counter] = defaultdict(Counter)

    def init(self):
        """Reset the state."""
        self.ghb = [None] * self.ghb_size
        self.uid_to_index.clear()
        self.index_table.clear()
        self._next_uid = 1
        self._insert_count = 0
        self._ghb_ptr = 0
        self.outstanding.clear()
        self.outstanding_q.clear()
        self.delta_credit.clear()
        logger.info(
            "GHB prefetcher initialized (size=%d search_depth=%d degree=%d)",
            self.ghb_size,
            self.search_depth,
            self.degree,
        )

    def _push_ghb(self, addr: int, pc: int) -> int:
        """
        Insert a new GHB entry for (addr, pc). Returns the new entry UID.
        Handles circular overwrite and uid->index mapping updates.
        """
        uid = self._next_uid
        self._next_uid += 1
        prev_uid = self.index_table.get(
            pc
        )  # previous head UID for this pc (may be None)

        # evict position
        idx = self._ghb_ptr
        # if overwriting an entry, remove its uid mapping
        old = self.ghb[idx]
        if old is not None:
            if old.uid in self.uid_to_index:
                del self.uid_to_index[old.uid]

        entry = GHBEntry(
            uid=uid, addr=addr, pc=pc, prev_uid=prev_uid, ts=self._insert_count
        )
        self._insert_count += 1

        # install
        self.ghb[idx] = entry
        self.uid_to_index[uid] = idx
        self.index_table[pc] = uid

        # advance pointer
        self._ghb_ptr = (self._ghb_ptr + 1) % self.ghb_size

        logger.debug(
            "GHB push: idx=%d uid=%d pc=0x%X addr=%d prev_uid=%s",
            idx,
            uid,
            pc,
            addr,
            str(prev_uid),
        )
        return uid

    def _collect_chain_addresses(self, head_uid: int, max_items: int) -> List[int]:
        """
        Walk the PC chain starting at head_uid and collect up to max_items addresses
        in chronological order [head, prev1, prev2, ...].
        """
        addrs: List[int] = []
        uid = head_uid
        steps = 0
        while uid is not None and steps < max_items:
            idx = self.uid_to_index.get(uid)
            if idx is None:
                # this UID no longer in GHB (overwritten) — stop
                break
            entry = self.ghb[idx]
            if entry is None:
                break
            addrs.append(entry.addr)
            uid = entry.prev_uid
            steps += 1
        return addrs

    def _choose_delta_from_chain(self, chain_addrs: List[int]) -> Optional[int]:
        """
        From a list of addresses [a0, a1, a2...], compute deltas a0-a1, a1-a2, ...
        Choose a delta candidate:
          - prefer deltas that repeat (most common)
          - if prefer_credited True, prefer deltas with credit in delta_credit
          - fallback to the most recent delta (a0-a1) if nothing repeats
        """
        if len(chain_addrs) < 2:
            return None
        deltas = [
            chain_addrs[i] - chain_addrs[i + 1] for i in range(len(chain_addrs) - 1)
        ]
        # count frequencies
        cnt = Counter(deltas)
        # find most common delta(s)
        most_common = cnt.most_common()
        if not most_common:
            return deltas[0]  # fallback

        # if prefer credited, sort by (credit_count, freq, recency preference)
        # build candidate list as (delta, freq, last_pos) where last_pos lower means more recent
        last_pos = {d: i for i, d in enumerate(deltas)}  # i=0 is most recent position
        candidates = []
        for d, f in most_common:
            credit = 0
            # gather global credits across PCs? We'll check per current PC in caller via delta_credit if desired.
            candidates.append((d, f, last_pos.get(d, len(deltas))))
        # prefer by freq then recency (smaller last_pos)
        candidates.sort(key=lambda t: (-t[1], t[2]))
        chosen_delta = candidates[0][0]
        return int(chosen_delta)

    def _record_outstanding(self, tgt: int, pc: int, delta: int):
        """Record an outstanding prefetch (for later crediting)."""
        if tgt in self.outstanding:
            return
        self.outstanding[tgt] = (pc, delta)
        self.outstanding_q.append(tgt)
        if len(self.outstanding_q) > self.outstanding_limit:
            old = self.outstanding_q.popleft()
            self.outstanding.pop(old, None)

    def _credit_prefetch_hit(self, addr: int) -> bool:
        """
        If addr matches an outstanding prefetch, credit the (pc, delta) that created it.
        Returns True if credited.
        """
        info = self.outstanding.pop(addr, None)
        if info is None:
            return False
        try:
            self.outstanding_q.remove(addr)
        except ValueError:
            pass
        pc, delta = info
        self.delta_credit[pc][delta] += 1
        logger.debug(
            "Prefetch hit credited: pc=0x%X delta=%d addr=%d cred=%d",
            pc,
            delta,
            addr,
            self.delta_credit[pc][delta],
        )
        return True

    def progress(self, access, prefetch_hit: bool) -> List[int]:
        """
        Process a MemoryAccess instance (must have .pc and .address).
        - prefetch_hit: True if the access was satisfied by an earlier prefetch.
        Returns a list of addresses to prefetch.
        """
        pc = access.pc
        addr = access.key

        # 1) if hit on a prefetched block, credit it
        if prefetch_hit:
            credited = self._credit_prefetch_hit(addr)
            if credited:
                logger.debug(
                    "Access %d (pc=0x%X) was a prefetch hit and credited.", addr, pc
                )
            else:
                logger.debug(
                    "Access %d (pc=0x%X) marked prefetch_hit but had no outstanding record.",
                    addr,
                    pc,
                )

        # 2) insert this access into GHB and link from PC head
        head_uid = self._push_ghb(addr=addr, pc=pc)

        # 3) choose delta:
        chosen_delta: Optional[int] = None

        # (A) If we have credited deltas for this PC, prefer the best credited delta
        if self.prefer_credited and pc in self.delta_credit and self.delta_credit[pc]:
            # pick delta with highest credit (break ties by absolute smallness)
            most = self.delta_credit[pc].most_common(1)
            if most:
                chosen_delta = most[0][0]
                logger.debug(
                    "PC 0x%X: using credited delta %d (cred=%d)",
                    pc,
                    chosen_delta,
                    most[0][1],
                )

        # (B) If no credited choice or prefer_credited False, analyze chain
        if chosen_delta is None:
            # collect chain addresses (most recent first) - we want up to search_depth+1 addresses to form deltas
            head = self.index_table.get(pc, None)
            if head is not None:
                chain_addrs = self._collect_chain_addresses(head, self.search_depth + 1)
                if chain_addrs:
                    chosen_delta = self._choose_delta_from_chain(chain_addrs)
                    logger.debug(
                        "PC 0x%X chain addrs=%s chosen_delta=%s",
                        pc,
                        chain_addrs,
                        str(chosen_delta),
                    )

        # (C) fallback: if nothing chosen, use most recent delta between this and previous head (if any)
        if chosen_delta is None:
            # try the previous head for pc (before we just inserted) — that prev_uid is stored in ghb entry we inserted
            idx = self.uid_to_index.get(head_uid)
            if idx is not None:
                prev_uid = self.ghb[idx].prev_uid
                if prev_uid is not None and prev_uid in self.uid_to_index:
                    prev_idx = self.uid_to_index[prev_uid]
                    prev_entry = self.ghb[prev_idx]
                    chosen_delta = addr - prev_entry.addr
                    logger.debug(
                        "PC 0x%X fallback delta from prev %d", pc, chosen_delta
                    )

        # if still None, give up (no prefetch)
        prefetches: List[int] = []
        if chosen_delta is None:
            logger.debug("PC 0x%X addr=%d no delta chosen -> no prefetch", pc, addr)
            return prefetches

        # 4) issue 'degree' prefetches: addr + chosen_delta * i
        for i in range(1, self.degree + 1):
            tgt = addr + chosen_delta * i
            # record outstanding so hits can be credited
            self._record_outstanding(tgt, pc, chosen_delta)
            prefetches.append(tgt)

        logger.info(
            "PC 0x%X addr=%d chosen_delta=%d degree=%d prefetches=%s",
            pc,
            addr,
            chosen_delta,
            self.degree,
            prefetches,
        )
        return prefetches

    def close(self):
        """Clear all structures."""
        self.ghb = [None] * self.ghb_size
        self.uid_to_index.clear()
        self.index_table.clear()
        self.outstanding.clear()
        self.outstanding_q.clear()
        self.delta_credit.clear()
        logger.info("GHB prefetcher closed")
