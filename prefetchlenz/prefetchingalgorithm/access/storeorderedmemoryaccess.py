from dataclasses import dataclass
from typing import Optional

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class StoreOrderedMemoryAccess(MemoryAccess):
    isWrite: bool = False
    tid: Optional[int] = None
