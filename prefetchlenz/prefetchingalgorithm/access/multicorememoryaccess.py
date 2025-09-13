from dataclasses import dataclass

from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess


@dataclass
class MulticoreMemoryAccess(MemoryAccess):
    cpu: int
