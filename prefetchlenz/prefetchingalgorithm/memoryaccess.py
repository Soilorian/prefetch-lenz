from dataclasses import dataclass


@dataclass
class MemoryAccess:
    address: int
    pc: int
