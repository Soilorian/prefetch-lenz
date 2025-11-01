from prefetchlenz.dataloader.impl.ArrayDataLoader import ArrayLoader
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess


def test_array_loader_returns_data():
    data = [
        MemoryAccess(address=1, pc=0),
        MemoryAccess(address=2, pc=0),
        MemoryAccess(address=3, pc=0),
        MemoryAccess(address=4, pc=0),
    ]
    loader = ArrayLoader(data)
    load = loader.data
    for i in range(len(data)):
        assert load[i] == data[i]
