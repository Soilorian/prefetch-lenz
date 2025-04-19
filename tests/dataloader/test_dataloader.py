from prefetchlenz.dataloader.dataloader import ArrayLoader


def test_array_loader_returns_data():
    data = [1, 2, 3, 4]
    loader = ArrayLoader(data)
    assert loader.load() == data
