from prefetchlenz.dataloader.impl.ArrayDataLoader import MemoryAccess
from prefetchlenz.prefetchingalgorithm.impl.referencepredictiontable import RptAlgorithm


def test_rpt_initial_no_predict():
    rpt = RptAlgorithm()
    rpt.init()
    # first encounter → no prediction
    assert (
        rpt.progress(
            MemoryAccess(
                address=1,
                pc=1,
            )
        )
        == []
    )


def test_rpt_predicts_stride():
    rpt = RptAlgorithm()
    rpt.init()
    # feed same stride three times to reach steady
    rpt.progress(
        MemoryAccess(
            address=100,
            pc=1,
        )
    )  # sets last_addr=100
    rpt.progress(
        MemoryAccess(
            address=104,
            pc=1,
        )
    )  # stride=4, state=0→0
    rpt.progress(
        MemoryAccess(
            address=108,
            pc=1,
        )
    )  # stride=4, state=0→1
    preds = rpt.progress(
        MemoryAccess(
            address=112,
            pc=1,
        )
    )  # stride=4, state=1→2 → predict 116
    assert preds == [116]
