from prefetchlenz.prefetchingalgorithm.impl.referencepredictiontable import RPTAlgorithm


def test_rpt_initial_no_predict():
    rpt = RPTAlgorithm()
    rpt.init()
    # first encounter → no prediction
    assert rpt.progress(100) == []


def test_rpt_predicts_stride():
    rpt = RPTAlgorithm()
    rpt.init()
    # feed same stride three times to reach steady
    rpt.progress(100)  # sets last_addr=100
    rpt.progress(104)  # stride=4, state=0→0
    rpt.progress(108)  # stride=4, state=0→1
    preds = rpt.progress(112)  # stride=4, state=1→2 → predict 116
    assert preds == [116]
