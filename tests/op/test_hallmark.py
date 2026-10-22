import pandas as pd

import decoupler as dc


def test_hallmark():
    hm = dc.op.hallmark()
    assert isinstance(hm, pd.DataFrame)
    cols = {"source", "target"}
    assert cols.issubset(hm.columns)
    assert not hm.duplicated(["source", "target"]).any()


def test_hallmark_as_future():
    future = dc.op.hallmark(as_future=True)
    hm = future.result()
    assert isinstance(hm, pd.DataFrame)
    cols = {"source", "target"}
    assert cols.issubset(hm.columns)
    assert not hm.duplicated(["source", "target"]).any()
