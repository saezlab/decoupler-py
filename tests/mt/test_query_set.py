import pandas as pd
import pytest

import decoupler as dc


def test_query_set(
    net,
):
    ft = set(net[net["source"] == "T1"]["target"])
    df = dc.mt.query_set(features=ft, net=net, tmin=0)
    assert isinstance(df, pd.DataFrame)
    cols = {"source", "stat", "pval", "padj"}
    assert cols.issubset(df.columns)
    df = dc.mt.query_set(features=ft, net=net, n_bg=None, tmin=0)


def test_query_set_n_bg(
    net,
):
    ft = set(net["target"])
    with pytest.raises(AssertionError, match="contingency table is invalid"):
        dc.mt.query_set(features=ft, net=net, n_bg=2, tmin=0)
