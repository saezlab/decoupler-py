import pandas as pd
import scipy.sparse as sps
import pytest

import decoupler as dc


def test_return(
    adata,
    net,
):
    mth = dc.mt.ulm
    adata = adata[:4].copy()
    adata.X[:, 0] = 0.
    es, pv = mth(data=adata.to_df(), net=net, tmin=0)
    r = dc.mt._run._return(name=mth.name, data=adata, es=es, pv=pv)
    assert r is None
    r = dc.mt._run._return(name=mth.name, data=adata.to_df(), es=es, pv=pv)
    assert isinstance(r, tuple)
    assert isinstance(r[0], pd.DataFrame)
    assert isinstance(r[1], pd.DataFrame)


@pytest.mark.parametrize(
    'mth,bsize',
    [
        [dc.mt.zscore, 2],
        [dc.mt.ora, 2],
        [dc.mt.gsva, 250_000],
    ]
)
def test_run(
    adata,
    net,
    mth,
    bsize,
):
    sdata = adata.copy()
    sdata.X = sps.csr_matrix(sdata.X)
    des, dpv = dc.mt._run._run(
        name=mth.name,
        func=mth.func,
        adj=mth.adj,
        test=mth.test,
        data=adata.to_df(),
        net=net,
        tmin=0,
    )
    ses, spv = dc.mt._run._run(
        name=mth.name,
        func=mth.func,
        adj=mth.adj,
        test=mth.test,
        data=sdata.to_df(),
        net=net,
        tmin=0,
    )
    assert (des.values == ses.values).all()
    