import pandas as pd
import pytest
import scanpy as sc

import decoupler as dc


@pytest.fixture
def tdata(
    tdata,
):
    sc.tl.pca(tdata)
    sc.pp.neighbors(tdata, n_neighbors=5)
    sc.tl.umap(tdata)
    return tdata


@pytest.mark.parametrize(
    'key,uns_key',
    [
        ['X_pca', 'rank_obsm'],
        ['X_pca', None],
        ['X_umap', 'other'],
    ]
)
def test_rankby_obsm(
    tdata,
    key,
    uns_key,
):
    res = dc.tl.rankby_obsm(tdata, key=key, uns_key=uns_key)
    if uns_key is None:
        assert isinstance(res, pd.DataFrame)
    else:
        assert res is None
        assert uns_key in tdata.uns
        assert isinstance(tdata.uns[uns_key], pd.DataFrame)
