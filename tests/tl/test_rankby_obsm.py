import pandas as pd
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'key,uns_key',
    [
        ['X_pca', 'rank_obsm'],
        ['X_pca', None],
        ['X_umap', 'other'],
        ['score_ulm', 'other'],
        ['score_ulm', None],
    ]
)
def test_rankby_obsm(
    tdata_obsm,
    key,
    uns_key,
):
    tdata_obsm = tdata_obsm.copy()
    tdata_obsm.obs['dose'] = 'Low'
    tdata_obsm.obs.loc[tdata_obsm.obs_names[5], 'dose'] = 'High'
    res = dc.tl.rankby_obsm(tdata_obsm, key=key, uns_key=uns_key)
    if uns_key is None:
        assert isinstance(res, pd.DataFrame)
    else:
        assert res is None
        assert uns_key in tdata_obsm.uns
        assert isinstance(tdata_obsm.uns[uns_key], pd.DataFrame)
