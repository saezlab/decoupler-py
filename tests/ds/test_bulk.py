import pandas as pd
import pytest
import anndata as ad

import decoupler as dc


def test_hsctgfb():
    adata = dc.ds.hsctgfb()
    assert isinstance(adata, ad.AnnData)
    assert isinstance(adata.obs, pd.DataFrame)
    assert {'condition', 'sample_id'}.issubset(adata.obs)

@pytest.mark.parametrize(
    'thr_fc', [None, -1]
)
def test_knocktf(
    thr_fc, # val, None
):
    adata = dc.ds.knocktf(thr_fc=thr_fc)
    assert isinstance(adata, ad.AnnData)
    assert isinstance(adata.obs, pd.DataFrame)
    assert {'source', 'type_p'}.issubset(adata.obs.columns)
    if thr_fc is not None:
        assert (adata.obs['logFC'] < thr_fc).all()
