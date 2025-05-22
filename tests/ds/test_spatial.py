import warnings

import pandas as pd
import pytest
import anndata as ad

import decoupler as dc


def test_msvisium():
    adata = dc.ds.msvisium()
    assert isinstance(adata, ad.AnnData)
    assert adata.raw is None
    assert isinstance(adata.obs, pd.DataFrame)
    cols = {'niches'}
    assert cols.issubset(adata.obs.columns)
    for col in cols:
        assert isinstance(adata.obs[col].dtype, pd.CategoricalDtype)
