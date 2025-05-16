import numpy as np
import pytest

import decoupler as dc


def test_extract(
    adata,
):
    data = [adata.X, adata.obs_names, adata.var_names]
    X, obs, var = dc.pp.extract(data=data)
    assert X.shape[0] == obs.size
    assert X.shape[1] == var.size
    X, obs, var = dc.pp.extract(data=adata.to_df())
    assert X.shape[0] == obs.size
    assert X.shape[1] == var.size
    X, obs, var = dc.pp.extract(data=adata)
    assert X.shape[0] == obs.size
    assert X.shape[1] == var.size
    adata.layers['counts'] = adata.X.round()
    X, obs, var = dc.pp.extract(data=adata, layer='counts')
    assert float(np.sum(X)).is_integer()
    nadata = adata.copy()
    nadata.X = nadata.X * -1
    adata.raw = nadata
    X, obs, var = dc.pp.extract(data=adata, raw=True)
    assert (X < 0).all()
