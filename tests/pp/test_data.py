import numpy as np
import scipy.sparse as sps
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
    sadata = adata.copy()
    sadata.X = sps.coo_matrix(sadata.X)
    X, obs, var = dc.pp.extract(data=sadata)
    assert isinstance(X, sps.csr_matrix)
    eadata = adata.copy()
    eadata.X[5, :] = 0.
    X, obs, var = dc.pp.extract(data=eadata, empty=True)
    assert X.shape[0] < eadata.shape[0]
    nadata = adata.copy()
    nadata.X = nadata.X * -1
    adata.raw = nadata
    X, obs, var = dc.pp.extract(data=adata, raw=True)
    assert (X < 0).all()
