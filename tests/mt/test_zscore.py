import numpy as np
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'flavor', ['KSEA', 'RoKAI']
)
def test_func_zscore(
    mat,
    adjmat,
    flavor,
):
    X, obs, var = mat
    es, pv = dc.mt._zscore._func_zscore(mat=X, adj=adjmat, flavor=flavor)
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()
