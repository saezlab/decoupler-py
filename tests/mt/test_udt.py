import numpy as np
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_estimators": 10},
        {"max_depth": 1},
        {"gamma": 0.01},
    ],
)
def test_func_udt(
    mat,
    adjmat,
    kwargs,
):
    X, obs, var = mat
    es = dc.mt._udt._func_udt(mat=X, adj=adjmat, **kwargs)[0]
    assert np.isfinite(es).all()
    assert ((0 <= es) & (es <= 1)).all()
