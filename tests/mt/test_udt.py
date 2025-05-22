import numpy as np
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(),
        dict(n_estimators=10),
        dict(max_depth=1),
        dict(gamma=0.01),
    ]
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
