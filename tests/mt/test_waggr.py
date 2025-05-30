import numpy as np
import pytest

import decoupler as dc


def test_funcs(
    rng
):
    x = np.array([1, 2, 3, 4], dtype=float)
    w = rng.random(x.size)
    es = dc.mt._waggr._wsum.py_func(x=x, w=w)
    assert isinstance(es, float)
    es = dc.mt._waggr._wmean.py_func(x=x, w=w)
    assert isinstance(es, float)


@pytest.mark.parametrize(
    'fun,times,seed',
    [
        ['wmean', 10, 42],
        ['wsum', 5, 23],
        [lambda x, w: 0, 5, 1],
        ['wmean', 0, 42],
    ]
)
def test_func_waggr(
    mat,
    adjmat,
    fun,
    times,
    seed,
):
    X, obs, var = mat
    es, pv = dc.mt._waggr._func_waggr(mat=X, adj=adjmat, fun=fun, times=times, seed=seed)
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()
