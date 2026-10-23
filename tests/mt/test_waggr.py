import numpy as np
import pytest

import decoupler as dc


def test_funcs(rng):
    x = np.array([1, 2, 3, 4], dtype=float)
    w = rng.random(x.size)
    es = dc.mt._waggr._wsum.py_func(x=x, w=w)
    assert isinstance(es, float)
    es = dc.mt._waggr._wmean.py_func(x=x, w=w)
    assert isinstance(es, float)


@pytest.mark.parametrize(
    "fun,times,seed",
    [
        ["wmean", 10, 42],
        ["wsum", 5, 23],
        [lambda x, w: 0, 5, 1],
        ["wmean", 0, 42],
    ],
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


def test_waggr_pvalues():
    rng = np.random.default_rng(0)
    nvar, times = 200, 1000
    mat = rng.normal(size=(1, nvar))
    adj = np.zeros((nvar, 2))
    # Score of exactly 0: no evidence of enrichment
    mat[0, :2] = 5.0
    adj[:2, 0] = [1.0, -1.0]
    # Strongly enriched set
    mat[0, 50:70] += 3.0
    adj[50:70, 1] = 1.0
    es, pv = dc.mt._waggr._func_waggr(mat=mat, adj=adj, fun="wmean", times=times, seed=42)
    assert pv[0, 0] > 0.5
    assert pv[0, 1] == 1 / (times + 1)
    # A constant score is never more extreme than its permutations
    es, pv = dc.mt._waggr._func_waggr(mat=mat, adj=adj, fun=lambda x, w: 0, times=times, seed=42)
    assert (pv == 1).all()
