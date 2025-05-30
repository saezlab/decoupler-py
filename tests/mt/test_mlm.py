import logging

import numpy as np
import statsmodels.api as sm
import pytest

import decoupler as dc


def test_fit(
    mat,
    adjmat,
):
    X, obs, var = mat
    n_features, n_fsets = adjmat.shape
    n_samples, _ = X.shape
    adjmat = np.column_stack((np.ones((n_features, )), adjmat))
    inv = np.linalg.inv(np.dot(adjmat.T, adjmat))
    df = n_features - n_fsets - 1
    coef, t = dc.mt._mlm._fit.py_func(
        X=adjmat,
        y=X.T,
        inv=inv,
        df=df,
    )
    # Assert output shapes
    assert isinstance(coef, np.ndarray)
    assert isinstance(t, np.ndarray)
    print(coef.shape, t.shape)
    assert coef.shape == (n_samples, n_fsets)
    assert t.shape == (n_samples, n_fsets)
    

@pytest.mark.parametrize('tval', [True, False])
def test_func_mlm(
    mat,
    adjmat,
    tval,
):
    X, obs, var = mat
    dc_es, dc_pv = dc.mt._mlm._func_mlm(mat=X, adj=adjmat, tval=tval)
    st_es, st_pv = np.zeros(dc_es.shape), np.zeros(dc_pv.shape)
    for i in range(st_es.shape[0]):
        y = X[i, :]
        x = sm.add_constant(adjmat)
        model = sm.OLS(y, x)
        res = model.fit()
        if tval:
            st_es[i, :] = res.tvalues[1:]
        else:
            st_es[i, :] = res.params[1:]
        st_pv[i, :] = res.pvalues[1:]
    assert np.allclose(dc_es, st_es)
    assert np.allclose(dc_pv, st_pv)
