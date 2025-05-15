import logging

import numpy as np
import statsmodels.api as sm
import pytest

import decoupler as dc


def test_func_mlm(
    mat,
    adjmat,
):
    X, obs, var = mat
    dc_es, dc_pv = dc.mt._mlm._func_mlm(mat=X, adj=adjmat)
    st_es, st_pv = np.zeros(dc_es.shape), np.zeros(dc_pv.shape)
    for i in range(st_es.shape[0]):
        y = X[i, :]
        x = sm.add_constant(adjmat)
        model = sm.OLS(y, x)
        res = model.fit()
        st_es[i, :] = res.tvalues[1:]
        st_pv[i, :] = res.pvalues[1:]
    assert np.allclose(dc_es, st_es)
    assert np.allclose(dc_pv, st_pv)
