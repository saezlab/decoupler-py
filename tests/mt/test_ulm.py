import numpy as np
import scipy.stats as sts
import pytest

import decoupler as dc


def test_cov(
    mat,
    adjmat,
):
    X, obs, var = mat
    dc_cov = dc.mt._ulm._cov(A=adjmat, b=X.T)
    nsrcs = adjmat.shape[1]
    np_cov = np.cov(m=adjmat, y=X.T, rowvar=False)[:nsrcs, nsrcs:].T
    assert np.allclose(np_cov, dc_cov)


def test_cor(
    mat,
    adjmat,
):
    X, obs, var = mat
    dc_cor = dc.mt._ulm._cor(adjmat, X.T)
    nsrcs = adjmat.shape[1]
    np_cor = np.corrcoef(adjmat, X.T, rowvar=False)[:nsrcs, nsrcs:].T
    assert np.allclose(dc_cor, np_cor)
    assert np.all((dc_cor <= 1) * (dc_cor >= -1))


def test_tval():
    t = dc.mt._ulm._tval(r=0.4, df=28)
    assert np.allclose(2.30940108, t)
    t = dc.mt._ulm._tval(r=0.99, df=3)
    assert np.allclose(12.15540081, t)
    t = dc.mt._ulm._tval(r=-0.05, df=99)
    assert np.allclose(-0.49811675, t)


def test_func_ulm(
    mat,
    adjmat,
):
    X, obs, var = mat
    dc_es, dc_pv = dc.mt._ulm._func_ulm(mat=X, adj=adjmat)
    st_es, st_pv = np.zeros(dc_es.shape), np.zeros(dc_pv.shape)
    for i in range(st_es.shape[0]):
        for j in range(st_es.shape[1]):
            x = X[i, :]
            y = adjmat[:, j]
            slope, _, _, st_pv[i, j], std_err = sts.linregress(x, y)
            st_es[i, j] = slope / std_err
    assert np.allclose(dc_es, st_es)
    assert np.allclose(dc_pv, st_pv)
    
    