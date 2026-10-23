import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
import scipy.stats as sts

import decoupler as dc


@pytest.mark.parametrize(
    "a,b,c,d",
    [
        [10, 1, 2, 1000],
        [0, 20, 35, 5],
        [1, 2, 3, 4],
        [0, 1, 2, 500],
    ],
)
def test_table(
    a,
    b,
    c,
    d,
):
    dc_es = dc.mt._ora._oddsr.py_func(a=a, b=b, c=c, d=d, ha_corr=0.0, log=False)
    dc_pv = dc.mt._ora._test1t.py_func(a=a, b=b, c=c, d=d)
    st_es, st_pv = sts.fisher_exact([[a, b], [c, d]])
    assert np.isclose(dc_es, st_es)
    assert np.isclose(dc_pv, st_pv)
    nb_pv = math.exp(-dc.mt._ora._mlnTest2t.py_func(a, a + b, a + c, a + b + c + d))
    assert np.isclose(dc_pv, nb_pv)


def test_runora(
    mat,
    idxmat,
):
    X, obs, var = mat
    cnct, starts, offsets = idxmat
    row = sts.rankdata(X[0], method="ordinal")
    ranks = np.arange(row.size, dtype=np.int_)
    row = ranks[row > (row.size - 2)]
    es, pv = dc.mt._ora._runora.py_func(
        row=set(row),
        ranks=set(ranks),
        cnct=cnct,
        starts=starts,
        offsets=offsets,
        n_bg=0,
        ha_corr=0.5,
    )
    assert isinstance(es, np.ndarray)
    assert isinstance(pv, np.ndarray)


def test_func_ora(
    mat,
    idxmat,
):
    X, obs, var = mat
    cnct, starts, offsets = idxmat
    n_up = 3
    ha_corr = 1
    dc_es, dc_pv = dc.mt._ora._func_ora(
        mat=sps.csr_matrix(X),
        cnct=cnct,
        starts=starts,
        offsets=offsets,
        n_up=n_up,
        n_bm=0,
        n_bg=None,
        ha_corr=1,
    )
    st_es, st_pv = np.zeros(dc_es.shape), np.zeros(dc_pv.shape)
    ranks = np.arange(X.shape[1], dtype=np.int_)
    rnk = set(ranks)
    for i in range(st_es.shape[0]):
        row = sts.rankdata(X[i], method="ordinal")
        row = set(ranks[row > (X.shape[1] - n_up)])
        for j in range(st_es.shape[1]):
            fset = dc.pp.net._getset(cnct=cnct, starts=starts, offsets=offsets, j=j)
            fset = set(fset)
            # Build table
            set_a = row.intersection(fset)
            set_b = fset.difference(row)
            set_c = row.difference(fset)
            a = len(set_a)
            b = len(set_b)
            c = len(set_c)
            set_u = set_a.union(set_b).union(set_c)
            set_d = rnk.difference(set_u)
            d = len(set_d)
            _, st_pv[i, j] = sts.fisher_exact([[a, b], [c, d]])
            a += ha_corr
            b += ha_corr
            c += ha_corr
            d += ha_corr
            es = sts.fisher_exact([[a, b], [c, d]])
            st_es[i, j], _ = np.log(es)
    assert np.isclose(dc_es, st_es).all()
    assert np.isclose(dc_pv, st_pv).all()


@pytest.mark.parametrize(
    "n_up,n_bm,obs_idxs",
    [
        [5, 0, [15, 16, 17, 18, 19]],
        [1, 0, [19]],
        [20, 0, list(range(20))],
        [3, 2, [0, 1, 17, 18, 19]],
        [0.5, 0, [19]],
    ],
)
def test_func_ora_selection(
    n_up,
    n_bm,
    obs_idxs,
):
    # Row values are ascending, so the top n_up features are the last ones
    nvar = 20
    X = np.arange(nvar, dtype=float).reshape(1, nvar)
    var = np.array([f"G{i:02d}" for i in range(nvar)])
    net = pd.DataFrame(
        {
            "source": ["S1"] * 5,
            "target": var[[15, 16, 17, 18, 19]],
            "weight": [1.0] * 5,
        }
    )
    sources, cnct, starts, offsets = dc.pp.idxmat(features=var, net=net, verbose=False)
    dc_es, dc_pv = dc.mt._ora._func_ora(
        mat=X, cnct=cnct, starts=starts, offsets=offsets, n_up=n_up, n_bm=n_bm, n_bg=None
    )
    # Build the expected contingency table from the features that should be selected
    row = set(obs_idxs)
    fset = {15, 16, 17, 18, 19}
    a = len(row & fset)
    b = len(fset - row)
    c = len(row - fset)
    d = nvar - len(row | fset)
    st_es = np.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
    st_pv = sts.fisher_exact([[a, b], [c, d]])[1]
    assert np.isclose(dc_es[0, 0], st_es)
    assert np.isclose(dc_pv[0, 0], st_pv)


def test_func_ora_validate(
    mat,
    idxmat,
):
    X, obs, var = mat
    cnct, starts, offsets = idxmat
    nvar = X.shape[1]
    kwargs = {"mat": X, "cnct": cnct, "starts": starts, "offsets": offsets}
    with pytest.raises(AssertionError, match="overlap"):
        dc.mt._ora._func_ora(**kwargs, n_up=nvar, n_bm=1, n_bg=None)
    with pytest.raises(AssertionError, match="contingency table is invalid"):
        dc.mt._ora._func_ora(**kwargs, n_up=5, n_bm=0, n_bg=4)


def test_ora_wide():
    # Selecting the top 5% of a wide matrix must not exceed n_bg
    adata, net = dc.ds.toy(nobs=5, nvar=1_000, seed=42, verbose=False)
    dc.mt.ora(adata, net, tmin=3, n_bg=100)
    es = adata.obsm["score_ora"].values
    pv = adata.obsm["padj_ora"].values
    assert np.isfinite(es).all()
    assert ((pv >= 0) & (pv <= 1)).all()
