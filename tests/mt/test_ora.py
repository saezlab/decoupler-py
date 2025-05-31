import math

import numpy as np
import scipy.stats as sts
import scipy.sparse as sps
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'a,b,c,d',
    [
        [10, 1, 2, 1000],
        [0, 20, 35, 5],
        [1, 2, 3, 4],
        [0, 1, 2, 500],
    ]
)
def test_table(
    a,
    b,
    c,
    d,
):
    dc_es = dc.mt._ora._oddsr.py_func(a=a, b=b, c=c, d=d, ha_corr=0., log=False)
    dc_pv = dc.mt._ora._test1t.py_func(a=a, b=b, c=c, d=d)
    st_es, st_pv = sts.fisher_exact([[a, b],[c, d]])
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
    row = sts.rankdata(X[0], method='ordinal')
    ranks = np.arange(row.size, dtype=np.int_)
    row = ranks[(row > 2) | (row < 0)]
    es, pv = dc.mt._ora._runora.py_func(
        row=row,
        ranks=ranks,
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
        row = sts.rankdata(X[i], method='ordinal')
        row = set(ranks[row > n_up])
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
            _, st_pv[i, j] = sts.fisher_exact([[a, b],[c, d]])
            a += ha_corr
            b += ha_corr
            c += ha_corr
            d += ha_corr
            es = sts.fisher_exact([[a, b],[c, d]])
            st_es[i, j], _ = np.log(es)
    assert np.isclose(dc_es, st_es).all()
    assert np.isclose(dc_pv, st_pv).all()
