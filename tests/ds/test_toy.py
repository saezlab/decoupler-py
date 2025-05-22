import logging

import numpy as np
import scipy.stats as sts
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'nvar,val,size,hasval',
    [
        [3, 0., 5, False],
        [10, 0., 10, True],
    ]
)
def test_fillval(
    nvar,
    val,
    size,
    hasval,
):
    arr = np.array([1., 2., 3., 4., 5.])
    farr = dc.ds._toy._fillval(arr=arr, nvar=nvar, val=val)

    assert farr.size == size
    assert (val == farr[-1]) == hasval


@pytest.mark.parametrize(
    'nobs,nvar,bval,pstime,seed,verbose',
    [
        [10, 15, 2, True, 42, False],
        [2, 12, 2, False, 42, False],
        [100, 50, 0, False, 0, True],
        [10, 500, 0, True, 0, True],
        
    ]
)
def test_toy(
    nobs,
    nvar,
    bval,
    pstime,
    seed,
    verbose,
    caplog,
):
    with caplog.at_level(logging.INFO):
        adata, net = dc.ds.toy(nobs=nobs, nvar=nvar, bval=bval, pstime=pstime, seed=seed, verbose=verbose)
    if verbose:
        assert len(caplog.text) > 0
    else:
        assert caplog.text == ''
    assert all(adata.obs['group'].cat.categories == ['A', 'B'])
    msk = adata.obs['group'] == 'A'
    assert all(adata[msk, :4].X.mean(0) > adata[~msk, :4].X.mean(0))
    assert all(adata[msk, 4:8].X.mean(0) < adata[~msk, 4:8].X.mean(0))
    assert nobs == adata.n_obs
    assert nvar == adata.n_vars
    assert ((bval - 1) < np.mean(adata.X[:, -1].ravel()) < (bval + 1)) or nvar == 12
    if pstime:
        assert 'pstime' in adata.obs.columns
        assert ((0. <= adata.obs['pstime']) & (adata.obs['pstime'] <= 1.)).all()


@pytest.mark.parametrize(
    'shuffle_r,seed,nobs,nvar,is_diff',
    [
        [0.0, 1, 20, 31, True],
        [0.1, 2, 36, 41, True],
        [0.9, 3, 49, 21, False],
        [1.0, 4, 18, 41, False],
        
    ]
)
def test_toy_bench(
    net,
    shuffle_r,
    seed,
    nobs,
    nvar,
    is_diff,
):
    adata, bmnet = dc.ds.toy_bench(shuffle_r=shuffle_r, seed=seed, nobs=nobs, nvar=nvar)
    assert (net == bmnet).values.all()
    assert adata.n_obs == nobs
    assert adata.n_vars == nvar
    msk = adata.obs['group'] == 'A'
    a_adata = adata[msk, :].copy()
    b_adata = adata[~msk, :].copy()
    for j in adata.var_names[:8]:
        a = a_adata[:, j].X.ravel()
        b = b_adata[:, j].X.ravel()
        stat, pval = sts.ranksums(a, b)
        if is_diff:
            assert pval < 0.05
        else:
            assert pval > 0.05

    