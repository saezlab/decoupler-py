import logging

import numpy as np
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
    'nobs,nvar,bval,seed,verbose',
    [
        [10, 15, 2, 42, False],
        [2, 12, 2, 42, False],
        [100, 50, 0, 0, True],
        [10, 500, 0, 0, True],
        
    ]
)
def test_toy(
    nobs,
    nvar,
    bval,
    seed,
    verbose,
    caplog,
):
    with caplog.at_level(logging.INFO):
        adata, net = dc.ds.toy(nobs=nobs, nvar=nvar, bval=bval, seed=seed, verbose=verbose)

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
    print(adata.X[:, -1].ravel())
    assert ((bval - 1) < np.mean(adata.X[:, -1].ravel()) < (bval + 1)) or nvar == 12
