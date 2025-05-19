import numpy as np
import pytest

import decoupler as dc


@pytest.fixture
def adata():
    adata, _ = dc.ds.toy(nobs=40, nvar=20, bval=2, seed=42, verbose=False)
    return adata


@pytest.fixture
def tdata():
    tdata, _ = dc.ds.toy(nobs=40, nvar=20, bval=2, seed=42, verbose=False, pstime=True)
    return tdata


@pytest.fixture
def net():
    _, net = dc.ds.toy(nobs=2, nvar=12, bval=2, seed=42, verbose=False)
    net = dc.pp.prune(features=net['target'].unique(), net=net, tmin=3)
    return net


@pytest.fixture
def unwnet(net):
    return net.drop(columns=['weight'], inplace=False)


@pytest.fixture
def mat(
    adata,
):
    return dc.pp.extract(data=adata)


@pytest.fixture
def idxmat(
    mat,
    net,
):
    X, obs, var = mat
    sources, cnct, starts, offsets = dc.pp.idxmat(features=var, net=net, verbose=False)
    return cnct, starts, offsets


@pytest.fixture
def adjmat(
    mat,
    net,
):
    X, obs, var = mat
    sources, targets, adjmat = dc.pp.adjmat(features=var, net=net, verbose=False)
    return adjmat
