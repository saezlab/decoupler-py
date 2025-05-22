import logging
import tempfile

import pandas as pd
import numpy as np
import pytest

import decoupler as dc


def test_read_gmt():
    gmt = 'S1\tlink\tG1\tG2\tG3\nS2\tlink\tG2\tG3\tG4\tG5\tG6'
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=True) as tmp:
        tmp.write(gmt)
        tmp.flush()
        gmt = dc.pp.read_gmt(tmp.name)
    assert isinstance(gmt, pd.DataFrame)
    assert {'source', 'target'}.issubset(gmt.columns)


@pytest.mark.parametrize('verbose', [True, False])
def test_validate_net(
    net,
    verbose,
    caplog,
):
    with caplog.at_level(logging.WARNING):
        vnet = dc.pp.net._validate_net(net=net, verbose=verbose)
    assert caplog.text == ''
    assert net.shape == vnet.shape
    assert 'source' in vnet.columns
    assert 'target' in vnet.columns
    assert 'weight' in vnet.columns

    net.drop(columns=['weight'], inplace=True)
    assert 'weight' not in net.columns
    with caplog.at_level(logging.WARNING):
        vnet = dc.pp.net._validate_net(net=net, verbose=verbose)
    if verbose:
        assert len(caplog.text) > 0
    assert 'weight' in vnet.columns

    enet = net.rename(columns={'source': 'tf'})
    with pytest.raises(AssertionError):
        dc.pp.net._validate_net(enet)


@pytest.mark.parametrize(
    'tmin,nrow_equal,raise_err',
    [
        [0, True, False],
        [3, True, False],
        [4, False, False],
        [5, False, True],
    ]
)
def test_prune(
    net,
    tmin,
    nrow_equal,
    raise_err,
):
    features = ['G{:02d}'.format(i + 1) for i in range(20)]
    if raise_err:
        with pytest.raises(AssertionError):
            dc.pp.net.prune(features=features, net=net, tmin=tmin)
    else:
        pnet = dc.pp.net.prune(features=features, net=net, tmin=tmin)
        if nrow_equal:
            assert pnet.shape[0] == net.shape[0]
        else:
            assert pnet.shape[0] < net.shape[0]


def test_adj(
    net,
):
    sources, targets, adj = dc.pp.net._adj(net=net)
    sources, targets = list(sources), list(targets)
    net = net.set_index(['source', 'target'])
    for s in sources:
        j = sources.index(s)
        snet = net.loc[s]
        for t in snet.index:
            i = targets.index(t)
            w_adj = adj[i, j]
            w_net = snet.loc[t]['weight']
            print(s, t, i, j, w_net, w_adj)
            assert w_net == w_adj
    assert (len(targets), len(sources)) == adj.shape
    assert adj.dtype == float


def test_order(
    net
):
    sources, targets, adjmat = dc.pp.net._adj(net=net)

    rtargets = targets[::-1]
    radjmat = adjmat[::-1]
    oadjmat = dc.pp.net._order(features=targets, targets=rtargets, adjmat=radjmat)
    assert (adjmat == oadjmat).all()
    
    mfeatures = list(targets) + ['x', 'y', 'z']
    madjmat = dc.pp.net._order(features=mfeatures, targets=targets, adjmat=adjmat)
    assert np.all(madjmat == 0, axis=1).sum() == 3
    
    lfeatures = targets[:5]
    assert lfeatures.size < targets.size
    ladjmat = dc.pp.net._order(features=lfeatures, targets=targets, adjmat=adjmat)
    assert ladjmat.shape[0] < adjmat.shape[0]
    assert np.all(ladjmat == 0, axis=1).sum() == 0
    

@pytest.mark.parametrize('verbose', [True, False])
def test_adjmat(
    adata,
    net,
    unwnet,
    verbose,
    caplog,
):
    features = adata.var_names
    with caplog.at_level(logging.INFO):
        sources, targets, adjmat = dc.pp.adjmat(
            features=features, net=net, verbose=verbose
        )
    adjmat = adjmat.ravel()
    non_zero_adjmat = adjmat[adjmat != 0.]
    assert not all(non_zero_adjmat == 1.)
    if verbose:
        assert len(caplog.text) > 0
    else:
        assert caplog.text == ''
    unwnet = dc.pp.net._validate_net(net=unwnet)
    sources, targets, adjmat = dc.pp.adjmat(
        features=features, net=unwnet
    )
    adjmat = adjmat.ravel()
    non_zero_adjmat = adjmat[adjmat != 0.]
    assert all(non_zero_adjmat == 1.)
    

@pytest.mark.parametrize('verbose', [True, False])
def test_idxmat(
    adata,
    net,
    verbose,
    caplog,
):
    features = adata.var_names
    with caplog.at_level(logging.INFO):
        sources, cnct, starts, offsets = dc.pp.idxmat(features=features, net=net, verbose=verbose)
    if verbose:
        assert len(caplog.text) > 0
    else:
        assert caplog.text == ''
    assert sources.size == starts.size
    assert starts.size == offsets.size
    assert (net.groupby('source')['target'].size().loc[sources] == offsets).all()
    assert cnct.size == offsets.sum()


@pytest.mark.parametrize('j', [0, 1, 2, 3])
def test_getset(
    idxmat,
    j,
):
    cnct, starts, offsets = idxmat
    fset = dc.pp.net._getset(cnct=cnct, starts=starts, offsets=offsets, j=j)
    assert isinstance(fset, np.ndarray)
    assert fset.size == offsets[j]


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_shuffle_net(
    net,
    seed,
):
    s_w_net = net.groupby(['source'])['weight'].apply(lambda x: set(sorted(x)))
    s_t_net = net.groupby(['source'])['target'].apply(lambda x: set(sorted(x)))
    t_net = dc.pp.shuffle_net(net=net, target=True, weight=False, same_seed=False, seed=seed)
    s_t_t_net = t_net.groupby(['source'])['target'].apply(lambda x: set(sorted(x)))
    s_w_t_net = t_net.groupby(['source'])['weight'].apply(lambda x: set(sorted(x)))
    inter = s_w_net.index.intersection(s_w_t_net.index)
    assert inter.size > 0
    assert not all(s_t_t_net.loc[s].issubset(s_t_net.loc[s]) for s in inter)
    assert all(s_w_t_net.loc[s].issubset(s_w_net.loc[s]) for s in inter)
    w_net = dc.pp.shuffle_net(net=net, target=False, weight=True, same_seed=False, seed=seed)
    s_t_w_net = w_net.groupby(['source'])['target'].apply(lambda x: set(sorted(x)))
    s_w_w_net = w_net.groupby(['source'])['weight'].apply(lambda x: set(sorted(x)))
    inter = s_w_net.index.intersection(s_w_w_net.index)
    assert inter.size > 0
    assert all(s_t_w_net.loc[s].issubset(s_t_net.loc[s]) for s in inter)
    assert not all(s_w_w_net.loc[s].issubset(s_w_net.loc[s]) for s in inter)
    tw_net = dc.pp.shuffle_net(net=net, target=True, weight=True, same_seed=False, seed=seed)
    stw_net = dc.pp.shuffle_net(net=net, target=True, weight=True, same_seed=True, seed=seed)
    t_net = net.groupby(['target'])['weight'].apply(lambda x: set(sorted(x)))
    t_tw_net = tw_net.groupby(['target'])['weight'].apply(lambda x: set(sorted(x)))
    t_stw_net = stw_net.groupby(['target'])['weight'].apply(lambda x: set(sorted(x)))
    inter = t_net.index.intersection(t_tw_net.index).intersection(t_stw_net.index)
    assert inter.size > 0
    assert not all(t_tw_net.loc[s].issubset(t_net.loc[s]) for s in inter)
    assert all(t_stw_net.loc[s].issubset(t_net.loc[s]) for s in inter)


def test_net_corr(
    adata,
    net,
):
    n_src = net['source'].unique().size
    corr = dc.pp.net_corr(net=net, data=None, tmin=0)
    assert isinstance(corr, pd.DataFrame)
    cols = {'source_a', 'source_b', 'corr', 'pval', 'padj'}
    assert cols.issubset(corr.columns)
    assert (corr['source_a'] != corr['source_b']).all()
    n_pairs = n_src * (n_src - 1) // 2
    assert corr.shape[0] == n_pairs
    corr = dc.pp.net_corr(net=net, data=adata, tmin=0)
    assert isinstance(corr, pd.DataFrame)
    n_pairs = n_src * (n_src - 1) // 2
    assert corr.shape[0] == n_pairs
