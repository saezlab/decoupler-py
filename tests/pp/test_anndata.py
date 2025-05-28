import pandas as pd
import numpy as np
import scipy.sparse as sps
import pytest
from anndata import AnnData

import decoupler as dc


@pytest.mark.parametrize(
    'key',
    ['X_pca', 'X_umap', 'score_ulm', 'padj_ulm']
)
def test_get_obsm(
    tdata_obsm,
    key,
):
    obsm = dc.pp.get_obsm(adata=tdata_obsm, key=key)
    assert isinstance(obsm, AnnData)
    assert obsm.n_obs == tdata_obsm.n_obs
    assert obsm.n_vars == tdata_obsm.obsm[key].shape[1]
    assert (obsm.obs == tdata_obsm.obs).values.all()
    assert (obsm.X == tdata_obsm.obsm[key]).all().all()


def test_swap_layer(
    adata,
):
    ldata = adata.copy()
    res = dc.pp.swap_layer(adata=ldata, key='counts', X_key=None, inplace=True)
    assert res is None
    assert ldata.X.sum().is_integer()
    assert list(ldata.layers.keys()) == ['counts']
    ldata = adata.copy()
    res = dc.pp.swap_layer(adata=ldata, key='counts', X_key=None, inplace=False)
    assert isinstance(res, AnnData)
    assert res.X.sum().is_integer()
    assert list(res.layers.keys()) == ['counts']
    res = dc.pp.swap_layer(adata=ldata, key='counts', X_key='X', inplace=False)
    assert isinstance(res, AnnData)
    assert res.X.sum().is_integer()
    assert list(res.layers.keys()) == ['counts', 'X']
    assert (ldata.X == res.layers['X']).all()
    res = dc.pp.swap_layer(adata=ldata, key='counts', X_key='X', inplace=True)
    assert res is None
    assert 'X' in ldata.layers


@pytest.mark.parametrize(
    'groups_col,mode,sparse,empty',
    [
        [None, 'sum', False, True],
        [None, 'sum', True, True],
        [None, 'mean', True, True],
        ['sample', 'sum', False, True],
        ['sample', 'sum', False, False],
        ['sample', 'sum', True, False],
        [['dose', 'group'], 'sum', False, True],
        ['group', 'median', False, False],
        ['group', lambda x: np.max(x) - np.min(x), True, True],
        ['group', dict(sum=np.sum, mean=np.mean), False, False],
    ]
)
def test_pseudobulk(
    adata,
    groups_col,
    mode,
    sparse,
    empty,
    rng,
):
    adata = adata.copy()
    adata.obs['sample'] = adata.obs['sample'].astype('object')
    adata.obs['dose'] = rng.choice(['low', 'medium', 'high'], size=adata.n_obs, replace=True)
    if empty:
        adata.X[:, 3] = 0.
        adata.layers['counts'][:, 3] = 0.
        msk = adata.obs['sample'] == 'S01'
        adata.X[msk, :] = 0.
        adata.layers['counts'][msk, :] = 0.
    if sparse:
        adata.X = sps.csr_matrix(adata.X)
    if mode == 'sum':
        layer = 'counts'
    else:
        layer = None
    pdata = dc.pp.pseudobulk(
        adata=adata,
        sample_col='sample',
        groups_col=groups_col,
        mode=mode,
        empty=empty,
        layer=layer,
        skip_checks=False,
    )
    assert isinstance(pdata, AnnData)
    assert pdata.shape[0] < adata.shape[0]
    if empty:
        assert pdata.shape[1] < adata.shape[1]
    else:
        assert pdata.shape[1] == adata.shape[1]
    assert not pdata.obs['sample'].str.contains('_').any()
    obs_cols = {'psbulk_cells', 'psbulk_counts'}
    assert obs_cols.issubset(pdata.obs.columns)
    assert 'psbulk_props' in pdata.layers
    prop = pdata.layers['psbulk_props']
    assert ((0. <= prop) & (prop <= 1.)).all()
    if sparse:
        assert isinstance(pdata.X, np.ndarray)
    if isinstance(mode, dict):
        assert set(mode.keys()).issubset(pdata.layers.keys())


@pytest.mark.parametrize('inplace', [True, False])
def test_filter_samples(
    pdata,
    inplace,
):
    f_pdata = pdata.copy()
    res = dc.pp.filter_samples(adata=f_pdata, min_counts=90, min_cells=4, inplace=inplace)
    if inplace:
        assert res is None
        assert f_pdata.shape[0] < pdata.shape[0]
    else:
        assert isinstance(res, np.ndarray)
        assert res.size < pdata.shape[0]


def test_filter_by_expr(
    pdata,
):
    """
    names_v <- c(
      'G11', 'G04', 'G05', 'G03', 'G07', 'G18', 'G17','G02','G10', 'G14',
      'G09', 'G16', 'G08', 'G13', 'G20', 'G01', 'G12', 'G15', 'G06', 'G19'
    )
    names_o <- c(
      'S01_A', 'S02_A', 'S03_A', 'S01_B', 'S02_B', 'S03_B'
    )
    data <- c(
    0., 0., 2., 0., 2., 0., 2.,17., 0., 3., 0., 3., 0., 3., 3.,18., 0., 0., 1., 0.,
    0.,35., 3.,44., 2., 6., 7.,26., 3., 6., 0., 6., 1., 5., 5.,24., 1., 3., 4., 5.,
    2., 0., 0.,10., 1., 4., 6.,25., 1., 5., 0., 2., 0., 3., 8.,35., 2., 2., 0.,13.,
    2., 0., 9., 1., 9., 3., 3., 0., 1., 4., 0., 0.,19., 0., 0., 0., 0., 3., 8., 4.,
    0., 1., 8., 1.,19., 0., 7., 2., 0., 7., 1., 2.,24., 3.,10., 3., 0., 5.,17., 2.,
    2., 0.,34., 4.,42., 3., 3., 1., 3.,10., 1., 0.,28., 6., 9., 0., 3., 4.,17., 5.
    )
    data <- matrix(data = data, byrow = TRUE, nrow = length(names_o))
    rownames(data) <- names_o
    colnames(data) <- names_v
    data <- t(data)
    group <- c('A', 'A', 'A', 'B', 'B', 'B')
    msk <- filterByExpr(
      y=data, group = group, lib.size = NULL, min.count = 10,
      min.total.count = 10, large.n = 10, min.prop = 0.7)
    rownames(data)[msk]
    msk <- filterByExpr(
      y=data, group = group, lib.size = NULL, min.count = 7,
      min.total.count = 10, large.n = 10, min.prop = 0.7)
    rownames(data)[msk]
    msk <- filterByExpr(
      y=data, group = group, lib.size = NULL, min.count = 7,
      min.total.count = 10, large.n = 0, min.prop = 0.1)
    rownames(data)[msk]
    msk <- filterByExpr(
      y=data, group = group, lib.size = 1, min.count = 3,
      min.total.count = 10, large.n = 0, min.prop = 0.1)
    rownames(data)[msk]
    """
    dc_var = dc.pp.filter_by_expr(
        adata=pdata, group='group', lib_size=None, min_count=10, min_total_count=10, large_n=10, min_prop=0.7, inplace=False
    )
    eg_var = np.array(["G07", "G02", "G08", "G01", "G06"])
    assert set(dc_var) == set(eg_var)
    dc_var = dc.pp.filter_by_expr(
        adata=pdata, group='group', lib_size=None, min_count=7, min_total_count=10, large_n=10, min_prop=0.7, inplace=False
    )
    eg_var = np.array(["G05", "G07", "G02", "G08", "G01", "G06"])
    assert set(dc_var) == set(eg_var)
    dc_var = dc.pp.filter_by_expr(
        adata=pdata, group='group', lib_size=None, min_count=7, min_total_count=10, large_n=0, min_prop=0.1, inplace=False
    )
    eg_var = np.array(["G04", "G05", "G03", "G07", "G17", "G02", "G14", "G08", "G20", "G01", "G06", "G19"])
    assert set(dc_var) == set(eg_var)
    dc_var = dc.pp.filter_by_expr(
        adata=pdata, group='group', lib_size=1, min_count=3, min_total_count=10, large_n=0, min_prop=0.1, inplace=False
    )
    eg_var = np.array([
        "G04", "G05", "G03", "G07", "G18", "G17", "G02", "G14", "G16", "G08", "G13", "G20", "G01", "G15", "G06", "G19"
    ])
    assert set(dc_var) == set(eg_var)
    pdata.X = sps.csr_matrix(pdata.X)
    dc_var = dc.pp.filter_by_expr(
        adata=pdata, group='group', lib_size=1, min_count=3, min_total_count=10, large_n=0, min_prop=0.1, inplace=False
    )
    assert set(dc_var) == set(eg_var)
    dc.pp.filter_by_expr(
        adata=pdata, group='group', lib_size=1, min_count=3, min_total_count=10, large_n=0, min_prop=0.1, inplace=True
    )
    assert set(pdata.var_names) == set(eg_var)


@pytest.mark.parametrize('inplace', [True, False])
def test_filter_by_prop(
    pdata,
    inplace,
):
    f_pdata = pdata.copy()
    res = dc.pp.filter_by_prop(adata=f_pdata, inplace=inplace)
    if inplace:
        assert res is None
        assert f_pdata.shape[1] < pdata.shape[1]
    else:
        assert isinstance(res, np.ndarray)
        assert res.size < pdata.shape[1]


@pytest.mark.parametrize('key', ['X_pca', 'score_ulm'])
def test_knn(
    tdata_obsm,
    key,
):
    k_adata = tdata_obsm.copy()
    res = dc.pp.knn(adata=k_adata, key=key)
    assert res is None
    k = f'{key}_connectivities'
    assert k in k_adata.obsp
    assert isinstance(k_adata.obsp[k], sps.csr_matrix)


@pytest.mark.parametrize(
    'names,order,label,nbins',
    [
        [None, 'pstime', None, 10],
        ['G05', 'pstime', None, 35],
        [['G05'], 'f_order', None, 10],
        [['G01', 'G06', 'G10'], 'pstime', 'group', 14],
    ]
)
def test_bin_order(
    tdata,
    names,
    order,
    label,
    nbins,
    rng,
):
    tdata.obs.loc[:, 'f_order'] = rng.random(tdata.n_obs)
    tdata.X = sps.csr_matrix(tdata.X)
    df = dc.pp.bin_order(adata=tdata, names=names, order=order, label=label, nbins=nbins)
    assert isinstance(df, pd.DataFrame)
    cols = {'name', 'order', 'value'}
    assert cols.issubset(df.columns)
    assert ((0. <= df['order']) & (df['order'] <= 1.)).all()
    if label is not None:
        lcols = {'label', 'color'}
        assert lcols.issubset(df.columns)
        s_lbl = set(tdata.obs[label])
        assert s_lbl  == set(df['label'])
        assert len(s_lbl) == len(set(df['color']))
        assert f'{label}_colors' in tdata.uns.keys()
