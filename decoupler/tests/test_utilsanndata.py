import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..utils_anndata import get_acts, swap_layer, extract_psbulk_inputs, check_X
from ..utils_anndata import format_psbulk_inputs, psbulk_profile, compute_psbulk, get_unq_dict, get_pseudobulk
from ..utils_anndata import check_if_skip, get_contrast, get_top_targets, format_contrast_results
from ..utils_anndata import get_filterbyexpr_inputs, get_min_sample_size, get_cpm_cutoff, get_cpm, filter_by_expr
from ..utils_anndata import filter_by_prop, rank_sources_groups


def test_get_acts():
    m = np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]])
    r = np.array(['S1', 'S2', 'S3'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    estimate = pd.DataFrame([[3.5, -0.5], [3.6, -0.6], [-1, 2]],
                            columns=['T1', 'T2'], index=r)
    adata = AnnData(df, dtype=np.float32)
    adata.obsm['estimate'] = estimate
    acts = get_acts(adata, 'estimate')
    assert acts.shape[0] == adata.shape[0]
    assert acts.shape[1] != adata.shape[1]
    assert np.any(acts.X < 0)
    assert not np.any(adata.X < 0)


def test_swap_layer():
    m = np.array([[1, 0, 2], [1, 0, 3], [0, 0, 1]])
    r = np.array(['S1', 'S2', 'S3'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df, dtype=np.float32)
    adata.layers['norm'] = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1)
    sdata = swap_layer(adata, layer_key='norm', X_layer_key='x', inplace=False)
    assert not np.all(np.mod(sdata.X, 1) == 0)
    assert 'x' in list(sdata.layers.keys())
    assert np.all(sdata.layers['x'] == adata.X)
    sdata = swap_layer(adata, layer_key='norm', X_layer_key=None, inplace=False)
    assert np.all(sdata.X == sdata.layers['norm'])
    assert 'x' not in list(sdata.layers.keys())
    swap_layer(adata, layer_key='norm', X_layer_key='x', inplace=True)
    assert not np.all(np.mod(adata.X, 1) == 0)
    assert 'x' in list(adata.layers.keys())
    adata = AnnData(df, dtype=np.float32)
    adata.layers['norm'] = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1)
    swap_layer(adata, layer_key='norm', X_layer_key=None, inplace=True)
    assert np.all(adata.X == adata.layers['norm'])
    assert 'x' not in list(adata.layers.keys())


def test_extract_psbulk_inputs():
    m = np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]])
    r = np.array(['S1', 'S2', 'S3'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    obs = pd.DataFrame([['C01', 'C01', 'C02']], columns=r, index=['celltype']).T
    adata = AnnData(df, obs=obs, dtype=np.float32)
    adata.layers['counts'] = adata.X
    adata_raw = adata.copy()
    adata_raw.raw = adata_raw
    extract_psbulk_inputs(adata, obs=None, layer='counts', use_raw=False)
    extract_psbulk_inputs(adata, obs=None, layer=None, use_raw=False)
    extract_psbulk_inputs(adata_raw, obs=None, layer=None, use_raw=True)
    extract_psbulk_inputs(df, obs=obs, layer=None, use_raw=False)
    with pytest.raises(ValueError):
        extract_psbulk_inputs(adata, obs=None, layer=None, use_raw=True)
    with pytest.raises(ValueError):
        extract_psbulk_inputs(df, obs=None, layer=None, use_raw=False)


def test_check_X():
    X = csr_matrix(np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]]))
    X_float = csr_matrix(np.array([[1.3, 0, 2.1], [1.48, 0.123, 3.33], [0, 0, 0]]))
    X_neg = csr_matrix(np.array([[1, 0, -2], [1, 0, -3], [0, 0, 0]]))
    X_inf = csr_matrix(np.array([[1, 0, np.nan], [1, 0, 3], [0, 0, 0]]))
    check_X(X, mode='sum', skip_checks=False)
    with pytest.raises(ValueError):
        check_X(X_inf, mode='sum', skip_checks=False)
    with pytest.raises(ValueError):
        check_X(X_inf, mode='sum', skip_checks=True)
    with pytest.raises(ValueError):
        check_X(X_neg, mode='sum', skip_checks=False)
    with pytest.raises(ValueError):
        check_X(X_float, mode='sum', skip_checks=False)
    check_X(X_neg, mode='sum', skip_checks=True)
    check_X(X_float, mode='sum', skip_checks=True)
    check_X(X_neg, mode=sum, skip_checks=False)
    check_X(X_neg, mode={'sum': sum}, skip_checks=False)


def test_format_psbulk_inputs():
    obs = pd.DataFrame([
        ['P1', 'S1', 'C1'],
        ['P1', 'S1', 'C1'],
        ['P1', 'S1', 'C2'],
        ['P1', 'S1', 'C2'],
        ['P1', 'S2', 'C1'],
        ['P1', 'S2', 'C1'],
        ['P1', 'S2', 'C2'],
        ['P1', 'S2', 'C2'],
        ['P2', 'S1', 'C1'],
        ['P2', 'S1', 'C1'],
        ['P2', 'S1', 'C2'],
        ['P2', 'S1', 'C2'],
        ['P2', 'S2', 'C1'],
        ['P2', 'S2', 'C1'],
        ['P2', 'S2', 'C2'],
        ['P2', 'S2', 'C2'],

    ], columns=['patient_id', 'sample_id', 'celltype'])

    new_obs, groups_col, smples, groups, n_rows = format_psbulk_inputs('patient_id', groups_col='patient_id', obs=obs)
    assert new_obs.shape[1] != obs.shape[1]
    assert groups_col is None
    assert smples.size == 2
    assert groups is None
    assert n_rows == 2
    new_obs, groups_col, smples, groups, n_rows = format_psbulk_inputs('patient_id', groups_col='celltype', obs=obs)
    assert new_obs.shape[1] != obs.shape[1]
    assert groups_col == 'celltype'
    assert smples.size == 2
    assert groups.size == 2
    assert n_rows == 4
    new_obs, groups_col, smples, groups, n_rows = format_psbulk_inputs('patient_id', groups_col=['sample_id', 'celltype'],
                                                                       obs=obs)
    assert new_obs.shape[1] != obs.shape[1]
    assert groups_col == 'sample_id_celltype'
    assert smples.size == 2
    assert groups.size == 4
    assert n_rows == 8


def test_psbulk_profile():
    profile = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    p = psbulk_profile(profile, mode='sum')
    assert np.all(p == np.array([12, 15, 18]))
    p = psbulk_profile(profile, mode='mean')
    assert np.all(p == np.array([4, 5, 6]))
    p = psbulk_profile(profile, mode='median')
    assert np.all(p == np.array([4, 5, 6]))
    p = psbulk_profile(profile, mode=sum)
    assert np.all(p == np.array([12, 15, 18]))
    with pytest.raises(ValueError):
        psbulk_profile(profile, mode='mode')


def test_compute_psbulk():
    sample_col, groups_col = 'sample_id', 'celltype'
    obs = pd.DataFrame([
        ['S1', 'C1'],
        ['S1', 'C1'],
        ['S1', 'C2'],
        ['S1', 'C2'],
        ['S2', 'C1'],
        ['S2', 'C1'],
        ['S2', 'C2'],
        ['S2', 'C2'],

    ], columns=[sample_col, groups_col])
    X = csr_matrix(np.array([
        [1, 0, 8],
        [1, 0, 3],
        [0, 2, 1],
        [0, 4, 0],
        [2, 0, 4],
        [2, 0, 1],
        [0, 6, 0],
        [1, 3, 2],
    ]))

    smples = np.array(['S1', 'S2'])
    groups = np.array(['C1', 'C2'])
    n_rows = len(smples) * len(groups)
    n_cols = X.shape[1]
    new_obs = pd.DataFrame(columns=obs.columns)
    min_cells, min_counts = 0., 0.
    mode = 'sum'
    dtype = np.float32

    psbulk, ncells, counts, props = compute_psbulk(n_rows, n_cols, X, sample_col, groups_col, smples, groups, obs,
                                                   new_obs, min_cells, min_counts, mode, dtype)
    assert np.all(np.sum(psbulk, axis=1) == counts)
    assert np.all(np.array(psbulk.shape) == np.array(props.shape))
    psbulk, ncells, counts, props = compute_psbulk(n_rows, n_cols, X, sample_col, groups_col, smples, groups, obs,
                                                   new_obs, min_cells, 9, mode, dtype)
    assert np.sum(psbulk, axis=1)[2] == 0.
    assert np.all(psbulk[2] == props[2])
    groups = smples
    n_rows = len(smples)
    obs = obs[['sample_id']].copy()
    new_obs = pd.DataFrame(columns=obs.columns)

    psbulk, ncells, counts, props = compute_psbulk(n_rows, n_cols, X, sample_col, None, smples, groups, obs,
                                                   new_obs, min_cells, min_counts, mode, dtype)
    assert np.all(np.sum(psbulk, axis=1) == counts)
    assert np.all(np.array(psbulk.shape) == np.array(props.shape))
    assert psbulk.shape[0] == 2
    psbulk, ncells, counts, props = compute_psbulk(n_rows, n_cols, X, sample_col, None, smples, groups, obs,
                                                   new_obs, min_cells, 21, mode, dtype)
    assert np.sum(psbulk, axis=1)[0] == 0.
    assert np.all(psbulk[0] == props[0])


def test_get_pseudobulk():
    sample_col, groups_col = 'sample_id', 'celltype'
    m = np.array([[6, 0, 1], [2, 0, 2], [1, 3, 3], [0, 1, 1], [1, 0, 1]])
    r = np.array(['B1', 'B2', 'B3', 'B4', 'B5'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    smples = np.array(['S1', 'S1', 'S1', 'S2', 'S2'])
    groups = np.array(['C1', 'C1', 'C1', 'C1', 'C2'])
    obs = pd.DataFrame([smples, groups], columns=r, index=[sample_col, groups_col]).T
    adata = AnnData(df, obs=obs, dtype=np.float32)

    pdata = get_pseudobulk(adata, groups_col, groups_col, min_cells=0, min_counts=0, min_prop=None, min_smpls=None)
    assert np.all(pdata.shape == np.array([2, 3]))
    pdata = get_pseudobulk(adata, groups_col, groups_col, min_cells=0, min_counts=0, min_prop=1, min_smpls=1)
    assert np.all(pdata.var_names == np.array(['G1', 'G3']))
    pdata = get_pseudobulk(adata, sample_col, groups_col, min_cells=0, min_counts=0, min_prop=0, min_smpls=0)
    assert np.all(pdata.shape == np.array([3, 3]))
    pdata = get_pseudobulk(adata, sample_col, groups_col, min_cells=0, min_counts=0, min_prop=1, min_smpls=2)
    assert np.all(pdata.var_names == np.array(['G1', 'G3']))
    pdata = get_pseudobulk(adata, sample_col, sample_col, min_cells=0, min_counts=0)
    assert pdata.shape[0] == 2
    pdata = get_pseudobulk(adata, sample_col, groups_col, min_cells=0, min_counts=0, mode='sum')
    assert pdata.shape[0] == 3
    assert np.all(pdata.X[0] == np.array([9., 3., 6.]))
    pdata = get_pseudobulk(adata, sample_col, groups_col, min_cells=0, min_counts=0, mode='mean')
    assert pdata.shape[0] == 3
    assert np.all(pdata.X[0] == np.array([3., 1., 2.]))
    pdata = get_pseudobulk(adata, sample_col, groups_col, min_cells=0, min_counts=0, mode='median')
    assert pdata.shape[0] == 3
    assert np.all(pdata.X[0] == np.array([2., 0., 2.]))
    pdata = get_pseudobulk(adata, sample_col, groups_col, min_cells=0, min_counts=0,
                           mode={'sum': sum, 'median': np.median, 'mean': np.mean})
    assert np.all(pdata.X == pdata.layers['sum'])
    assert pdata.layers['sum'] is not None
    assert pdata.layers['median'] is not None
    assert pdata.layers['mean'] is not None


def test_get_unq_dict():
    col = pd.Series(['C1', 'C1', 'C2', 'C3'], index=['S1', 'S2', 'S3', 'S4'])
    condition = 'C1'
    reference = 'C2'
    get_unq_dict(col, condition, reference)
    get_unq_dict(col, condition, 'rest')


def test_check_if_skip():
    grp = 'Group'
    condition_col = 'celltype'
    condition = 'C1'
    reference = 'C2'
    unq_dict = {'C1': 2}
    check_if_skip(grp, condition_col, condition, reference, unq_dict)
    unq_dict = {'C2': 2}
    check_if_skip(grp, condition_col, condition, reference, unq_dict)
    unq_dict = {'C1': 2, 'C2': 1}
    check_if_skip(grp, condition_col, condition, reference, unq_dict)
    unq_dict = {'C1': 1, 'C2': 2}
    check_if_skip(grp, condition_col, condition, reference, unq_dict)


def test_get_contrast():
    groups_col, condition_col = 'celltype', 'condition'
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    condition = 'Ds'
    reference = 'Ht'
    obs = pd.DataFrame([['C1', 'C1', 'C1', 'C1'], [condition, condition, reference, reference]],
                       columns=r, index=[groups_col, condition_col]).T
    adata = AnnData(df, obs=obs, dtype=np.float32)
    get_contrast(adata, groups_col, condition_col, condition, None)
    get_contrast(adata, groups_col, condition_col, condition, reference)
    get_contrast(adata, None, condition_col, condition, reference)
    with pytest.raises(ValueError):
        get_contrast(adata, groups_col, condition_col, condition, condition)
    obs = pd.DataFrame([['C1', 'C1', 'C1', 'C1'], [condition, condition, condition, reference]],
                       columns=r, index=[groups_col, condition_col]).T
    get_contrast(adata, groups_col, condition_col, condition, reference)


def test_get_top_targets():
    logFCs = pd.DataFrame([[3, 0, -3], [1, 2, -5]], index=['C1', 'C2'], columns=['G1', 'G2', 'G3'])
    pvals = pd.DataFrame([[.3, .02, .01], [.9, .1, .003]], index=['C1', 'C2'], columns=['G1', 'G2', 'G3'])
    contrast = 'C1'
    name = 'T1'
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 1], ['T2', 'G3', 1], ['T2', 'G4', 0.5]],
                       columns=['source', 'target', 'weight'])
    get_top_targets(logFCs, pvals, contrast, name=name, net=net, sign_thr=1, lFCs_thr=0.0, fdr_corr=True)
    with pytest.raises(ValueError):
        get_top_targets(logFCs, pvals, contrast, name=None, net=net, sign_thr=1, lFCs_thr=0.0, fdr_corr=True)
    get_top_targets(logFCs, pvals, contrast, name=None, net=None, sign_thr=1, lFCs_thr=0.0, fdr_corr=True)
    get_top_targets(logFCs, pvals, contrast, name=None, net=None, sign_thr=1, lFCs_thr=0.0, fdr_corr=False)


def test_format_contrast_results():
    logFCs = pd.DataFrame([[3, 0, -3], [1, 2, -5]], index=['C1', 'C2'], columns=['G1', 'G2', 'G3'])
    logFCs.name = 'contrast_logFCs'
    pvals = pd.DataFrame([[.3, .02, .01], [.9, .1, .003]], index=['C1', 'C2'], columns=['G1', 'G2', 'G3'])
    pvals.name = 'contrast_pvals'
    format_contrast_results(logFCs, pvals)


def test_get_filterbyexpr_inputs():
    samples = ['S1', 'S2']
    genes = ['G1', 'G2', 'G3']
    df = pd.DataFrame([
        [2, 2, 4],
        [4, 4, 0]
    ], index=samples, columns=genes)
    obs = pd.DataFrame([['S1'], ['S2']], index=samples, columns=['group'])
    aobs = pd.DataFrame([['C1'], ['C2']], index=samples, columns=['group'])
    adata = AnnData(df, obs=obs, dtype=np.float32)

    y, nobs, var_names = get_filterbyexpr_inputs(adata=adata, obs=None)
    assert np.all(y == adata.X)
    assert np.all(nobs['group'].values == obs['group'].values)
    assert np.all(nobs['group'].values != aobs['group'].values)
    assert np.all(var_names == adata.var_names)
    y, nobs, var_names = get_filterbyexpr_inputs(adata=adata, obs=aobs)
    assert np.all(nobs['group'].values != aobs['group'].values)
    y, nobs, var_names = get_filterbyexpr_inputs(adata=df, obs=None)
    assert np.all(y == df.values)
    assert 'group' not in nobs.columns
    assert np.all(var_names == df.columns)
    y, nobs, var_names = get_filterbyexpr_inputs(adata=df, obs=obs)
    assert np.all(nobs['group'].values == obs['group'].values)
    with pytest.raises(ValueError):
        y, nobs, var_names = get_filterbyexpr_inputs(adata=[''], obs=None)
    with pytest.raises(ValueError):
        y, nobs, var_names = get_filterbyexpr_inputs(adata=[''], obs=obs)


def test_get_min_sample_size():
    obs = pd.DataFrame([['S1'], ['S1'], ['S1'], ['S2'], ['S2']], columns=['group'])
    min_sample_size = get_min_sample_size(group=None, obs=obs, large_n=10, min_prop=0.7)
    assert min_sample_size == 5
    min_sample_size = get_min_sample_size(group=None, obs=obs, large_n=4, min_prop=0.5)
    assert min_sample_size == 4.5
    min_sample_size = get_min_sample_size(group='group', obs=obs, large_n=10, min_prop=0.7)
    assert min_sample_size == 2
    min_sample_size = get_min_sample_size(group=obs['group'], obs=obs, large_n=10, min_prop=0.7)
    assert min_sample_size == 2
    min_sample_size = get_min_sample_size(group=obs['group'].values, obs=obs, large_n=10, min_prop=0.7)
    assert min_sample_size == 2
    min_sample_size = get_min_sample_size(group=list(obs['group'].values), obs=obs, large_n=10, min_prop=0.7)
    assert min_sample_size == 2


def test_get_cpm_cutoff():
    cut = get_cpm_cutoff(lib_size=1e6, min_count=1)
    assert cut == 1.0
    cut = get_cpm_cutoff(lib_size=1e5, min_count=1)
    assert cut == 10.
    cut = get_cpm_cutoff(lib_size=[1e5, 1e5, 1e6], min_count=1)
    assert cut == 10.


def test_get_cpm():
    y = np.array([
        [2, 2, 4, 4, 8],
        [4, 4, 0, 2, 6]
    ])
    lib_size = np.sum(y, axis=1)

    cpm = get_cpm(y, lib_size=lib_size)
    assert np.all(cpm.ravel() == np.array([1e5, 1e5, 2e5, 2e5, 4e5, 25e4, 25e4, 0, 125e3, 375e3]))
    cpm = get_cpm(y, lib_size=1e6)
    assert np.all(cpm.ravel() == y.ravel())


def test_filter_by_expr():
    index = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    columns = ['G1', 'G2', 'G3', 'G4', 'G5']
    # empty, BG, SG
    df = pd.DataFrame([
        [0, 0, 8, 4, 1],
        [0, 0, 8, 2, 1],
        [0, 0, 8, 4, 1],
        [0, 0, 2, 4, 1],
        [0, 8, 0, 2, 1],
        [0, 4, 0, 8, 1],
        [0, 4, 0, 8, 1],
    ], index=index, columns=columns)
    obs = pd.DataFrame([
        ['A'],
        ['A'],
        ['A'],
        ['A'],
        ['B'],
        ['B'],
        ['B'],
    ], index=index, columns=['group'])
    adata = AnnData(df, obs=obs, dtype=np.float32)

    genes = filter_by_expr(adata, obs=None, group=None, lib_size=None, min_count=10,
                           min_total_count=15, large_n=10, min_prop=0.7)
    assert genes.size == 0.
    genes = filter_by_expr(adata, obs=None, group=None, lib_size=None, min_count=1,
                           min_total_count=15, large_n=10, min_prop=0.7)
    assert np.all(genes == np.array(['G4']))
    genes = filter_by_expr(adata, obs=None, group=None, lib_size=None, min_count=1,
                           min_total_count=7, large_n=10, min_prop=0.7)
    assert np.all(genes == np.array(['G4', 'G5']))
    genes = filter_by_expr(adata, obs=None, group='group', lib_size=None, min_count=3,
                           min_total_count=10, large_n=10, min_prop=0.7)
    assert np.all(genes == np.array(['G2', 'G3', 'G4']))
    genes = filter_by_expr(adata, obs=None, group='group', lib_size=7, min_count=3,
                           min_total_count=10, large_n=10, min_prop=0.7)
    assert np.all(genes == np.array(['G2', 'G3', 'G4']))
    genes = filter_by_expr(adata, obs=None, group='group', lib_size=None, min_count=1,
                           min_total_count=0, large_n=0, min_prop=0.1)
    assert np.all(genes == np.array(['G2', 'G3', 'G4', 'G5']))
    genes = filter_by_expr(adata, obs=None, group=None, lib_size=None, min_count=1,
                           min_total_count=0, large_n=0, min_prop=0.55)
    assert np.all(genes == np.array(['G3', 'G4', 'G5']))
    genes = filter_by_expr(adata, obs=None, group=None, lib_size=None, min_count=1,
                           min_total_count=15, large_n=0, min_prop=0.55)
    assert np.all(genes == np.array(['G3', 'G4']))


def test_filter_by_prop():
    index = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    columns = ['G1', 'G2', 'G3', 'G4', 'G5']
    # empty, BG, SG
    df = pd.DataFrame([
        [0, 0, 8, 4, 1],
        [0, 0, 8, 2, 1],
        [0, 0, 8, 4, 1],
        [0, 0, 2, 4, 1],
        [0, 8, 0, 2, 1],
        [0, 4, 0, 8, 1],
        [0, 4, 0, 8, 1],
    ], index=index, columns=columns)
    props = pd.DataFrame([
        [0, .0, .8, .4, .1],
        [0, .0, .8, .2, .1],
        [0, .0, .8, .4, .1],
        [0, .0, .2, .4, .1],
        [0, .8, .0, .2, .1],
        [0, .4, .0, .8, .1],
        [0, .4, .0, .8, .1],
    ], index=index, columns=columns).values
    obs = pd.DataFrame([
        ['A'],
        ['A'],
        ['A'],
        ['A'],
        ['B'],
        ['B'],
        ['B'],
    ], index=index, columns=['group'])
    adata = AnnData(df, obs=obs, layers={'psbulk_props': props}, dtype=np.float32)

    g = filter_by_prop(adata, min_prop=0.2, min_smpls=2)
    assert np.all(g == np.array(['G2', 'G3', 'G4']))
    g = filter_by_prop(adata, min_prop=0.2, min_smpls=5)
    assert np.all(g == np.array(['G4']))
    g = filter_by_prop(adata, min_prop=0.1, min_smpls=7)
    assert np.all(g == np.array(['G4', 'G5']))
    g = filter_by_prop(adata, min_prop=0.4, min_smpls=2)
    assert np.all(g == np.array(['G2', 'G3', 'G4']))
    g = filter_by_prop(adata, min_prop=0.1, min_smpls=1)
    assert np.all(g == np.array(['G2', 'G3', 'G4', 'G5']))
    g = filter_by_prop(adata, min_prop=0.8, min_smpls=2)
    assert np.all(g == np.array(['G3', 'G4']))


def test_rank_sources_groups():
    var = pd.DataFrame(index=['T1', 'T2', 'T3'])
    X = pd.DataFrame(
        [[3., 1., -3.],
         [5., 0., -1.],
         [-1., 1., 4.],
         [-4., 1., 6.],
         [-3., 0., 3.],
         [0., 5., 1.],
         [1., 4., 0.]],
        columns=var.index
    )
    obs = pd.DataFrame(
        [['A'],
         ['A'],
         ['B'],
         ['B'],
         ['B'],
         ['C'],
         ['C']], columns=['group'], index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
        )
    acts = AnnData(X, obs=obs, var=var, dtype=np.float32)
    gt_up = np.array([True, False, False, True, False, False, True, False, False])
    gt_dw = np.array([False, False, True, False, False, True, False, False, False])

    df = rank_sources_groups(acts, groupby='group', reference='rest',
                             method='wilcoxon')
    assert df.shape[0] == 9
    assert np.all((df['statistic'].values > 1.7) == gt_up)
    assert np.all((df['statistic'].values < -1.7) == gt_dw)
    df = rank_sources_groups(acts, groupby='group', reference='rest',
                             method='t-test')
    assert df.shape[0] == 9
    assert np.all((df['statistic'].values > 1.7) == gt_up)
    assert np.all((df['statistic'].values < -1.7) == gt_dw)
    df = rank_sources_groups(acts, groupby='group', reference='rest',
                             method='t-test_overestim_var')
    assert df.shape[0] == 9
    assert np.all((df['statistic'].values > 1.7) == gt_up)
    assert np.all((df['statistic'].values < -1.7) == gt_dw)

    df = rank_sources_groups(acts, groupby='group', reference='A',
                             method='t-test')
    assert df.shape[0] == 9
    assert np.all(df['statistic'][:3] == 0.)
    assert np.all(df['statistic'][3:] != 0.)

    df = rank_sources_groups(acts, groupby='group', reference=['A', 'B', 'C'],
                             method='t-test')
    assert df.shape[0] == 9
    assert np.all((df['statistic'].values > 1.7) == gt_up)
    assert np.all((df['statistic'].values < -1.7) == gt_dw)

    with pytest.raises(ValueError):
        rank_sources_groups(acts, groupby='group', reference=['A', 'B', 'C'],
                            method='asdadssad')
    with pytest.raises(AssertionError):
        rank_sources_groups(acts, groupby='group', reference='asdadssad',
                            method='t-test')
    acts.X[0, 0] = np.nan
    acts.X[1, 1] = np.inf
    with pytest.raises(AssertionError):
        rank_sources_groups(acts, groupby='group', reference='A',
                             method='t-test')
