import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..utils_anndata import get_acts, extract_psbulk_inputs, check_for_raw_counts, format_psbulk_inputs
from ..utils_anndata import compute_psbulk, get_unq_dict, get_pseudobulk, check_if_skip, get_contrast
from ..utils_anndata import get_top_targets, format_contrast_results


def test_get_acts():
    m = np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]])
    r = np.array(['S1', 'S2', 'S3'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    estimate = pd.DataFrame([[3.5, -0.5, 0.3], [3.6, -0.6, 0.04], [-1, 2, -1.8]],
                            columns=['T1', 'T2', 'T3'], index=r)
    adata = AnnData(df)
    adata.obsm['estimate'] = estimate
    get_acts(adata, 'estimate')


def test_extract_psbulk_inputs():
    m = np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]])
    r = np.array(['S1', 'S2', 'S3'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    obs = pd.DataFrame([['C01', 'C01', 'C02']], columns=r, index=['celltype']).T
    adata = AnnData(df, obs=obs)
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


def test_check_for_raw_counts():
    X = csr_matrix(np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]]))
    X_float = csr_matrix(np.array([[1.3, 0, 2.1], [1.48, 0.123, 3.33], [0, 0, 0]]))
    X_neg = csr_matrix(np.array([[1, 0, -2], [1, 0, -3], [0, 0, 0]]))
    X_inf = csr_matrix(np.array([[1, 0, np.nan], [1, 0, 3], [0, 0, 0]]))
    check_for_raw_counts(X)
    with pytest.raises(ValueError):
        check_for_raw_counts(X_float)
    with pytest.raises(ValueError):
        check_for_raw_counts(X_neg)
    with pytest.raises(ValueError):
        check_for_raw_counts(X_inf)


def test_format_psbulk_inputs():
    sample_col, groups_col = 'sample_id', 'celltype'
    obs = pd.DataFrame([['S1', 'S2', 'S3'], ['C1', 'C1', 'C2']], columns=['S1', 'S2', 'S3'], index=[sample_col, groups_col]).T
    format_psbulk_inputs(sample_col, groups_col, obs)
    format_psbulk_inputs(sample_col, None, obs)


def test_compute_psbulk():
    sample_col, groups_col = 'sample_id', 'celltype'
    min_cells, min_counts, min_prop = 0., 0., 0.
    X = csr_matrix(np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]]))
    smples = np.array(['S1', 'S1', 'S2'])
    groups = np.array(['C1', 'C1', 'C1'])
    n_rows = len(smples)
    n_cols = X.shape[1]
    psbulk = np.zeros((n_rows, n_cols))
    props = np.full((n_rows, n_cols), False)
    obs = pd.DataFrame([smples, groups], columns=smples, index=[sample_col, groups_col]).T
    new_obs = pd.DataFrame(columns=obs.columns)
    compute_psbulk(psbulk, props, X, sample_col, groups_col, np.unique(smples),
                   np.unique(groups), obs, new_obs, min_cells, min_counts, min_prop)
    compute_psbulk(psbulk, props, X, sample_col, None, np.unique(smples),
                   np.unique(groups), obs, new_obs, min_cells, min_counts, min_prop)


def test_get_pseudobulk():
    sample_col, groups_col = 'sample_id', 'celltype'
    m = np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]])
    r = np.array(['S1', 'S2', 'S3'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    smples = np.array(['S1', 'S2', 'S3'])
    groups = np.array(['C1', 'C1', 'C2'])
    obs = pd.DataFrame([smples, groups], columns=smples, index=[sample_col, groups_col]).T
    adata = AnnData(df, obs=obs)
    get_pseudobulk(adata, sample_col, sample_col, min_prop=0, min_cells=0, min_counts=0, min_smpls=0)
    get_pseudobulk(adata, sample_col, groups_col, min_prop=0, min_cells=0, min_counts=0, min_smpls=0)


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
    adata = AnnData(df, obs=obs)
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
