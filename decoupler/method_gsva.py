"""
Method GSVA.
Code to run the Gene Set Variation Analysis (GSVA) method.
"""

import numpy as np
import pandas as pd

from scipy.stats import norm
from numpy.random import default_rng

from .pre import extract, rename_net, filt_min_n
from .method_gsea import std

from anndata import AnnData
from tqdm import tqdm

import numba as nb


def init_cdfs(pre_res=10000, max_pre=10):
    pre_cdf = norm.cdf(np.arange(pre_res+1) * max_pre / pre_res, loc=0, scale=1).astype(np.float32)

    return pre_cdf


@nb.njit(nb.f4[:](nb.f4[:]), cache=True)
def apply_ecdf(x):
    v = np.sort(x)
    n = len(x)
    return (np.searchsorted(v, x, side='right').astype(nb.f4)) / n


@nb.njit(nb.f4[:, :](nb.f4[:, :]), parallel=True, cache=True)
def mat_ecdf(mat):
    for j in nb.prange(mat.shape[1]):
        mat[:, j] = apply_ecdf(mat[:, j])
    return mat


@nb.njit(nb.f4[:](nb.f4[:], nb.f4[:]), cache=True)
def col_d(x, pre_cdf):
    size = x.shape[0]
    bw = (std(x, 1) / 4.0)
    col = np.zeros(size, dtype=nb.f4)
    for j in nb.prange(size):
        left_tail = 0.0
        for i in nb.prange(size):
            diff = (x[j] - x[i]) / bw
            if diff < -10:
                left_tail += 0.0
            elif diff > 10:
                left_tail += 1.0
            else:
                cdf_val = pre_cdf[int(np.abs(diff) / 10 * 10000)]
                if diff < 0:
                    left_tail += 1.0 - cdf_val
                else:
                    left_tail += cdf_val
        left_tail = left_tail/size
        col[j] = -1.0 * np.log((1.0-left_tail)/left_tail)
    return col


@nb.njit(nb.f4[:, :](nb.f4[:, :], nb.f4[:]), parallel=True, cache=True)
def mat_d(mat, pre_cdf):
    D = np.zeros(mat.shape, dtype=nb.f4)
    for j in nb.prange(mat.shape[1]):
        D[:, j] = col_d(mat[:, j], pre_cdf)
    return D


def density(mat, kcdf=False):
    if kcdf:
        pre_cdf = init_cdfs()
        mat = mat_d(mat, pre_cdf)
    else:
        mat = mat_ecdf(mat)
    return mat


@nb.njit(nb.types.Tuple((nb.f4[:, :], nb.i4[:, :]))(nb.f4[:, :]), parallel=True)
def nb_get_D_I(mat):
    n = mat.shape[1]
    rev_idx = np.abs(np.arange(start=n, stop=0, step=-1, dtype=nb.f4) - n / 2)
    Idx = np.zeros(mat.shape, dtype=nb.i4)
    for i in nb.prange(mat.shape[0]):
        Idx[i] = np.argsort(-mat[i])
        tmp = np.zeros(n, dtype=nb.f4)
        tmp[Idx[i]] = rev_idx
        mat[i] = tmp
    return mat, Idx


@nb.njit(nb.f4(nb.f4[:], nb.i4[:], nb.i4, nb.i4[:], nb.i4[:], nb.i4, nb.f4))
def ks_sample(D, Idx, n_genes, geneset_mask, fset, n_geneset, dec):

    sum_gset = 0.0
    for i in nb.prange(n_geneset):
        sum_gset += D[fset[i]]

    mx_value_sign = 0.0
    cum_sum = 0.0
    mx_pos = 0.0
    mx_neg = 0.0

    for i in nb.prange(n_genes):
        idx = Idx[i]
        if geneset_mask[idx] == 1:
            cum_sum += D[idx] / sum_gset
        else:
            cum_sum -= dec

        if cum_sum > mx_pos:
            mx_pos = cum_sum
        if cum_sum < mx_neg:
            mx_neg = cum_sum

    mx_value_sign = mx_pos + mx_neg

    return mx_value_sign


@nb.njit(nb.f4[:](nb.f4[:, :], nb.i4[:, :], nb.i4[:]), parallel=True)
def ks_matrix(D, Idx, fset):
    n_samples, n_genes = D.shape
    n_geneset = fset.shape[0]

    geneset_mask = np.zeros(n_genes, dtype=nb.i4)
    geneset_mask[fset] = 1

    dec = 1.0 / (n_genes - n_geneset)

    res = np.zeros(n_samples, dtype=nb.f4)
    for i in nb.prange(n_samples):
        res[i] = ks_sample(D[i], Idx[i], n_genes, geneset_mask, fset, n_geneset, dec)

    return res


def gsva(mat, net, kcdf=False, verbose=False):

    # Get feature Density
    mat = density(mat, kcdf=kcdf)
    mat, Idx = nb_get_D_I(mat)

    # Run GSVA for each feature set
    acts = np.zeros((mat.shape[0], len(net)))
    for j in tqdm(range(len(net)), disable=not verbose):
        fset = net.iloc[j]
        acts[:, j] = ks_matrix(mat, Idx, fset)

    return acts


def run_gsva(mat, net, source='source', target='target', kcdf=False, mx_diff=True, abs_rnk=False, min_n=5, seed=42,
             verbose=False, use_raw=True):
    """
    Gene Set Variation Analysis (GSVA).

    GSVA (Hänzelmann et al., 2013) starts by transforming the input molecular readouts in `mat` to a readout-level statistic
    using Gaussian kernel estimation of the cumulative density function. Then, readout-level statistics are ranked per sample
    and normalized to up-weight the two tails of the rank distribution. Afterwards, an enrichment score `gsva_estimate` is
    calculated using a running sum statistic that is normalized by subtracting the largest negative estimate from the largest
    positive one.

    Hänzelmann S. et al. (2013) GSVA: gene set variation analysis for microarray and RNA-seq data. BMC Bioinformatics, 14, 7.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    kcdf : bool
        Wether to use a Gaussian kernel or not during the non-parametric estimation of the cumulative distribution function.
        By default no kernel is used (faster), to reproduce GSVA original behaviour in R set to True.
    mx_diff : bool
        Changes how the enrichment statistic (ES) is calculated. If True (default), ES is calculated as the difference between
        the maximum positive and negative random walk deviations. If False, ES is calculated as the maximum positive to 0.
    abs_rnk : bool
        Used when mx_diff = True. If False (default), the enrichment statistic (ES) is calculated taking the magnitude
        difference between the largest positive and negative random walk deviations. If True, feature sets with features
        enriched on either extreme (high or low) will be regarded as 'highly' activated.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.

    Returns
    -------
    estimate : DataFrame
        GSVA scores. Stored in `.obsm['gsva_estimate']` if `mat` is AnnData.
    """

    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Transform net
    net = rename_net(net, source=source, target=target, weight=None)
    net = filt_min_n(c, net, min_n=min_n)

    # Randomize feature order to break ties randomly
    rng = default_rng(seed=seed)
    idx = np.arange(m.shape[1])
    rng.shuffle(idx)
    m, c = m[:, idx], c[idx]

    # Transform targets to indxs
    table = {name: i for i, name in enumerate(c)}
    net['target'] = [table[target] for target in net['target']]
    net = net.groupby('source')['target'].apply(lambda x: np.array(x, dtype=np.int32))

    if verbose:
        print('Running gsva on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), len(net)))

    # Run GSVA
    estimate = gsva(m.A, net, kcdf=kcdf, verbose=verbose)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = 'gsva_estimate'

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
    else:
        return estimate
