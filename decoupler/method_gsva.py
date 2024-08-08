"""
Method GSVA.
Code to run the Gene Set Variation Analysis (GSVA) method.
"""

import numpy as np
import pandas as pd

from scipy.sparse import issparse
import math

from .pre import extract, rename_net, filt_min_n, return_data, break_ties

from tqdm.auto import tqdm

import numba as nb


@nb.njit(nb.f8(nb.f8[:], nb.i8), cache=True, error_model='numpy')
def std(arr, ddof):
    N = arr.shape[0]
    m = np.mean(arr)
    var = np.sum((arr - m)**2) / (N - ddof)
    sd = np.sqrt(var)
    return sd


@nb.njit(nb.f8[:](nb.f8[:]), cache=True)
def erf(x):
    a1, a2, a3, a4, a5, a6 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    sign = np.sign(x)
    abs_x = np.abs(x)
    t = 1.0 / (1.0 + a6 * abs_x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-abs_x * abs_x))
    return sign * y


@nb.njit(nb.f8[:](nb.f8[:], nb.f8, nb.f8), cache=True)
def norm_cdf(x, mu=0.0, sigma=1.0):
    e = erf((x - mu) / (sigma * np.sqrt(2.0)))
    return (0.5 * (1.0 + e))


@nb.njit(nb.f8(nb.f8, nb.f8), cache=True)
def poisson_pmf(k, lam):
    if k < 0 or lam < 0:
        return 0.0
    if k == 0:
        return np.exp(-lam)

    log_pmf = -lam + k * np.log(lam) - math.lgamma(k + 1)
    return np.exp(log_pmf)


@nb.njit(nb.f8(nb.f8, nb.f8), cache=True)
def ppois(k, lam):
    cdf_sum = 0.0
    for i in range(int(k) + 1):
        cdf_sum += poisson_pmf(i, lam)
    if cdf_sum > 1:
        cdf_sum = 1.
    return cdf_sum


@nb.njit(nb.f8[:](), cache=True)
def init_cdfs():
    pre_res = 10000
    max_pre = 10
    pre_cdf = norm_cdf(np.arange(pre_res + 1) * max_pre / pre_res, 0, 1)
    return pre_cdf


@nb.njit(nb.f8[:](nb.f8[:]), cache=True)
def apply_ecdf(x):
    v = np.sort(x)
    n = len(x)
    return np.searchsorted(v, x, side='right') / n


@nb.njit(nb.f8[:, :](nb.f8[:, :]), parallel=True, cache=True)
def mat_ecdf(mat):
    D = mat.copy()
    for j in nb.prange(mat.shape[1]):
        D[:, j] = apply_ecdf(mat[:, j])
    return D


@nb.njit(nb.f8[:](nb.f8[:], nb.b1, nb.f8[:]), cache=True)
def col_d(x, gauss, pre_cdf):
    size = x.shape[0]
    if gauss:
        bw = std(x, 1) / 4.0
    else:
        bw = 0.5
    col = np.zeros(size, dtype=nb.f8)
    for j in range(size):
        left_tail = 0.0
        for i in range(size):
            if gauss:
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
            else:
                left_tail += ppois(x[j], x[i] + bw)
        left_tail = left_tail/size
        col[j] = -1.0 * np.log((1.0-left_tail)/left_tail)
    return col


@nb.njit(nb.f8[:, :](nb.f8[:, :], nb.b1), parallel=True, cache=True)
def mat_d(mat, gauss):
    pre_cdf = init_cdfs()
    D = np.zeros(mat.shape, dtype=nb.f8)
    for j in nb.prange(mat.shape[1]):
        D[:, j] = col_d(mat[:, j], gauss, pre_cdf)
    return D


def density(mat, kcdf):
    mat = mat.astype(float)
    if kcdf is None:
        mat = mat_ecdf(mat)
    else:
        if kcdf == 'gaussian':
            gauss = True
        elif kcdf == 'poisson':
            gauss = False
        else:
            raise ValueError("kcdf needs to be either 'gaussian', 'poisson', or None")
        mat = mat_d(mat, gauss=gauss)

    return mat


@nb.njit(nb.types.Tuple((nb.i8[:, :], nb.i8[:, :]))(nb.f8[:, :]), parallel=True, cache=True)
def order_rankstat(mat):
    n_rows, n_cols = mat.shape
    ord_mat = np.zeros((n_rows, n_cols), dtype=nb.i8)
    rst_mat = np.zeros((n_rows, n_cols), dtype=nb.i8)
    for i in range(n_rows):
        ord = np.argsort(-mat[i, :]) + 1
        rst = np.zeros(n_cols, dtype=nb.i8)
        for j in range(n_cols):
            rst[ord[j] - 1] = abs(n_cols - j - (n_cols // 2))
        ord_mat[i, :] = ord
        rst_mat[i, :] = rst
    return ord_mat, rst_mat


@nb.njit(nb.types.UniTuple(nb.f8, 2)(nb.i8[:], nb.i8, nb.i8[:], nb.i8[:], nb.i8), cache=True)
def rnd_walk(gsetidx, k, generanking, rankstat, n):
    stepcdfingeneset = np.zeros(n, dtype=np.int32)
    stepcdfoutgeneset = np.ones(n, dtype=np.int32)
    for i in range(k):
        idx = gsetidx[i] - 1
        stepcdfingeneset[idx] = rankstat[generanking[idx] - 1]
        stepcdfoutgeneset[idx] = 0
    for i in range(1, n):
        stepcdfingeneset[i] += stepcdfingeneset[i-1]
        stepcdfoutgeneset[i] += stepcdfoutgeneset[i-1]
    walkstatpos = -np.inf
    walkstatneg = np.inf
    walkstat = np.zeros(n, dtype=np.float64)
    for i in range(n):
        wlkstat = (stepcdfingeneset[i] / stepcdfingeneset[-1]) - (stepcdfoutgeneset[i] / stepcdfoutgeneset[-1])
        walkstat[i] = wlkstat
        if wlkstat > walkstatpos:
            walkstatpos = wlkstat
        if wlkstat < walkstatneg:
            walkstatneg = wlkstat
    return walkstatpos, walkstatneg


@nb.njit(nb.f8(nb.i8[:], nb.i8[:], nb.i8[:], nb.b1, nb.b1), cache=True)
def score_geneset(gsetidx, generanking, rankstat, maxdiff, absrnk):
    n = len(generanking)
    k = len(gsetidx)
    walkstatpos, walkstatneg = rnd_walk(gsetidx, k, generanking, rankstat, n)
    if maxdiff:
        if absrnk:
            es = walkstatpos - walkstatneg
        else:
            es = walkstatpos + walkstatneg
    else:
        es = walkstatpos if abs(walkstatpos) > abs(walkstatneg) else walkstatneg
    return es


@nb.njit(nb.i8[:](nb.i8[:], nb.i8[:]), cache=True)
def match(a, b):
    max_b = np.max(b) if len(b) > 0 else 0
    index_array = np.full(max_b + 1, -1, dtype=nb.i8)
    for idx, value in enumerate(b):
        if 0 <= value <= max_b:
            index_array[value] = idx
    result = np.full(len(a), -1, dtype=nb.i8)
    for i in range(len(a)):
        if 0 <= a[i] <= max_b:
            result[i] = index_array[a[i]]
    return result + 1


@nb.njit(nb.f8[:](nb.i8[:, :], nb.i8[:, :], nb.i8[:], nb.b1, nb.b1), parallel=True, cache=True)
def ks_fset(ord, rst, fset, maxdiff, absrnk):
    n_samples, n_genes = ord.shape
    res = np.zeros(n_samples, dtype=nb.f8)
    for i in nb.prange(n_samples):
        generanking = ord[i]
        rankstat = rst[i]
        genesetsrankidx = match(fset, generanking)
        res[i] = score_geneset(genesetsrankidx, generanking, rankstat, maxdiff, absrnk)
    return res


def gsva(mat, net, kcdf=False, maxdiff=True, absrnk=False, verbose=False):

    if issparse(mat):
        mat = mat.toarray()
    # Get feature Density
    mat = density(mat, kcdf=kcdf)
    ord, rst = order_rankstat(mat)

    # Run GSVA for each feature set
    acts = np.zeros((ord.shape[0], len(net)))
    for j in tqdm(range(len(net)), disable=not verbose):
        fset = net.iloc[j] + 1
        acts[:, j] = ks_fset(ord, rst, fset, maxdiff, absrnk)

    return acts


def run_gsva(mat, net, source='source', target='target', kcdf='gaussian', mx_diff=True, abs_rnk=False, min_n=5, seed=42,
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
        Whether to use a Gaussian kernel or not during the non-parametric estimation of the cumulative distribution function.
        To reproduce GSVA original behaviour in R set to True.
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

    # Remove repeated features
    if issparse(m):
        m = m.toarray()
    if m.shape[0] > 1:
        msk = ~np.all(m == m[0, :], axis=0)
        m = m[:, msk]
        c = c[msk]

    # Transform net
    net = rename_net(net, source=source, target=target, weight=None)
    net = filt_min_n(c, net, min_n=min_n)

    # Randomize feature order to break ties randomly
    m, c = break_ties(m, c, seed)

    # Transform targets to indxs
    table = {name: i for i, name in enumerate(c)}
    net['target'] = [table[target] for target in net['target']]
    net = net.groupby('source', observed=True)['target'].apply(lambda x: np.array(x, dtype=np.int64))

    if verbose:
        print('Running gsva on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), len(net)))

    # Run GSVA
    estimate = gsva(m, net, kcdf=kcdf, maxdiff=mx_diff, absrnk=abs_rnk, verbose=verbose)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = 'gsva_estimate'

    return return_data(mat=mat, results=(estimate, ))
