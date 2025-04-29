from typing import Tuple
import math

import numpy as np
import scipy.stats as sts
import scipy.sparse as sps
from tqdm.auto import tqdm
import numba as nb

from decoupler._log import _log
from decoupler._Method import MethodMeta, Method
from decoupler.pp.net import _getset
from decoupler.mt._gsea import _std


@nb.njit(cache=True)
def _erf(
    x: np.ndarray,
) -> np.ndarray:
    a1, a2, a3, a4, a5, a6 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    sign = np.sign(x)
    abs_x = np.abs(x)
    t = 1.0 / (1.0 + a6 * abs_x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-abs_x * abs_x))
    return sign * y


@nb.njit(cache=True)
def _norm_cdf(
    x: np.ndarray,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> np.ndarray:
    e = _erf((x - mu) / (sigma * np.sqrt(2.0)))
    return (0.5 * (1.0 + e))


@nb.njit(cache=True)
def _poisson_pmf(
    k: float,
    lam: float,
) -> float:
    if k < 0 or lam < 0:
        return 0.0
    if k == 0:
        return np.exp(-lam)
    log_pmf = -lam + k * np.log(lam) - math.lgamma(k + 1)
    return np.exp(log_pmf)


@nb.njit(cache=True)
def _ppois(
    k: float,
    lam: float
) -> float:
    cdf_sum = 0.0
    for i in range(int(k) + 1):
        cdf_sum += _poisson_pmf(i, lam)
    if cdf_sum > 1:
        cdf_sum = 1.
    return cdf_sum


@nb.njit(cache=True)
def _init_cdfs(
) -> np.ndarray:
    pre_res = 10000
    max_pre = 10
    pre_cdf = _norm_cdf(np.arange(pre_res + 1) * max_pre / pre_res, 0, 1)
    return pre_cdf


@nb.njit(cache=True)
def _apply_ecdf(
    x: np.ndarray
) -> np.ndarray:
    v = np.sort(x)
    n = len(x)
    return np.searchsorted(v, x, side='right') / n


@nb.njit(parallel=True, cache=True)
def _mat_ecdf(
    mat: np.ndarray
) -> np.ndarray:
    D = mat.copy()
    for j in nb.prange(mat.shape[1]):
        D[:, j] = _apply_ecdf(mat[:, j])
    return D


@nb.njit(cache=True)
def _col_d(
    x: np.ndarray,
    gauss: bool,
    pre_cdf: np.ndarray
) -> np.ndarray:
    size = x.shape[0]
    if gauss:
        bw = _std(x, 1) / 4.0
    else:
        bw = 0.5
    col = np.zeros(size)
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
                left_tail += _ppois(x[j], x[i] + bw)
        left_tail = left_tail / size
        col[j] = -1.0 * np.log((1.0 - left_tail) / left_tail)
    return col


@nb.njit(parallel=True, cache=True)
def _mat_d(
    mat: np.ndarray,
    gauss: bool
) -> np.ndarray:
    pre_cdf = _init_cdfs()
    D = np.zeros(mat.shape)
    for j in nb.prange(mat.shape[1]):
        D[:, j] = _col_d(mat[:, j], gauss, pre_cdf)
    return D


def _density(
    mat: np.ndarray,
    kcdf: str | None,
) -> np.ndarray:
    assert isinstance(kcdf, str) or kcdf is None, \
    'kcdf must be gaussian, poisson or None'
    if kcdf == 'gaussian':
        mat = _mat_d(mat, gauss=True)
    elif kcdf == 'poisson':
        mat = _mat_d(mat, gauss=False)
    else:
        mat = _mat_ecdf(mat)
    return mat


@nb.njit(parallel=True, cache=True)
def _order_rankstat(
    mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = mat.shape
    ord_mat = np.zeros((n_rows, n_cols), dtype=np.int_)
    rst_mat = np.zeros((n_rows, n_cols))
    for i in nb.prange(n_rows):
        ord = np.argsort(-mat[i, :]) + 1
        rst = np.zeros(n_cols)
        for j in range(n_cols):
            rst[ord[j] - 1] = abs(n_cols - j - (n_cols // 2))
        ord_mat[i, :] = ord
        rst_mat[i, :] = rst
    return ord_mat, rst_mat


@nb.njit(cache=True)
def _rnd_walk(
    gsetidx: np.ndarray,
    k: int,
    generanking: np.ndarray,
    rankstat: np.ndarray,
    n: int,
) -> Tuple[float, int]:
    stepcdfingeneset = np.zeros(n, dtype=np.int_)
    stepcdfoutgeneset = np.ones(n, dtype=np.int_)
    for i in range(k):
        idx = gsetidx[i] - 1
        stepcdfingeneset[idx] = rankstat[generanking[idx] - 1]
        stepcdfoutgeneset[idx] = 0
    for i in range(1, n):
        stepcdfingeneset[i] += stepcdfingeneset[i-1]
        stepcdfoutgeneset[i] += stepcdfoutgeneset[i-1]
    walkstatpos = -np.inf
    walkstatneg = np.inf
    walkstat = np.zeros(n)
    for i in range(n):
        wlkstat = (stepcdfingeneset[i] / stepcdfingeneset[-1]) - (stepcdfoutgeneset[i] / stepcdfoutgeneset[-1])
        walkstat[i] = wlkstat
        if wlkstat > walkstatpos:
            walkstatpos = wlkstat
        if wlkstat < walkstatneg:
            walkstatneg = wlkstat
    return walkstatpos, walkstatneg


@nb.njit(cache=True)
def _score_geneset(
    gsetidx: np.ndarray,
    generanking: np.ndarray,
    rankstat: np.ndarray,
    maxdiff: bool,
    absrnk: bool,
) -> float:
    n = len(generanking)
    k = len(gsetidx)
    walkstatpos, walkstatneg = _rnd_walk(gsetidx, k, generanking, rankstat, n)
    if maxdiff:
        if absrnk:
            es = walkstatpos - walkstatneg
        else:
            es = walkstatpos + walkstatneg
    else:
        es = walkstatpos if abs(walkstatpos) > abs(walkstatneg) else walkstatneg
    return es


@nb.njit(cache=True)
def _match(
    a: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    max_b = np.max(b) if len(b) > 0 else 0
    index_array = np.full(max_b + 1, -1)
    for idx, value in enumerate(b):
        if 0 <= value <= max_b:
            index_array[value] = idx
    result = np.full(len(a), -1)
    for i in range(len(a)):
        if 0 <= a[i] <= max_b:
            result[i] = index_array[a[i]]
    return result + 1


@nb.njit(parallel=True, cache=True)
def _ks_fset(
    ordr: np.ndarray,
    rst: np.ndarray,
    fset: np.ndarray,
    maxdiff: bool,
    absrnk: bool,
) -> np.ndarray:
    n_samples, n_genes = ordr.shape
    res = np.zeros(n_samples)
    for i in range(n_samples): #nb.prange
        generanking = ordr[i]
        rankstat = rst[i]
        genesetsrankidx = _match(fset, generanking)
        res[i] = _score_geneset(genesetsrankidx, generanking, rankstat, maxdiff, absrnk)
    return res


def _func_gsva(
    mat: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    kcdf: str | None = 'gaussian',
    maxdiff: bool = True,
    absrnk: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, None]:
    if isinstance(mat, sps.csr_matrix):
        m = f'gsva - Converting sparse matrix to dense format before density transformation'
        _log(m, level='info', verbose=verbose)
        mat = mat.toarray()
    m = f'gsva - computing density with kcdf={kcdf}'
    _log(m, level='info', verbose=verbose)
    # Compute density
    if mat.shape[0] > 1:
        mat = _density(mat, kcdf=kcdf)
    ordr, rst = _order_rankstat(mat)
    # Compute GSVA
    nsrc = starts.size
    m = f'gsva - calculating {nsrc} scores with maxdiff={maxdiff}, absrnk={absrnk}'
    _log(m, level='info', verbose=verbose)
    es = np.zeros((ordr.shape[0], nsrc))
    for j in tqdm(range(nsrc), disable=not verbose):
        fset = (_getset(cnct, starts, offsets, j) + 1).astype(int)
        es[:, j] = _ks_fset(ordr, rst, fset, maxdiff, absrnk)
    return es, None


params = """\
kcdf
    Which kernel to use during the non-parametric estimation of the cumulative distribution function.
    Options are gaussian, poisson or None.
mx_diff
    Changes how the enrichment statistic (ES) is calculated. If ``True`` (default), ES is calculated as the difference between
    the maximum positive and negative random walk deviations. If ``False``, ES is calculated as the maximum positive to 0.
abs_rnk : bool
    Used when ``mx_diff=True``. If ``False`` (default), the enrichment statistic (ES) is calculated taking the magnitude
    difference between the largest positive and negative random walk deviations. If ``True``, feature sets with features
    enriched on either extreme (high or low) will be regarded as 'highly' activated."""

_gsva = MethodMeta(
    name='gsva',
    func=_func_gsva,
    stype='numerical',
    adj=False,
    weight=False,
    test=False,
    limits=(-1, +1),
    reference='https://doi.org/10.1186/1471-2105-14-7',
    params=params,
)
gsva = Method(_method=_gsva)
