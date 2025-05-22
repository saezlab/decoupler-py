from typing import Tuple
import math

import numpy as np
import scipy.stats as sts
import scipy.sparse as sps
from tqdm.auto import tqdm
import numba as nb

from decoupler._docs import docs
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
def _ecdf(arr):
    ecdf = np.searchsorted(np.sort(arr), arr, side='right') / len(arr)
    return ecdf


@nb.njit(parallel=True, cache=True)
def _mat_ecdf(
    mat: np.ndarray
) -> np.ndarray:
    D = np.zeros(mat.shape)
    for j in range(mat.shape[1]):
        D[:, j] = _ecdf(mat[:, j])
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
    assert (isinstance(kcdf, str) and kcdf in ['gaussian', 'poisson']) or kcdf is None, \
    'kcdf must be gaussian, poisson or None'
    if kcdf == 'gaussian':
        mat = _mat_d(mat, gauss=True)
    elif kcdf == 'poisson':
        assert mat.sum().is_integer(), \
        f'when kcdf={kcdf} input data must be integers (e.g. 3, 4, etc.), not decimal values (e.g. 3.5, 4.9, etc.)'
        mat = _mat_d(mat, gauss=False)
    elif kcdf is None:
        mat = _mat_ecdf(mat)
    return mat


@nb.njit(cache=True)
def _rankdata(values):
    n = len(values)
    ranks = np.empty(n, dtype=np.int_)
    indices = np.arange(n)
    sorted_indices = np.empty(n, dtype=np.int_)
    sorted_values = np.empty(n, dtype=values.dtype)
    for i in range(n):
        sorted_indices[i] = indices[i]
        sorted_values[i] = values[i]
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = sorted_values[i], sorted_values[j]
            ii, ij = sorted_indices[i], sorted_indices[j]
            if (vj < vi) or (vj == vi and ij > ii):
                sorted_values[i], sorted_values[j] = vj, vi
                sorted_indices[i], sorted_indices[j] = ij, ii
    for rank, idx in enumerate(sorted_indices, 1):
        ranks[idx] = rank
    return ranks


@nb.njit(cache=True)
def _dos_srs(r):
    mask = (r == 0)
    p = len(r)
    r_dense = r.astype(np.int_).copy()
    if mask.any():
        nzs = mask.sum()
        r_dense[~mask] += nzs
        cnt = 1
        for i in range(p):
            if mask[i]:
                r_dense[i] = cnt
                cnt += 1
    dos = p - r_dense + 1
    srs = np.empty(p)
    if mask.any():
        r_mod = r.copy()
        for i in range(p):
            if mask[i]:
                r_mod[i] = 1
            else:
                r_mod[i] += 1
        max_r = np.max(r_mod)
        half_max = max_r / 2
        for i in range(p):
            srs[i] = abs(half_max - r_mod[i])
    else:
        half_p = p / 2
        for i in range(p):
            srs[i] = abs(half_p - r_dense[i])
    return dos, srs


@nb.njit(parallel=True, cache=True)
def _rankmat(
    mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = mat.shape
    dos_mat = np.zeros((n_rows, n_cols), dtype=np.int_)
    srs_mat = np.zeros((n_rows, n_cols), dtype=np.int_)
    for i in nb.prange(n_rows):
        r = _rankdata(mat[i, :])
        dos_mat[i, :], srs_mat[i, :] = _dos_srs(r)
    return dos_mat, srs_mat


@nb.njit(cache=True)
def _rnd_walk(
    gsetidx: np.ndarray,
    k: int,
    decordstat: np.ndarray,
    symrnkstat: np.ndarray,
    n: int,
    tau: int | float,
) -> Tuple[float, int]:
    gsetrnk = np.empty(k, dtype=np.int_)
    for i in range(k):
        gsetrnk[i] = decordstat[gsetidx[i] - 1]
    stepcdfingeneset = np.zeros(n)
    stepcdfoutgeneset = np.ones(n, dtype=np.int_)
    for i in range(k):
        idx = gsetrnk[i] - 1
        if tau == 1.0:
            stepcdfingeneset[idx] = symrnkstat[gsetidx[i] - 1]
        else:
            stepcdfingeneset[idx] = symrnkstat[gsetidx[i] - 1] ** tau
        stepcdfoutgeneset[idx] = 0
    # cumulative sums
    for i in range(1, n):
        stepcdfingeneset[i] += stepcdfingeneset[i - 1]
        stepcdfoutgeneset[i] += stepcdfoutgeneset[i - 1]
    walkstatpos = -np.inf
    walkstatneg = np.inf
    if stepcdfingeneset[n - 1] > 0 and stepcdfoutgeneset[n - 1] > 0:
        walkstatpos = 0.0
        walkstatneg = 0.0
        for i in range(n):
            wlkstat = (stepcdfingeneset[i] / stepcdfingeneset[n - 1]) - \
                      (stepcdfoutgeneset[i] / stepcdfoutgeneset[n - 1])
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
    tau: int | float,
) -> float:
    n = len(generanking)
    k = len(gsetidx)
    walkstatpos, walkstatneg = _rnd_walk(
        gsetidx=gsetidx, k=k, decordstat=generanking, symrnkstat=rankstat, n=n, tau=tau
    )
    if maxdiff:
        if absrnk:
            es = walkstatpos - walkstatneg
        else:
            es = walkstatpos + walkstatneg
    else:
        es = walkstatpos if abs(walkstatpos) > abs(walkstatneg) else walkstatneg
    return es


@nb.njit(parallel=True, cache=True)
def _ks_fset(
    dos: np.ndarray,
    srs: np.ndarray,
    fset: np.ndarray,
    maxdiff: bool,
    absrnk: bool,
    tau: int | float,
) -> np.ndarray:
    n_samples, n_genes = dos.shape
    res = np.zeros(n_samples)
    for i in nb.prange(n_samples):
        generanking = dos[i]
        rankstat = srs[i]
        genesetsrankidx = fset
        res[i] = _score_geneset(genesetsrankidx, generanking, rankstat, maxdiff, absrnk, tau)
    return res


@docs.dedent
def _func_gsva(
    mat: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    kcdf: str | None = 'gaussian',
    maxdiff: bool = True,
    absrnk: bool = False,
    tau: int | float = 1,
    verbose: bool = False,
) -> Tuple[np.ndarray, None]:
    r"""
    Gene Set Variation Analysis (GSVA) :cite:`gsva`.

    Each feature is first transformed and smoothed using a kernel density estimation method:
    
    - Gaussian
    - Poisson
    - Empirical cumulative distribution function

    Features are then ranked based on a continuous metric (e.g., expression value, score, or correlation).
    
    Then, a score for each feature in a set is computed by walking down the ranked list,
    increasing a running-sum statistic when a feature belongs to the set and decreasing it otherwise.

    .. math::
    
       \delta(F, i) = 
       \begin{cases}
       \frac{|r_i|}{\sum\limits_{j \in F} |r_j|} & \text{if feature } i \in F \\
       -\frac{1}{l} & \text{if feature } i \notin F
       \end{cases}

    Where:

    - :math:`F` is a feature set
    - :math:`r` is the ranking of the feature statistics in descending order
    - :math:`r_i` is the value for feature :math:`i`
    - :math:`r_j` is the value for feature :math:`j` in :math:`F`
    - :math:`k` is the number of features in :math:`F`
    - :math:`N` is the total number of features in :math:`r`
    - :math:`l=N-k` is the number of features not in :math:`F` but present in :math:`r`

    For each feature, the function :math:`\delta(F,i)` is applied and stored as a sequence :math:`L`.

    .. math::

        L = \delta(F, i)\text{ for i} = \text{1, 2, ... , N}

    The enrichment score :math:`ES` is computed as the sum of the maximum positive and maximum negative deviations
    of the running-sum statistic from zero.

    .. math::

        ES = \max_{1 \leq i \leq N} L_i + \min_{1 \leq i \leq N} L_i

    %(notest)s

    %(params)s
    kcdf
        Which kernel to use during the non-parametric estimation of the cumulative distribution function.
        Options are gaussian, poisson or None.
    mx_diff
        Changes how the enrichment statistic (ES) is calculated.
        If ``True`` (default), ES is calculated as the difference between the maximum positive and
        negative random walk deviations.
        If ``False``, ES is calculated as the maximum positive to 0.
    abs_rnk : bool
        Used when ``mx_diff=True``. If ``False`` (default), the enrichment statistic (ES) is calculated taking the magnitude
        difference between the largest positive and negative random walk deviations.
        If ``True``, feature sets with features enriched on either extreme (high or low)
        will be regarded as 'highly' activated.

    %(returns)s
    """
    if isinstance(mat, sps.csr_matrix):
        m = f'gsva - Converting sparse matrix to dense format before density transformation'
        _log(m, level='info', verbose=verbose)
        mat = mat.toarray()
    m = f'gsva - computing density with kcdf={kcdf}'
    _log(m, level='info', verbose=verbose)
    # Compute density
    if mat.shape[0] > 1:
        mat = _density(mat, kcdf=kcdf)
    dos, srs = _rankmat(mat)
    # Compute GSVA
    nsrc = starts.size
    m = f'gsva - calculating {nsrc} scores with maxdiff={maxdiff}, absrnk={absrnk}'
    _log(m, level='info', verbose=verbose)
    es = np.zeros((dos.shape[0], nsrc))
    for j in tqdm(range(nsrc), disable=not verbose):
        fset = (_getset(cnct, starts, offsets, j) + 1).astype(int)
        es[:, j] = _ks_fset(dos=dos, srs=srs, fset=fset, maxdiff=maxdiff, absrnk=absrnk, tau=tau)
    return es, None


_gsva = MethodMeta(
    name='gsva',
    desc='Gene Set Variation Analysis (GSVA)',
    func=_func_gsva,
    stype='numerical',
    adj=False,
    weight=False,
    test=False,
    limits=(-1, +1),
    reference='https://doi.org/10.1186/1471-2105-14-7',
)
gsva = Method(_method=_gsva)
