from typing import Tuple

import numpy as np
import scipy.stats as sts
import scipy.sparse as sps
from tqdm.auto import tqdm
import numba as nb

from decoupler._docs import docs
from decoupler._log import _log
from decoupler._Method import MethodMeta, Method
from decoupler.pp.net import _getset


@nb.njit(cache=True)
def _std(
    arr: np.ndarray,
    ddof: int,
) -> float:
    N = arr.shape[0]
    m = np.mean(arr)
    var = np.sum((arr - m)**2) / (N - ddof)
    sd = np.sqrt(var)
    return sd


def _ridx(
    times: int,
    nvar: int,
    seed: int | None,
):
    idx = np.tile(np.arange(nvar), (times, 1))
    if seed:
        rng = np.random.default_rng(seed=seed)
        for i in idx:
            rng.shuffle(i)
    return idx


@nb.njit(cache=True)
def _esrank(
    row: np.ndarray,
    rnks: np.ndarray,
    set_msk: np.ndarray,
    dec: float,
) -> Tuple[float, int, np.ndarray]:
    # Init empty
    mx_value = 0.0
    cum_sum = 0.0
    mx_pos = 0.0
    mx_neg = 0.0
    j_pos = 0
    j_neg = 0
    es = np.zeros(rnks.size)
    # Compute norm
    sum_set = np.sum(np.abs(row[set_msk]))
    if sum_set == 0.:
        return 0., 0, np.zeros(rnks.size)
    # Compute ES
    for i in rnks:
        if set_msk[i]:
            cum_sum += np.abs(row[i]) / sum_set
            es[i] = cum_sum
        else:
            cum_sum -= dec
            es[i] = cum_sum
        # Update max scores and idx
        if cum_sum > mx_pos:
            mx_pos = cum_sum
            j_pos = i
        if cum_sum < mx_neg:
            mx_neg = cum_sum
            j_neg = i
    # Determine if pos or neg are more enriched
    if mx_pos > -mx_neg:
        mx_value = mx_pos
        j = j_pos
    else:
        mx_value = mx_neg
        j = j_neg
    return mx_value, j, es


@nb.njit(cache=True)
def _nesrank(
    ridx: np.ndarray,
    row: np.ndarray,
    rnks: np.ndarray,
    set_msk: np.ndarray,
    dec: float,
    es: float,
) -> Tuple[float, float]:
    # Keep old set_msk upstream
    set_msk = set_msk.copy()
    # Compute null
    times, nvar = ridx.shape
    if times == 0:
        return 0., 1.
    null = np.zeros(times)
    for i in range(times):
        null[i], _, _ = _esrank(row=row, rnks=rnks, set_msk=set_msk[ridx[i]], dec=dec)
    # Compute NES
    pos_null_msk = null >= 0.
    neg_null_msk = null < 0.
    pos_null_sum = pos_null_msk.sum()
    neg_null_sum = neg_null_msk.sum()
    if (es >= 0) and (pos_null_sum > 0):
        pval = (null[pos_null_msk] >= es).sum() / pos_null_sum
        pos_null_mean = null[pos_null_msk].mean()
        nes = es / pos_null_mean
    elif (es < 0) and (neg_null_sum > 0):
        pval = (null[neg_null_msk] <= es).sum() / neg_null_sum
        neg_null_mean = null[neg_null_msk].mean()
        nes = -es / neg_null_mean
    else:
        nes = np.inf
        pval = np.inf
    return nes, pval


@nb.njit(parallel=True, cache=True)
def _stsgsea(
    row: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    ridx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Sort features
    idx = np.argsort(-row)
    row = row[idx]
    # Init empty
    nvar = row.size
    nsrc = starts.size
    rnks = np.arange(nvar)
    es = np.zeros(nsrc)
    nes = np.zeros(nsrc)
    pv = np.ones(nsrc)
    for j in nb.prange(nsrc):
        # Extract fset
        fset = _getset(cnct, starts, offsets, j)
        # Get decending penalty
        dec = 1.0 / (nvar - fset.size)
        # Get msk
        set_msk = np.zeros(nvar, dtype=np.bool_)
        set_msk[fset] = True
        set_msk = set_msk[idx]
        # Compute es per feature
        es[j], _, _ = _esrank(row=row, rnks=rnks, set_msk=set_msk, dec=dec)
        nes[j], pv[j] = _nesrank(ridx=ridx, row=row, rnks=rnks, set_msk=set_msk, dec=dec, es=es[j])
    return es, nes, pv


@docs.dedent
def _func_gsea(
    mat: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    times: int | float = 1000,
    seed: int | float = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Gene Set Enrichment Analysis (GSEA) :cite:`gsea`.

    Features are ranked based on a continuous statistic (e.g., expression, score, or correlation).
    The enrichment score (ES) for a feature set is computed by walking down the ranked list and increasing a running-sum
    statistic when a feature is in the set, and decreasing it when it is not.

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

    The enrichment score :math:`ES` corresponds to the maximum deviation from zero of this running sum.

    .. math::

        ES = L_{arg max |L|}

    When multiple random permutations are done (``times > 1``), statistical significance is assessed via empirical testing.

    .. math::

        p_{value}=\frac{ES_{rand} \geq ES}{P}

    Where:

    - :math:`ES_{rand}` are the enrichment scores of the random permutations
    - :math:`P` is the total number of permutations

    Additionaly, :math:`ES` is updated to a normalized enrichment score :math:`NES`.

    .. math::

        NES = \begin{cases} \frac{ES}{\mu_{+}} & \text{if } ES > 0 \\ \frac{ES}{\mu_{-}} & \text{if } ES < 0  \end{cases}

    Where:

    - :math:`\mu{+}` is the mean of positive values in :math:`ES_{rand}`
    - :math:`\mu{-}` is the mean of negative values in :math:`ES_{rand}`

    %(yestest)s

    %(params)s
    %(times)s
    %(seed)s

    %(returns)s
    """
    nobs, nvar = mat.shape
    assert isinstance(times, (int, float)) and times >= 0, 'times must be numeric and >= 0'
    assert isinstance(seed, (int, float)) and seed >= 0, 'seed must be numeric and >= 0'
    times, seed = int(times), int(seed)
    # Compute
    nsrc = starts.size
    m = f'gsea - calculating {nsrc} scores across {nobs} observations'
    _log(m, level='info', verbose=verbose)
    if times > 1:
        m = f'gsea - comparing estimates against {times} random permutations'
        _log(m, level='info', verbose=verbose)
        ridx = _ridx(times=times, nvar=nvar, seed=seed)
    else:
        ridx = _ridx(times=times, nvar=nvar, seed=None)
    es = np.zeros(shape=(nobs, nsrc))
    nes = np.zeros(shape=(nobs, nsrc))
    pv = np.zeros(shape=(nobs, nsrc))
    for i in tqdm(range(nobs), disable=not verbose):
        if isinstance(mat, sps.csr_matrix):
            row = mat[i].toarray()[0]
        else:
            row = mat[i]
        es[i, :], nes[i, :], pv[i, :] = _stsgsea(
            row=row,
            cnct=cnct,
            starts=starts,
            offsets=offsets,
            ridx=ridx,
        )
    if times > 1:
        es = nes
    return es, pv


_gsea = MethodMeta(
    name='gsea',
    desc='Gene Set Enrichment Analysis (GSEA)',
    func=_func_gsea,
    stype='numerical',
    adj=False,
    weight=False,
    test=True,
    limits=(-np.inf, +np.inf),
    reference='https://doi.org/10.1073/pnas.0506580102',
)
gsea = Method(_method=_gsea)
