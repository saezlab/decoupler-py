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


def _maxn() -> int:
    l = 1; n = 2; h = float('inf')
    while l < n:
        if abs(math.lgamma(n+1) - math.lgamma(n) - math.log(n)) >= 1: h = n
        else: l = n
        n = (l + min(h, l * 3)) // 2
    return n

MAXN = _maxn()

@nb.njit(cache=True)
def _mlnTest2t(
    a: int,
    ab: int,
    ac: int,
    abcd: int,
):
    if 0 > a or a > ab or a > ac or ab + ac > abcd + a: raise ValueError('invalid contingency table')
    if abcd > MAXN: raise OverflowError('the grand total of contingency table is too large')
    a_min = max(0, ab + ac - abcd)
    a_max = min(ab, ac)
    if a_min == a_max: return 0.
    p0 = math.lgamma(ab + 1) + math.lgamma(ac + 1) + math.lgamma(abcd - ac + 1) + math.lgamma(abcd - ab + 1) - math.lgamma(abcd + 1)
    pa = math.lgamma(a + 1) + math.lgamma(ab - a + 1) + math.lgamma(ac - a + 1) + math.lgamma(abcd - ab - ac + a + 1)
    st = 1.
    if ab * ac < a * abcd:
        for i in range(min(a - 1, int(round(ab * ac / abcd))), a_min - 1, -1):
            pi = math.lgamma(i + 1) + math.lgamma(ab - i + 1) + math.lgamma(ac - i + 1) + math.lgamma(abcd - ab - ac + i + 1)
            if pi < pa: continue
            st_new = st + math.exp(pa - pi)
            if st_new == st: break
            st = st_new
        for i in range(a + 1, a_max + 1):
            pi = math.lgamma(i + 1) + math.lgamma(ab - i + 1) + math.lgamma(ac - i + 1) + math.lgamma(abcd - ab - ac + i + 1)
            st_new = st + math.exp(pa - pi)
            if st_new == st: break
            st = st_new
    else:
        for i in range(a - 1, a_min - 1, -1):
            pi = math.lgamma(i + 1) + math.lgamma(ab - i + 1) + math.lgamma(ac - i + 1) + math.lgamma(abcd - ab - ac + i + 1)
            st_new = st + math.exp(pa - pi)
            if st_new == st: break
            st = st_new
        for i in range(max(a + 1, int(round(ab * ac / abcd))), a_max + 1):
            pi = math.lgamma(i + 1) + math.lgamma(ab - i + 1) + math.lgamma(ac - i + 1) + math.lgamma(abcd - ab - ac + i + 1)
            if pi < pa: continue
            st_new = st + math.exp(pa - pi)
            if st_new == st: break
            st = st_new
    return max(0, pa - p0 - math.log(st))


@nb.njit(cache=True)
def _test1t(
    a: int,
    b: int,
    c: int,
    d: int,
) -> float:
    # https://github.com/painyeph/FishersExactTest/blob/master/fisher.py
    return math.exp(-_mlnTest2t(a, a + b, a + c, a + b + c + d))


@nb.njit(cache=True)
def _oddsr(
    a: int,
    b: int,
    c: int,
    d: int,
    ha_corr: int | float = 0.5,
    log: bool = True,
):
    # Haldane-Anscombe correction
    a += ha_corr
    b += ha_corr
    c += ha_corr
    d += ha_corr
    r = (a * d) / (b * c)
    if log and r != 0.:
        r = math.log(r)
    return r


@nb.njit(parallel=True, cache=True)
def _runora(
    row: np.ndarray,
    ranks: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_bg: int | None,
    ha_corr: int | float = 0.5,
) -> Tuple[float, float]:
    nvar = row.size
    nsrc = starts.size
    # Transform to set
    row = set(row)
    ranks = set(ranks)
    es = np.zeros(nsrc)
    pv = np.zeros(nsrc)
    for j in nb.prange(nsrc):
        # Extract feature set
        fset = _getset(cnct=cnct, starts=starts, offsets=offsets, j=j)
        fset = set(fset)
        # Build table
        set_a = row.intersection(fset)
        set_b = fset.difference(row)
        set_c = row.difference(fset)
        a = len(set_a)
        b = len(set_b)
        c = len(set_c)
        if n_bg == 0:
            set_u = set_a.union(set_b).union(set_c)
            set_d = ranks.difference(set_u)
            d = len(set_d)
        else:
            d = n_bg - a - b - c
        es[j] = _oddsr(a=a, b=b, c=c, d=d, ha_corr=ha_corr, log=True)
        pv[j] = _test1t(a=a, b=b, c=c, d=d)
    return es, pv


@docs.dedent
def _func_ora(
    mat: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_up: int | float | None = None,
    n_bm: int | float = 0,
    n_bg: int | float | None = 20_000,
    ha_corr: int | float = 0.5,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Over Representation Analysis (ORA) :cite:`ora`.

    This approach first creates a contingency table.

    .. list-table:: 2×2 Contingency Table
       :header-rows: 1
       :widths: 20 20 20

       * -
         - :math:`\in F`
         - :math:`\notin F`
       * - :math:`\in Sign. features`
         - :math:`a`
         - :math:`b`
       * - :math:`\notin Sign. features`
         - :math:`c`
         - :math:`d`

    Where:

    - :math:`a` is the number of features that are both significant and in :math:`F`
    - :math:`b` is the number of features that are signficiant but not in :math:`F`
    - :math:`c` is the number of features that are not signficiant but in :math:`F`
    - :math:`d` is the number of features that are not signficiant and not in :math:`F`

    .. figure:: /_static/images/ora.png
       :alt: Over Representation Analysis (ORA) schematic.
       :align: center
       :width: 100%

       Over Representation Analysis (ORA) scheme.
    
    The statistic is calculated as the Odds Ratio :math:`OR` with Haldane-Anscombe correction.

    .. math::
        
        \text{OR} = \log{\frac{\frac{a + 0.5}{b + 0.5}}{\frac{c + 0.5}{d + 0.5}}}

    And the :math:`p_{value}` is obtained afer computing a two-tailed Fisher’s exact test with the same table.

    %(yestest)s

    %(params)s

    n_up
        Number of top-ranked features, based on their magnitude, to select as observed features.
        If ``None``, the top 5% of positive features are selected.
    n_bm
        Number of bottom-ranked features, based on their magnitude, to select as observed features.
    n_bg
        Number indicating the background size.

    %(returns)s
    """
    nobs, nvar = mat.shape
    nsrc = starts.size
    if n_up is None:
        n_up = int(np.max([np.ceil(0.05 * nvar), 2]))
        m = f'ora - setting n_up={n_up}' 
        _log(m, level='info', verbose=verbose)
    if n_bg is None:
        n_bg = 0
        m = f'ora - not using n_bg, a feature specific background will be used instead' 
        _log(m, level='info', verbose=verbose)
    assert isinstance(n_up, (int, float)) and n_up > 0, 'n_up must be numeric and > 0'
    assert isinstance(n_bm, (int, float)) and n_bm >= 0, 'n_bm must be numeric and positive'
    assert isinstance(n_bg, (int, float)) and n_bg >= 0, 'n_bg must be numeric and positive'
    m = f'ora - calculating {nsrc} scores across {nobs} observations with n_up={n_up}, n_bm={n_bm}, n_bg={n_bg}' 
    _log(m, level='info', verbose=verbose)
    es = np.zeros((nobs, nsrc))
    pv = np.zeros((nobs, nsrc))
    ranks = np.arange(nvar, dtype=np.int_)
    for i in tqdm(range(nobs), disable=not verbose):
        if isinstance(mat, sps.csr_matrix):
            row = mat[i].toarray()[0]
        else:
            row = mat[i]
        # Find ranks
        row = sts.rankdata(row, method='ordinal')
        row = ranks[(row > n_up) | (row < n_bm)]
        es[i], pv[i] = _runora(row=row, ranks=ranks, cnct=cnct, starts=starts, offsets=offsets, n_bg=n_bg, ha_corr=ha_corr)
    return es, pv


_ora = MethodMeta(
    name='ora',
    desc='Over Representation Analysis (ORA)',
    func=_func_ora,
    stype='categorical',
    adj=False,
    weight=False,
    test=True,
    limits=(-np.inf, +np.inf),
    reference='https://doi.org/10.2307/2340521',
)
ora = Method(_method=_ora)
