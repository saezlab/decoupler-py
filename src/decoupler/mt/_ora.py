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


@nb.njit(cache=True)
def _mlnTest2r(
    a: int,
    ab: int,
    ac: int,
    abcd: int,
) -> float:    
    if 0 > a or a > ab or a > ac or ab + ac > abcd + a:
        raise ValueError('invalid contingency table')
    a_min = max(0, ab + ac - abcd)
    a_max = min(ab, ac)
    if a_min == a_max:
        return 0.
    p0 = math.lgamma(ab + 1) + math.lgamma(ac + 1) + math.lgamma(abcd - ac + 1) + \
    math.lgamma(abcd - ab + 1) - math.lgamma(abcd + 1)
    pa = math.lgamma(a + 1) + math.lgamma(ab - a + 1) + math.lgamma(ac - a + 1) + \
    math.lgamma(abcd - ab - ac + a + 1)
    if ab * ac > a * abcd:
        sl = 0.
        for i in range(a - 1, a_min - 1, -1):
            sl_new = sl + math.exp(
                pa - math.lgamma(i + 1) - math.lgamma(ab - i + 1) - \
                math.lgamma(ac - i + 1) - math.lgamma(abcd - ab - ac + i + 1)
            )
            if sl_new == sl:
                break
            sl = sl_new
        return -math.log(1. - max(0, math.exp(p0 - pa) * sl))
    else:
        sr = 1.
        for i in range(a + 1, a_max + 1):
            sr_new = sr + math.exp(
                pa - math.lgamma(i + 1) - math.lgamma(ab - i + 1) - \
                math.lgamma(ac - i + 1) - math.lgamma(abcd - ab - ac + i + 1)
            )
            if sr_new == sr:
                break
            sr = sr_new
        return max(0, pa - p0 - math.log(sr))


@nb.njit(cache=True)
def _test1r(
    a: int,
    b: int,
    c: int,
    d: int,
) -> float:
    # https://github.com/painyeph/FishersExactTest/blob/master/fisher.py
    return math.exp(-_mlnTest2r(a, a + b, a + c, a + b + c + d))


@nb.njit(parallel=True, cache=True)
def _stsora(
    row: np.ndarray,
    ranks: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_bg: int | None,
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
        nset = fset.size
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
        # Haldane-Anscombe correction
        es[j] = ((a + 0.5) * (n_bg - nset + 0.5)) / ((nset + 0.5) * (nvar - a + 0.5))
        pv[j] = _test1r(a, b, c, d)
    return es, pv


def _func_ora(
    mat: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_up: int | float | None = None,
    n_bm: int | float = 0,
    n_bg: int | float | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    nobs, nvar = mat.shape
    nsrc = starts.size
    if n_up is None:
        n_up = int(np.ceil(0.05 * nvar))
        m = f'ora - setting n_up={n_up}' 
        _log(m, level='info', verbose=verbose)
    if n_bg is None:
        #n_bg = int(np.max(cnct) + 1)
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
        es[i], pv[i] = _stsora(row=row, ranks=ranks, cnct=cnct, starts=starts, offsets=offsets, n_bg=n_bg)
    return es, pv


params = """\
n_up
    Number of top-ranked features, based on their magnitude, to select as observed features.
    If ``None``, the top 5% of positive features are selected.
n_bm
    Number of bottom-ranked features, based on their magnitude, to select as observed features.
n_bg
    Number indicating the background size. If ``None``, is `` mat - (n_up | n_bm)``.
"""

_ora = MethodMeta(
    name='ora',
    func=_func_ora,
    stype='categorical',
    adj=False,
    weight=False,
    test=True,
    limits=(-np.inf, +np.inf),
    reference='https://doi.org/10.2307/2340521',
    params=params,
)
ora = Method(_method=_ora)
