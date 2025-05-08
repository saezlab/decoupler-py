from typing import Tuple

import numpy as np
import scipy.stats as sts

from decoupler._log import _log
from decoupler._Method import MethodMeta, Method


def _cov(
    A: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    return np.dot(b.T - b.mean(), A - A.mean(axis=0)) / (b.shape[0]-1)


def _cor(
    A: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    cov = _cov(A, b)
    ssd = np.std(A, axis=0, ddof=1) * np.std(b, axis=0, ddof=1).reshape(-1, 1)
    return cov / ssd


def _tval(
    r: np.ndarray,
    df: float
) -> np.ndarray:
    return r * np.sqrt(df / ((1.0 - r + 2.2e-16) * (1.0 + r + 2.2e-16)))


def _func_ulm(
    mat: np.ndarray,
    adj: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    # Get degrees of freedom
    n_var, n_src = adj.shape
    df = n_var - 2
    m = f'ulm - fitting {n_src} univariate models of {n_var} observations (targets) with {df} degrees of freedom'
    _log(m, level='info', verbose=verbose)
    # Compute R value for all
    r = _cor(adj, mat.T)
    # Compute t-value
    es = _tval(r, df)
    # Compute p-value
    pv = sts.t.sf(abs(es), df) * 2
    return es, pv


_ulm = MethodMeta(
    name='ulm',
    desc='Univariate Linear Model (ULM)',
    func=_func_ulm,
    stype='numerical',
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference='https://doi.org/10.1093/bioadv/vbac016',
    params='',
)
ulm = Method(_method=_ulm)
