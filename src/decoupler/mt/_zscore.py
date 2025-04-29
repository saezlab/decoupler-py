from typing import Tuple

import numpy as np
import scipy.stats as sts

from decoupler._log import _log
from decoupler._Method import MethodMeta, Method


def _func_zscore(
    mat: np.ndarray,
    adj: np.ndarray,
    flavor: str = 'RoKAI',
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(flavor, str), 'flavor must be str'
    nobs, nvar = mat.shape
    nvar, nsrc = adj.shape
    m = f'zscore - calculating {nsrc} scores with flavor={flavor}'
    _log(m, level='info', verbose=verbose)
    stds = np.std(mat, axis=1, ddof=1)
    mean_all = np.zeros(stds.shape)
    if flavor == 'RoKAI':
        mean_all = np.mean(mat, axis=1)
    n = np.sqrt(np.count_nonzero(adj, axis=0))
    mean = mat.dot(adj) / np.sum(np.abs(adj), axis=0)
    es = ((mean - mean_all.reshape(-1, 1)) * n) / stds.reshape(-1, 1)
    pv = sts.norm.cdf(-np.abs(es))
    return es, pv


_zscore = MethodMeta(
    name='zscore',
    func=_func_zscore,
    stype='numerical',
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference='https://doi.org/10.1038/s41467-021-21211-6',
    params='',
)
zscore = Method(_method=_zscore)
