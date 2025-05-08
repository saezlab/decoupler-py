from typing import Tuple

import numpy as np
import numba as nb
import scipy.stats as sts

from decoupler._log import _log
from decoupler._Method import MethodMeta, Method


@nb.njit(parallel=True, cache=True)
def _tval(X, y, inv, df):
    X = np.ascontiguousarray(X)
    n_samples = y.shape[1]
    n_fsets = X.shape[1]
    coef, sse, _, _ = np.linalg.lstsq(X, y)
    assert len(sse) > 0, 'Could not fit a multivariate linear model. This can happen because there are more sources\n \
    (covariates) than unique targets (samples), or because the network adjacency matrix rank is smaller than the number\n \
    of sources'
    sse = sse / df
    se = np.zeros((n_samples, n_fsets))
    for i in nb.prange(n_samples):
        se[i] = np.sqrt(np.diag(sse[i] * inv))
    t = coef.T / se
    return t.astype(nb.f4)


def _func_mlm(
    mat: np.ndarray,
    adj: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    # Get dims
    n_features, n_fsets = adj.shape
    # Add intercept
    adj = np.column_stack((np.ones((n_features, )), adj))
    # Compute inv and df for lm
    inv = np.linalg.inv(np.dot(adj.T, adj))
    df = n_features - n_fsets - 1
    m = f'mlm - fitting {n_fsets} multivariate models of {n_features} observations with {df} degrees of freedom'
    _log(m, level='info', verbose=verbose)
    # Compute tval
    es = _tval(adj, mat.T, inv, df)[:, 1:]
    # Compute pval
    pv = 2 * (1 - sts.t.cdf(x=np.abs(es), df=df))
    return es, pv


_mlm = MethodMeta(
    name='mlm',
    desc='Multivariate Linear Model (MLM)',
    func=_func_mlm,
    stype='numerical',
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference='https://doi.org/10.1093/bioadv/vbac016',
    params='',
)
mlm = Method(_method=_mlm)
