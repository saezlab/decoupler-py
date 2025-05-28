import warnings
from typing import Tuple

import numpy as np
import scipy.sparse as sps
from tqdm.auto import tqdm

from decoupler._odeps import xgboost, _check_import
from decoupler._docs import docs
from decoupler._log import _log
from decoupler._Method import MethodMeta, Method


def _xgbr(
    x: np.ndarray,
    y: np.ndarray,
    **kwargs,
) -> np.ndarray:
    kwargs.setdefault('n_estimators', 10)
    # Init model
    reg = xgboost.XGBRegressor(**kwargs)
    # Fit
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    reg = reg.fit(x, y)
    # Get R score
    es = reg.score(x, y)
    # Clip to [0, 1]
    es = np.clip(es, 0, 1)
    return es


@docs.dedent
def _func_udt(
    mat: np.ndarray,
    adj: np.ndarray,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, None]:
    """
    Univariate Decision Tree (UDT) :cite:`decoupler`.

    This approach uses the molecular features from one observation as the population of samples
    and it fits a gradient boosted decision trees model with a single covariate,
    which is the feature weights of a set :math:`F`.
    It uses the implementation provided by ``xgboost`` :cite:`xgboost`.

    The enrichment score :math:`ES` is then calculated as the coefficient of determination :math:`R^2`.

    %(notest)s

    %(params)s

    kwargs
        All other keyword arguments are passed to ``xgboost.XGBRegressor``.
    %(returns)s
    """
    _check_import(xgboost)
    nobs = mat.shape[0]
    nvar, nsrc = adj.shape
    m = f'udt - fitting {nsrc} univariate decision tree models (XGBoost) of {nvar} targets across {nobs} observations' 
    _log(m, level='info', verbose=verbose)
    es = np.zeros(shape=(nobs, nsrc))
    for i in tqdm(range(nobs), disable=not verbose):
        obs = mat[i]
        for j in range(adj.shape[1]):
            es[i, j] = _xgbr(x=adj[:, j], y=obs, **kwargs)
    return es, None


_udt = MethodMeta(
    name='udt',
    desc='Univariate Decision Tree (UDT)',
    func=_func_udt,
    stype='numerical',
    adj=True,
    weight=True,
    test=False,
    limits=(0, 1),
    reference='https://doi.org/10.1093/bioadv/vbac016',
)
udt = Method(_method=_udt)
