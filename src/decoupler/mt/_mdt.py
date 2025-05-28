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
    # Init model
    reg = xgboost.XGBRegressor(**kwargs)
    # Fit
    y = y.reshape(-1, 1)
    reg = reg.fit(x, y)
    # Get R score
    es = reg.feature_importances_
    return es


@docs.dedent
def _func_mdt(
    mat: np.ndarray,
    adj: np.ndarray,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, None]:
    r"""
    Multivariate Decision Trees (MDT) :cite:`decoupler`.

    This approach uses the molecular features from one observation as the population of samples
    and it fits a gradient boosted decision trees model with multiple covariates,
    which are the weights of all feature sets :math:`F`. It uses the implementation provided by ``xgboost`` :cite:`xgboost`.

    The enrichment score :math:`ES` for each :math:`F` is then calculated as the importance of each covariate in the model.
    
    %(notest)s

    %(params)s

    kwargs
        All other keyword arguments are passed to ``xgboost.XGBRegressor``.
    %(returns)s
    """
    _check_import(xgboost)
    nobs = mat.shape[0]
    nvar, nsrc = adj.shape
    m = f'mdt - fitting {nsrc} multivariate decision tree models (XGBoost) of {nvar} targets across {nobs} observations'
    _log(m, level='info', verbose=verbose)
    es = np.zeros(shape=(nobs, nsrc))
    for i in tqdm(range(nobs), disable=not verbose):
        obs = mat[i]
        es[i, :] = _xgbr(x=adj, y=obs, **kwargs)
    return (es, None)


_mdt = MethodMeta(
    name='mdt',
    desc='Multivariate Decision Tree (MDT)',
    func=_func_mdt,
    stype='numerical',
    adj=True,
    weight=True,
    test=False,
    limits=(0, 1),
    reference='https://doi.org/10.1093/bioadv/vbac016',
)
mdt = Method(_method=_mdt)
