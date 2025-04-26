from typing import Tuple

import numpy as np
import scipy.sparse as sps
from tqdm.auto import tqdm
from xgboost import XGBRegressor

from decoupler._log import _log
from decoupler._Method import MethodMeta, Method


def _xgbr(
    x: np.ndarray,
    y: np.ndarray,
    **kwargs,
) -> np.ndarray:
    # Init model
    reg = XGBRegressor(**kwargs)
    # Fit
    y = y.reshape(-1, 1)
    reg = reg.fit(x, y)
    # Get R score
    es = reg.feature_importances_
    return es


def _func_mdt(
    mat: np.ndarray,
    adj: np.ndarray,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, None]:
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
    func=_func_mdt,
    stype='numerical',
    adj=True,
    weight=True,
    test=False,
    limits=(0, 1),
    reference='https://doi.org/10.1093/bioadv/vbac016',
)
mdt = Method(_method=_mdt)
