from typing import Tuple

import numpy as np
import scipy.stats as sts

from decoupler._docs import docs
from decoupler._log import _log
from decoupler._Method import MethodMeta, Method


@docs.dedent
def _func_zscore(
    mat: np.ndarray,
    adj: np.ndarray,
    flavor: str = 'RoKAI',
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Z-score (ZSCORE) :cite:`zscore`.

    This approach computes the mean value of the molecular features for known targets,
    optionally subtracts the overall mean of all measured features,
    and normalizes the result by the standard deviation of all features and the square
    root of the number of targets.
    
    This formulation was originally introduced in KSEA, which explicitly includes the
    subtraction of the global mean to compute the enrichment score :math:`ES`.

    .. math::

        ES = \frac{(\mu_s-\mu_p) \times \sqrt m }{\sigma}

    Where:

    - :math:`\mu_s` is the mean of targets
    - :math:`\mu_p` is the mean of all features
    - :math:`m` is the number of targets
    - :math:`\sigma` is the standard deviation of all features
    
    However, in the RoKAI implementation, this global mean subtraction was omitted.

    .. math::

        ES = \frac{\mu_s \times \sqrt m }{\sigma}

    A two-sided :math:`p_{value}` is then calculated from the consensus score using
    the survival function :math:`sf` of the standard normal distribution.

    .. math::

        p = 2 \times \mathrm{sf}\bigl(\lvert \mathrm{ES} \rvert \bigr)

    %(yestest)s

    %(params)s

    flavor
        Which flavor to use when calculating the z-score, either KSEA or RoKAI.

    %(returns)s
    """
    assert isinstance(flavor, str) and flavor in ['KSEA', 'RoKAI'], \
    'flavor must be str and KSEA or RoKAI'
    nobs, nvar = mat.shape
    nvar, nsrc = adj.shape
    m = f'zscore - calculating {nsrc} scores with flavor={flavor}'
    _log(m, level='info', verbose=verbose)
    stds = np.std(mat, axis=1, ddof=1)
    if flavor == 'RoKAI':
        mean_all = np.mean(mat, axis=1)
    elif flavor == 'KSEA':
        mean_all = np.zeros(stds.shape)
    n = np.sqrt(np.count_nonzero(adj, axis=0))
    mean = mat.dot(adj) / np.sum(np.abs(adj), axis=0)
    es = ((mean - mean_all.reshape(-1, 1)) * n) / stds.reshape(-1, 1)
    pv = 2 * sts.norm.sf(np.abs(es))
    return es, pv


_zscore = MethodMeta(
    name='zscore',
    desc='Z-score (ZSCORE)',
    func=_func_zscore,
    stype='numerical',
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference='https://doi.org/10.1038/s41467-021-21211-6',
)
zscore = Method(_method=_zscore)
