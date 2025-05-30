from typing import Tuple

import numpy as np
import scipy.stats as sts

from decoupler._docs import docs
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


@docs.dedent
def _func_ulm(
    mat: np.ndarray,
    adj: np.ndarray,
    tval: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Univariate Linear Model (ULM) :cite:`decoupler`.

    This approach uses the molecular features from one observation as the population of samples
    and it fits a linear model with a single covariate, which is the feature weights of a set :math:`F`.

    .. math::

        y_i = \beta_0 + \beta_1 x_i + \varepsilon, \quad i = 1, 2, \ldots, n

    Where:

    - :math:`y_i` is the observed feature statistic (e.g. gene expression, :math:`log_{2}FC`, etc.) for feature :math:`i`
    - :math:`x_i` is the weight of feature :math:`i` in feature set :math:`F`. For unweighted sets, membership in the set is indicated by 1, and non-membership by 0.
    - :math:`\beta_0` is the intercept
    - :math:`\beta_1` is the slope coefficient
    - :math:`\varepsilon` is the error term for feature :math:`i`

    .. figure:: /_static/images/ulm.png
       :alt: Univariate Linear Model (ULM) schematic.
       :align: center
       :width: 75%

       Univariate Linear Model (ULM) scheme.
       In this example, the observed gene expression of :math:`Sample_1` is predicted using
       the interaction weights of :math:`TF_1`.
       Since the target genes that have negative weights are lowly expressed,
       and the positive target genes are highly expressed,
       the relationship between the two variables is positive so the obtained :math:`ES` score is positive.
       Scores can be interpreted as active when positive, repressive when negative, and inconclusive when close to 0.

    The enrichment score :math:`ES` is then calculated as the t-value of the slope coefficient.

    .. math::

        ES = t_{\beta_1} = \frac{\hat{\beta}_1}{\mathrm{SE}(\hat{\beta}_1)}

    Where:

    - :math:`t_{\beta_1}` is the t-value of the slope
    - :math:`\mathrm{SE}(\hat{\beta}_1)` is the standard error of the slope

    Next, :math:`p_{value}` are obtained by evaluating the two-sided survival function
    (:math:`sf`) of the Student’s t-distribution.

    .. math::

        p_{value} = 2 \times \mathrm{sf}(|ES|, \text{df})

    %(yestest)s

    %(params)s
    %(tval)s

    %(returns)s
    """
    # Get degrees of freedom
    n_var, n_src = adj.shape
    df = n_var - 2
    m = f'ulm - fitting {n_src} univariate models of {n_var} observations (targets) with {df} degrees of freedom'
    _log(m, level='info', verbose=verbose)
    # Compute R value for all
    r = _cor(adj, mat.T)
    # Compute t-value
    t = _tval(r, df)
    # Compute p-value
    pv = sts.t.sf(abs(t), df) * 2
    if tval:
        es = t
    else:
        # Compute coef
        es = r * (np.std(mat.T, ddof=1, axis=0).reshape(-1, 1) / np.std(adj, ddof=1, axis=0))
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
)
ulm = Method(_method=_ulm)
