from typing import Tuple

import numpy as np
import numba as nb
import scipy.stats as sts

from decoupler._docs import docs
from decoupler._log import _log
from decoupler._Method import MethodMeta, Method


@nb.njit(parallel=True, cache=True)
def _fit(
    X: np.ndarray,
    y: np.ndarray,
    inv: np.ndarray,
    df: float,
) -> np.ndarray:
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
    coef = coef.T
    tval = coef / se
    return coef[:, 1:], tval[:, 1:]


@docs.dedent
def _func_mlm(
    mat: np.ndarray,
    adj: np.ndarray,
    tval: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Multivariate Linear Model (MLM) :cite:`decoupler`.

    This approach uses the molecular features from one observation as the population of samples
    and it fits a linear model with with multiple covariates, which are the weights of all feature sets :math:`F`.

    .. math::

        y^i = \beta_0 + \beta_1 x_{1}^{i} + \beta_2 x_{2}^{i} + \cdots + \beta_p x_{p}^{i} + \varepsilon

    Where:

    - :math:`y^i` is the observed feature statistic (e.g. gene expression, :math:`log_{2}FC`, etc.) for feature :math:`i`
    - :math:`x_{p}^{i}` is the weight of feature :math:`i` in feature set :math:`F_p`. For unweighted sets, membership in the set is indicated by 1, and non-membership by 0.
    - :math:`\beta_0` is the intercept
    - :math:`\beta_p` is the slope coefficient for feature set :math:`F_p`
    - :math:`\varepsilon` is the error term for feature :math:`i`

    .. figure:: /_static/images/mlm.png
       :alt: Multivariate Linear Model (MLM) schematic.
       :align: center
       :width: 75%

       Multivariate Linear Model (MLM) scheme.
       In this example, the observed gene expression of :math:`Sample_1` is predicted using
       the interaction weights of two pathways, :math:`P_1` and :math:`P_2`.
       For :math:`P2`, since its target genes that have negative weights are lowly expressed,
       and its positive target genes are highly expressed,
       the relationship between the two variables is positive so the obtained :math:`ES` score is positive.
       Scores can be interpreted as active when positive, repressive when negative, and inconclusive when close to 0.

    The enrichment score :math:`ES` for each :math:`F` is then calculated as the t-value of the slope coefficients.

    .. math::

        ES = t_{\beta_1} = \frac{\hat{\beta}_1}{\mathrm{SE}(\hat{\beta}_1)}

    Where:

    - :math:`t_{\beta_1}` is the t-value of the slope
    - :math:`\mathrm{SE}(\hat{\beta}_1)` is the standard error of the slope

    Next, :math:`p_{value}` are obtained by evaluating the two-sided survival function
    (:math:`sf`) of the Student’s t-distribution.

    .. math::

        p_{value} = 2 \times \mathrm{sf}(|ES|, \text{df})
    

    %(params)s
    %(tval)s

    %(returns)s
    """
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
    coef, t = _fit(adj, mat.T, inv, df)
    # Compute pval
    pv = 2 * (1 - sts.t.cdf(x=np.abs(t), df=df))
    # Return coef or tval
    if tval:
        es = t
    else:
        es = coef
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
)
mlm = Method(_method=_mlm)
