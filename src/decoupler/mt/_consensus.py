from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as sts
import numba as nb
from anndata import AnnData

from decoupler._docs import docs
from decoupler._log import _log
from decoupler.mt._gsea import _std
from decoupler.mt._run import _return


@nb.njit(cache=True)
def _zscore(
    sel: np.ndarray,
) -> np.ndarray:
    # Skip if all zeros
    if np.all(sel == 0):
        return sel
    # N selection
    n_sel = len(sel)
    # Mirror distr
    sel = np.append(sel, -sel)
    # Compute z-score (mean is 0)
    z = (sel / _std(sel, 1))[:n_sel]
    return z


@nb.njit(parallel=True, cache=True)
def _mean_zscores(
    scores: np.ndarray,
) -> np.ndarray:
    # Make a copy not to overwrite
    x = scores.copy()
    # Extract dims
    n_methods, n_samples, n_ftrs = x.shape
    # Init empty cons scores
    cons = np.zeros((n_samples, n_ftrs))
    # For each sample
    for i in nb.prange(n_samples):
        sample = x[:, i, :]
        # Compute z-score per method
        for k in range(n_methods):
            # Extract method scores
            methd = sample[k, :]
            # Compute pos z-score
            msk = methd > 0
            if np.any(msk):
                methd[msk] = _zscore(methd[msk])
            # Compute neg z-score
            msk = methd <= 0
            if np.any(msk):
                methd[msk] = _zscore(methd[msk])
        # Compute mean per feature
        for j in range(n_ftrs):
            ftr = sample[:, j]
            cons[i, j] = np.mean(ftr)
    return cons


@docs.dedent
def consensus(
    result: dict | AnnData,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame] | None:
    r"""
    Consensus score across methods.

    For each method, enrichment scores are split into positive and negative subsets
    and transformed independently into z-scores.
    
    1. Subset values based on sign (direction).
    2. Mirror each subset into positive and negative values with the same magnitude.
    3. Compute z-scores for each subset: :math:`z_i = \frac{x_i - \mu}{\sigma}`.
    4. Restore the original signs to the z-scored values

    This transformation ensures comparability across methods while preserving the
    biological interpretation of activation (positive) and inhibition (negative).
    The final consensus enrichment score :math:`ES` is computed as the mean of
    these signed z-scores across methods.

    .. math::

        ES = \frac{\sum_{m=1}^{M} z_{i}^{(m)}}{M} 

    Where:

    - :math:`M` is the number of methods
    - :math:`z_{i}^{(m)}` is the z-score from method :math:`m`.

    A two-sided :math:`p_{value}` is then calculated from the consensus score using
    the survival function :math:`sf` of the standard normal distribution.

    .. math::

        p = 2 \times \mathrm{sf}\bigl(\lvert \mathrm{ES} \rvert \bigr)

    %(yestest)s

    Parameters
    ----------
    result
        Results from ``decoupler.mt.decouple``.
    %(verbose)s

    Returns
    -------
    Consensus enrichment scores and p-values.
    """
    # Validate
    assert isinstance(result, (dict, AnnData)), 'scores must be dict or anndata.AnnData'
    # Transform to mat
    if isinstance(result, AnnData):
        keys = [k for k in result.obsm if 'score_' in k]
        scores = [result.obsm[k].values for k in keys]
        obs_names = result.obs_names
        var_names = result.obsm[keys[0]].columns
    else:
        keys = [k for k in result if 'score_' in k]
        scores = [result[k].values for k in keys]
        obs_names = result[keys[0]].index
        var_names = result[keys[0]].columns
    names = {k.split('_')[1] for k in keys}
    m = f'consensus - running consensus using methods={names}'
    _log(m, level='info', verbose=verbose)
    scores = np.array(scores)
    # Compute mean z-scores
    es = _mean_zscores(scores)
    # Compute p-vals
    pv = 2 * sts.norm.sf(np.abs(es))
    # FDR
    pv = sts.false_discovery_control(pv, axis=1, method='bh')
    # Transform to df
    es = pd.DataFrame(es, columns=var_names, index=obs_names)
    pv = pd.DataFrame(pv, columns=var_names, index=obs_names)
    return _return(name='consensus', data=result, es=es, pv=pv, verbose=False)
