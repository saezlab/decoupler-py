import numpy as np
import pandas as pd

from .method_gsea import std

from scipy.stats import norm

import numba as nb


@nb.njit(nb.f4[:](nb.f4[:]), cache=True)
def z_score(sel):
    # Skip if all zeros
    if np.all(sel == 0):
        return sel

    # N selection
    n_sel = len(sel)

    # Mirror distr
    sel = np.append(sel, -sel)

    # Compute z-score (mean is 0)
    z = (sel / std(sel, 1))[:n_sel]

    return z


@nb.njit(nb.f4[:, :](nb.f4[:, :, :]), parallel=True, cache=True)
def consensus(acts):

    # Make a copy not to overwrite
    x = acts.copy()

    # Extract dims
    n_methods, n_samples, n_ftrs = x.shape

    # Init empty cons acts
    cons = np.zeros((n_samples, n_ftrs), dtype=nb.f4)

    # For each sample
    for i in nb.prange(n_samples):
        sample = x[:, i, :]

        # Compute z-score per method
        for k in range(n_methods):

            # Extract method acts
            methd = sample[k, :]

            # Compute pos z-score
            msk = methd > 0
            if np.any(msk):
                methd[msk] = z_score(methd[msk])

            # Compute neg z-score
            msk = methd <= 0
            if np.any(msk):
                methd[msk] = z_score(methd[msk])

        # Compute mean per feature
        for j in range(n_ftrs):
            ftr = sample[:, j]
            cons[i, j] = np.mean(ftr)

    return cons


def run_consensus(res):
    """
    Consensus.

    Computes a consensus score after running different methods with decouple. For each method, the obtained activities are
    transformed into z-scores, first for positive values and then for negative ones. These two sets of z-score transformed
    activities are computed by subsetting the values bigger or lower than 0, then by mirroring the selected values into their
    opposite sign and finally calculating a classic z-score. This transformation ensures that values across methods are
    comparable, and that they remain in their original sign (active or inactive). The final consensus score is the mean across
    different methods.

    Parameters
    ----------
    res : dict
        Results from `decouple`.

    Returns
    -------
    estimate : DataFrame
        Consensus scores.
    pvals : DataFrame
        Obtained p-values.
    """

    acts = np.array([res[k].values for k in res if 'pvals' not in k and not
                     np.all(np.isnan(res[k].values))]).astype(np.float32)

    # Compute mean z-scores
    estimate = consensus(acts)

    # Compute p-vals
    pvals = norm.cdf(-np.abs(estimate)) * 2

    # Transform to df
    k = list(res.keys())[0]
    index = res[k].index
    columns = res[k].columns
    estimate = pd.DataFrame(estimate, columns=columns, index=index)
    estimate.name = 'consensus_estimate'
    pvals = pd.DataFrame(pvals, columns=columns, index=index)
    pvals.name = 'consensus_pvals'

    return estimate, pvals
