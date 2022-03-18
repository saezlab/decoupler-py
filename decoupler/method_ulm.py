"""
Method ULM.
Code to run the Univariate Linear Model (ULM) method.
"""

import numpy as np
import pandas as pd

from scipy.stats import t

from .pre import extract, match, rename_net, get_net_mat, filt_min_n

from anndata import AnnData

import numba as nb


@nb.njit(nb.f4[:, :](nb.i4, nb.i4, nb.f4[:], nb.i4[:], nb.i4[:], nb.f4[:, :]), parallel=True, cache=True)
def nb_ulm(n_samples, n_features, data, indptr, indices, net):

    df, n_fsets = net.shape
    df = df - 2

    es = np.zeros((n_samples, n_fsets), dtype=nb.f4)

    for i in nb.prange(n_samples):

        # Extract sample from sparse matrix
        row = np.zeros(n_features, dtype=nb.f4)
        s, e = indptr[i], indptr[i+1]
        row[indices[s:e]] = data[s:e]

        for j in range(n_fsets):

            # Get fset column
            x = net[:, j]

            # Compute lm
            ssxm, ssxym, _, ssym = np.cov(x, row, bias=1).flat

            # Compute R value
            r = ssxym / np.sqrt(ssxm * ssym)

            # Compute t-value
            es[i, j] = r * np.sqrt(df / ((1.0 - r + 1.0e-20)*(1.0 + r + 1.0e-20)))

    return es


def ulm(mat, net):

    # Get df
    df = net.shape[0]
    df = df - 2

    # Compute ulm
    n_samples, n_features = mat.shape
    es = nb_ulm(n_samples, n_features, mat.data, mat.indptr, mat.indices, net)

    # Get p-values
    pvals = t.sf(abs(es), df) * 2

    return es, pvals


def run_ulm(mat, net, source='source', target='target', weight='weight', min_n=5, verbose=False, use_raw=True):
    """
    Univariate Linear Model (ULM).

    ULM fits a linear model for each sample and regulator, where the observed molecular readouts in `mat` are the response
    variable and the regulator weights in `net` are the explanatory one. Target features with no associated weight are set to
    zero. The obtained t-value from the fitted model is the activity (`ulm_estimate`) of a given regulator.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.

    Returns
    -------
    estimate : DataFrame
        ULM scores. Stored in `.obsm['ulm_estimate']` if `mat` is AnnData.
    pvals : DataFrame
        Obtained p-values. Stored in `.obsm['ulm_pvals']` if `mat` is AnnData.
    """

    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    sources, targets, net = get_net_mat(net)

    # Match arrays
    net = match(c, targets, net)

    if verbose:
        print('Running ulm on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), net.shape[1]))

    # Run ULM
    estimate, pvals = ulm(m, net)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = 'ulm_estimate'
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = 'ulm_pvals'

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, pvals
