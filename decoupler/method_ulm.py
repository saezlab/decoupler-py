"""
Method ULM.
Code to run the Univariate Linear Model (ULM) method.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from scipy.stats import t

from .pre import extract, match, rename_net, get_net_mat, filt_min_n, return_data

from tqdm import tqdm


def mat_cov(A, b):
    return np.dot(b.T - b.mean(), A - A.mean(axis=0)) / (b.shape[0]-1)


def mat_cor(A, b):
    cov = mat_cov(A, b)
    ssd = np.std(A, axis=0, ddof=1) * np.std(b, axis=0, ddof=1).reshape(-1, 1)
    return cov / ssd


def t_val(r, df):
    return r * np.sqrt(df / ((1.0 - r + 1.0e-16)*(1.0 + r + 1.0e-16)))


def ulm(mat, net, batch_size=10000, verbose=False):

    # Get dims
    n_samples = mat.shape[0]
    n_features, n_fsets = net.shape
    df = n_features - 2

    if isinstance(mat, csr_matrix):
        n_batches = int(np.ceil(n_samples / batch_size))
        es = np.zeros((n_samples, n_fsets), dtype=np.float32)
        for i in tqdm(range(n_batches), disable=not verbose):

            # Subset batch
            srt, end = i * batch_size, i * batch_size + batch_size
            batch = mat[srt:end].A.T

            # Compute R for batch
            r = mat_cor(net, batch)

            # Compute t-value
            es[srt:end] = t_val(r, df)
    else:
        # Compute R value for all
        r = mat_cor(net, mat.T)

        # Compute t-value
        es = t_val(r, df)

    # Compute p-value
    pv = t.sf(abs(es), df) * 2

    return es, pv


def run_ulm(mat, net, source='source', target='target', weight='weight', batch_size=10000,
            min_n=5, verbose=False, use_raw=True):
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
    batch_size : int
        Size of the samples to use for each batch. Increasing this will consume more memmory but it will run faster.
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
    estimate, pvals = ulm(m, net, batch_size=batch_size, verbose=verbose)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = 'ulm_estimate'
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = 'ulm_pvals'

    return return_data(mat=mat, results=(estimate, pvals))
