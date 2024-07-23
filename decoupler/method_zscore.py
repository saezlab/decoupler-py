"""
Method zscore.
Code to run the z-score (RoKAI, KSEA) method.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import isspmatrix_csr

from scipy.stats import t
from scipy.stats import norm

from .pre import extract, match, rename_net, get_net_mat, filt_min_n, return_data

from tqdm import tqdm


def zscore(m, net, flavor='RoKAI', verbose=False):
    stds = np.std(m, axis=1, ddof=1)
    if flavor != 'RoKAI':
        mean_all = np.mean(m, axis=1)
    else:
        mean_all = np.zeros(stds.shape)
    n = np.sqrt(np.count_nonzero(net, axis=0))
    mean = m.dot(net) / np.sum(np.abs(net), axis=0)
    es = ((mean - mean_all.reshape(-1, 1)) * n) / stds.reshape(-1, 1)
    pv = norm.cdf(-np.abs(es))
    return es, pv


def run_zscore(mat, net, source='source', target='target', weight='weight', batch_size=10000, flavor='RoKAI',
             min_n=5, verbose=False, use_raw=True):
    """
    z-score.

    Calculates regulatory activities using a z-score as descibed in KSEA or RoKAI. The z-score calculates the mean of the molecular features of the 
    known targets for each regulator and adjusts it for the number of identified targets for the regulator, the standard deviation of all molecular 
    features (RoKAI), as well as the mean of all moleculare features (KSEA).

    Parameters
    ----------
    mat : list, DataFrame
        List of [features, matrix], dataframe (samples x features).
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    flavor : int
        Whether to use the implementation of RoKAI (default) or KSEA.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.

    Returns
    -------
    estimate : DataFrame
        Z-scores.
    pvals : DataFrame
        Obtained p-values.
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
        print('Running zscore on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), net.shape[1]))

    # Run ULM
    estimate, pvals = zscore(m, net, flavor=flavor)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = 'zscore_estimate'
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = 'zscore_pvals'

    return return_data(mat=mat, results=(estimate, pvals))
