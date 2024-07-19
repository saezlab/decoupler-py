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

    # Get dims
    n_samples = m.shape[0]
    n_features, n_fsets = net.shape
    
    es = np.zeros((n_samples, n_fsets))
    pv = np.zeros((n_samples, n_fsets))
                     
    # Compute each element of Matrix3
    for i in range(n_samples):
        for f in range(n_fsets):
            m_f = m[:, i]
            if isspmatrix_csr(m_f):
                m_f = m_f.toarray()
                m_f = m_f.reshape(1, -1)
            net_i = net[:, f]

            mean_product = np.sum(m_f * net_i) / np.sum(abs(net_i))
            mean_m_f = 0 if flavor == "RoKAI" else np.mean(m_f)
            std_m_f = np.std(m_f)
            count_non_zeros = np.count_nonzero(net_i)

            es[i, f] = (mean_product - mean_m_f) * np.sqrt(count_non_zeros) / std_m_f
            pv[i, f] = norm.cdf(-abs(es[i, f]))

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