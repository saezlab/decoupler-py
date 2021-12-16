"""
Method MLM.
Code to run the Multivariate Linear Model (MLM) method. 
"""

import numpy as np
import pandas as pd

from .pre import extract, match, rename_net, get_net_mat, filt_min_n

from anndata import AnnData
from scipy import linalg, stats
from tqdm import tqdm


def mlm(mat, net):
    """
    Multivariate Linear Model (MLM).
    
    Computes MLM to infer regulator activities.
    
    Parameters
    ----------
    mat : csr_matrix
        Gene expression matrix.
    net : csr_matrix
        Regulatory adjacency matrix.
    
    Returns
    -------
    x : Array of activities.
    """
    
    X = np.hstack([np.ones((net.shape[0],)).reshape(-1,1), net])
    y = mat.T.A

    coef, residues, rank, singular = linalg.lstsq(X, y, check_finite=False)
    df = X.shape[0] - X.shape[1]
    sse = np.sum((X.dot(coef) - y)**2, axis=0) / df
    inv = np.linalg.inv(np.dot(X.T, X))
    se = np.array([np.sqrt(np.diagonal(sse[i] * inv)) for i in range(sse.shape[0])])
    t = coef.T/se
    
    return t[:,1:]


def run_mlm(mat, net, source='source', target='target', weight='weight', min_n=5, verbose=False):
    """
    Multivariate Linear Model (MLM).
    
    Wrapper to run MLM.
    
    Parameters
    ----------
    mat : list, pd.DataFrame or AnnData
        List of [genes, matrix], dataframe (samples x genes) or an AnnData
        instance.
    net : pd.DataFrame
        Network in long format.
    source : str
        Column name with source nodes.
    target : str
        Column name with target nodes.
    weight : str
        Column name with weights.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    verbose : bool
        Whether to show progress. 
    
    Returns
    -------
    estimate : activity estimates.
    pvals : p-values of the obtained activities.
    """
    
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat)
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    sources, targets, net = get_net_mat(net)
    
    # Match arrays
    net = match(m, c, targets, net)
    
    if verbose:
        print('Running mlm on {0} samples and {1} sources.'.format(m.shape[0], net.shape[1]))
    
    # Run estimate
    estimate = mlm(m, net.A)
    
    # Get pvalues
    pvals = 2 * (1 - stats.t.cdf(np.abs(estimate), m.shape[1] - net.shape[1]))
    
    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = 'mlm_estimate'
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = 'mlm_pvals'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, pvals
