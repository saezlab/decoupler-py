import numpy as np
import pandas as pd

import scipy.stats.stats

from .pre import extract, match, rename_net, get_net_mat

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


def run_mlm(mat, net, source='source', target='target', weight='weight', min_n=5):
    """
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
        Minimum of targets per TF. If less, returns 0s.
    
    Returns
    -------
    estimate : activity estimates.
    pvals : p-values of the obtained activities.
    """
    
    # Extract sparse matrix and array of genes
    m, c = extract(mat)
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    sources, targets, net = get_net_mat(net)
    
    # Match arrays
    net = match(m, c, targets, net)
    
    # Run estimate
    estimate = mlm(m, net.A)
    
    # Get pvalues
    pvals = 2 * (1 - stats.t.cdf(np.abs(estimate), m.shape[1] - net.shape[1]))
    
    # Transform to df
    estimate = pd.DataFrame(estimate, columns=sources)
    estimate.name = 'mlm_estimate'
    pvals = pd.DataFrame(pvals, columns=sources)
    pvals.name = 'mlm_pvals'
    
    return estimate, pvals

