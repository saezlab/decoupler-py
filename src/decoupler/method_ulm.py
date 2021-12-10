import numpy as np
import pandas as pd

import scipy.stats.stats

from .pre import extract, match, rename_net, get_net_mat

from tqdm import tqdm


def ulm(mat, net, TINY = 1.0e-20):
    """
    Univariate Linear Model (ULM).
    
    Computes ULM to infer regulator activities.
    
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
    
    # Init empty matrix of activities
    x = np.zeros((mat.shape[0], net.shape[1]))
    
    df, n_repeat = net.shape
    df = df - 2
    
    for i in tqdm(range(mat.shape[0])):
        # Get row
        mat_row = mat[i].A
        
        # Repeat mat_row for each regulator
        smp_mat = np.repeat(mat_row, n_repeat, axis=0)
    
        # Compute lm
        cov = np.cov(net.T, smp_mat, bias=1)
        ssxm, ssym = np.split(np.diag(cov), 2)
        ssxym = np.diag(cov, k=len(net.T))
        
        # Compute R value
        r = ssxym / np.sqrt(ssxm * ssym)
        
        # Compute t-value
        x[i] = r * np.sqrt(df / ((1.0 - r + TINY)*(1.0 + r + TINY)))
        
    return x


def run_ulm(mat, net, source='source', target='target', weight='weight', min_n=5):
    """
    Wrapper to run ULM.
    
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
    estimate : ulm activity estimates.
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
    estimate = ulm(m, net.A)
    
    # Get pvalues
    df = net.shape[0] - 2
    _, pvals = scipy.stats.stats._ttest_finish(df, estimate, 'two-sided')
    
    # Transform to df
    estimate = pd.DataFrame(estimate, columns=sources)
    estimate.name = 'ulm_estimate'
    pvals = pd.DataFrame(pvals, columns=sources)
    pvals.name = 'ulm_pvals'
    
    return estimate, pvals

