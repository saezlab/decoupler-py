"""
Method ULM.
Code to run the Univariate Linear Model (ULM) method. 
"""

import numpy as np
import pandas as pd

import scipy.stats.stats

from .pre import extract, match, rename_net, get_net_mat, filt_min_n

from anndata import AnnData
from tqdm import tqdm


def ulm(mat, net, TINY = 1.0e-20, verbose=False):
    """
    Univariate Linear Model (ULM).
    
    Computes ULM to infer regulator activities.
    
    Parameters
    ----------
    mat : np.array
        Input matrix with molecular readouts.
    net : np.array
        Regulatory adjacency matrix.
    verbose : bool
        Whether to show progress.
    
    Returns
    -------
    x : Array of activities.
    """
    
    # Init empty matrix of activities
    x = np.zeros((mat.shape[0], net.shape[1]))
    
    df, n_repeat = net.shape
    df = df - 2
    
    for i in tqdm(range(mat.shape[0]), disable=not verbose):
        # Get row
        mat_row = mat[i]
        
        # Repeat mat_row for each regulator
        smp_mat = np.repeat([mat_row], n_repeat, axis=0)
    
        # Compute lm
        cov = np.cov(net.T, smp_mat, bias=1)
        ssxm, ssym = np.split(np.diag(cov), 2)
        ssxym = np.diag(cov, k=len(net.T))
        
        # Compute R value
        r = ssxym / np.sqrt(ssxm * ssym)
        
        # Compute t-value
        x[i] = r * np.sqrt(df / ((1.0 - r + TINY)*(1.0 + r + TINY)))
        
    return x


def run_ulm(mat, net, source='source', target='target', weight='weight', min_n=5, verbose=False):
    """
    Univariate Linear Model (ULM).
    
    Wrapper to run ULM.
    
    Parameters
    ----------
    mat : list, pd.DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData
        instance.
    net : pd.DataFrame
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
    
    Returns
    -------
    Returns ulm activity estimates and p-values or stores them in 
    `mat.obsm['ulm_estimate']` and `mat.obsm['ulm_pvals']`.
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
        print('Running ulm on {0} samples and {1} sources.'.format(m.shape[0], net.shape[1]))
    
    # Run estimate
    estimate = ulm(m.A, net.A, verbose=verbose)
    
    # Get pvalues
    df = net.shape[0] - 2
    _, pvals = scipy.stats.stats._ttest_finish(df, estimate, 'two-sided')
    
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
