"""
Method WSUM.
Code to run the Weighted sum (WSUM) method. 
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.sparse import csr_matrix

from .pre import extract, match, rename_net, get_net_mat, filt_min_n

from anndata import AnnData
from tqdm import tqdm


def wsum(mat, net):
    """
    Weighted sum (WSUM).
    
    Computes WSUM to infer regulator activities.
    
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
    
    # Mat mult
    x = mat.dot(net)
    
    return x.A


def run_wsum(mat, net, source='source', target='target', weight='weight', times=100, 
             min_n=5, seed=42, verbose=False):
    """
    Weighted sum (WSUM).
    
    Wrapper to run WSUM.
    
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
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.
    
    Returns
    -------
    estimate : wsum activity estimates.
    norm : norm_wsum activity estimates.
    corr : corr_wsum activity estimates.
    pvals : empirical p-values of the obtained activities.
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
        print('Running wsum on {0} samples and {1} sources.'.format(m.shape[0], net.shape[1]))
    
    # Run estimate
    estimate = wsum(m, net)
    
    # Permute
    norm, corr, pvals = None, None, None
    if times > 1:
        # Init null distirbution
        n_smp, n_src = estimate.shape
        null_dst = np.zeros((n_smp, n_src, times))
        pvals = np.ones(estimate.shape)
        rng = default_rng(seed=seed)
        idxs = np.arange(net.shape[0])
        
        # Permute
        for i in tqdm(range(times), disable=not verbose):
            null_dst[:,:,i] = wsum(m, net[rng.permutation(idxs)])
            pvals += np.abs(null_dst[:,:,i]) > np.abs(estimate)
        
        # Compute empirical p-value
        pvals = pvals / times
        
        # Compute z-score
        null_dst = np.array(null_dst)
        norm = (estimate - np.mean(null_dst, axis=2)) / np.std(null_dst, axis=2)
        
        # Compute corr score
        corr = estimate * -np.log10(pvals)
    
    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = 'wsum_estimate'
    norm = pd.DataFrame(norm, index=r, columns=sources)
    norm.name = 'wsum_norm'
    corr = pd.DataFrame(corr, index=r, columns=sources)
    corr.name = 'wsum_corr'
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = 'wsum_pvals'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[norm.name] = norm
        mat.obsm[corr.name] = corr
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, norm, corr, pvals
