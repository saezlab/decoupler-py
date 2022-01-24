"""
Method WMEAN.
Code to run the Weighted mean (WMEAN) method. 
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng

from .pre import extract, match, rename_net, get_net_mat, filt_min_n
from .method_wsum import wsum

from anndata import AnnData
from tqdm import tqdm


def wmean(mat, net):
    """
    Weighted mean (WMEAN).
    
    Computes WMEAN to infer regulator activities.
    
    Parameters
    ----------
    mat : csr_matrix
        Input matrix with molecular readouts.
    net : csr_matrix
        Regulatory adjacency matrix.
    
    Returns
    -------
    x : Array of activities.
    """
    
    # Compute WSUM
    x = wsum(mat, net)
    
    # Divide by abs sum of weights
    x = x / np.sum(np.abs(net), axis=0)
        
    return x


def run_wmean(mat, net, source='source', target='target', weight='weight', times=100, 
              min_n=5, seed=42, verbose=False, use_raw=True):
    """
    Weighted mean (WMEAN).
    
    Wrapper to run WMEAN.
    
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
    times : int
        How many random permutations to do.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.
    
    Returns
    -------
    Returns wmean, norm_wmean, corr_wmean activity estimates and p-values 
    or stores them in `mat.obsm['wmean_estimate']`, `mat.obsm['wmean_norm']`,
    `mat.obsm['wmean_corr']` and `mat.obsm['wmean_pvals']`.
    """
    
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw)
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    sources, targets, net = get_net_mat(net)
    
    # Match arrays
    net = match(c, targets, net)
    
    if verbose:
        print('Running wmean on {0} samples and {1} sources.'.format(m.shape[0], net.shape[1]))
    
    # Run WMEAN
    estimate = wmean(m, net)
    
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
            null_dst[:,:,i] = wmean(m, net[rng.permutation(idxs)])
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
    estimate.name = 'wmean_estimate'
    norm = pd.DataFrame(norm, index=r, columns=sources)
    norm.name = 'wmean_norm'
    corr = pd.DataFrame(corr, index=r, columns=sources)
    corr.name = 'wmean_corr'
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = 'wmean_pvals'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[norm.name] = norm
        mat.obsm[corr.name] = corr
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, norm, corr, pvals
