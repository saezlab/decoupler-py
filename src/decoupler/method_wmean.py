import numpy as np
import pandas as pd

from numpy.random import default_rng

from .pre import extract, match, rename_net, get_net_mat
from .method_wsum import wsum

from tqdm import tqdm


def wmean(mat, net):
    """
    Weighted mean (WMEAN).
    
    Computes WMEAN to infer regulator activities.
    
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
    
    # Compute WSUM
    x = wsum(mat, net)
    
    # Divide by abs sum of weights
    x = x / np.sum(np.abs(net), axis=0)
        
    return x.A


def run_wmean(mat, net, source='source', target='target', weight='weight', times=100, min_n=5, seed=42):
    """
    Wrapper to run WMEAN.
    
    Parameters
    ----------
    mat : list, pd.DataFrame or AnnData
        List of [genes, matrix], dataframe (samples x genes) or an AnnData
        instance.
    net : pd.DataFrame
        Network in long format.
    min_n : int
        Minimum of targets per TF. If less, returns 0s.
    
    Returns
    -------
    estimate : wmean activity estimates.
    norm : norm_wmean activity estimates.
    corr : corr_wmean activity estimates.
    pvals : empirical p-values of the obtained activities.
    """
    
    # Extract sparse matrix and array of genes
    m, c = extract(mat)
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    sources, targets, net = get_net_mat(net)
    
    # Match arrays
    net = match(m, c, targets, net)
    
    # Run estimate
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
        for i in tqdm(range(times)):
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
    estimate = pd.DataFrame(estimate, columns=sources)
    estimate.name = 'wmean_estimate'
    norm = pd.DataFrame(norm, columns=sources)
    norm.name = 'wmean_norm'
    corr = pd.DataFrame(corr, columns=sources)
    corr.name = 'wmean_corr'
    pvals = pd.DataFrame(pvals, columns=sources)
    pvals.name = 'wmean_pvals'
    
    return estimate, norm, corr, pvals
