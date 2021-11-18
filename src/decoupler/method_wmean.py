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
    div = np.sum(np.abs(net), axis=0)
    x = x / div
        
    return x


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
    sources, targets, regX = get_net_mat(net)
    
    # Match arrays
    regX = match(m, c, targets, regX)
    
    # Run estimate
    estimate = wmean(m, regX)
    
    # Permute
    norm, corr, pvals = None, None, None
    if times > 1:
        # Init null distirbution
        n_src, n_tgt = estimate.shape
        null_dst = np.zeros((n_src, n_tgt, times))
        pvals = np.zeros(estimate.shape)
        rng = default_rng(seed=seed)
        
        # Permute
        for i in tqdm(range(times)):
            null_dst[:,:,i] = wmean(m, rng.permutation(regX))
            pvals += np.abs(null_dst[:,:,i]) > np.abs(estimate)
        
        # Compute empirical p-value
        pvals = pvals / times
        pvals[pvals == 0] = 1 / times
        
        # Compute z-score
        null_dst = np.array(null_dst)
        norm = (estimate - np.mean(null_dst, axis=2)) / np.std(null_dst, axis=2)
        
        # Compute corr score
        corr = estimate * -np.log10(pvals)
    
    # Transform to df
    estimate = pd.DataFrame(estimate, columns=sources)
    norm = pd.DataFrame(norm, columns=sources)
    corr = pd.DataFrame(corr, columns=sources)
    pvals = pd.DataFrame(pvals, columns=sources)
    
    return estimate, norm, corr, pvals
