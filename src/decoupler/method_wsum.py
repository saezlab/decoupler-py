import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.sparse import csr_matrix

from .pre import extract, match, rename_net, get_net_mat

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


def run_wsum(mat, net, source='source', target='target', weight='weight', times=100, min_n=5, seed=42):
    """
    Wrapper to run WSUM.
    
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
    estimate : wsum activity estimates.
    norm : norm_wsum activity estimates.
    corr : corr_wsum activity estimates.
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
        for i in tqdm(range(times)):
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
    estimate = pd.DataFrame(estimate, columns=sources)
    norm = pd.DataFrame(norm, columns=sources)
    corr = pd.DataFrame(corr, columns=sources)
    pvals = pd.DataFrame(pvals, columns=sources)
    
    return estimate, norm, corr, pvals