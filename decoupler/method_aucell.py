"""
Method AUCell.
Code to run the AUCell method. 
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.stats import rankdata

from .pre import extract, match, rename_net, filt_min_n

from anndata import AnnData
from tqdm import tqdm


def auc(x, n_up, max_auc):
    """
    Computes Area Under the Curve (AUC) for given ranks.
    
    Parameters
    ----------
    x : np.array, list
        Sample specific rankings.
    n_up : int
        Number of top ranked features to select.
    max_auc : float
        Maximum AUC possible for this ranking.
    
    Returns
    -------
    The value of the AUC.
    """
    
    x = np.sort(x[x<n_up])
    y = np.arange(x.shape[0]) + 1
    x = np.append(x,n_up)
    return np.sum(np.diff(x) * y)/max_auc


def aucell(mat, c, net, n_up, verbose=False):
    """
    AUCell.
    
    Computes AUCell to infer biological activities.
    
    Parameters
    ----------
    mat : np.array
        Input matrix with molecular readouts.
    c : np.array
        Feature (column) names of mat.
    net : pd.Series
        Series of feature sets as lists.
    n_up : int
        Number of top ranked features to select.
    verbose : bool
        Whether to show progress.
    
    Returns
    -------
    acts : Array of activities.
    """
    
    # Rank data
    mat = rankdata(-mat, axis=1, method='ordinal')
    
    # Compute AUC per fset
    acts = np.zeros((mat.shape[0], net.shape[0]))
    for j in tqdm(range(len(net)), disable=not verbose):
        fset = net.iloc[j]
        rnks = mat[:,np.isin(c, fset)]
        x_th = np.arange(start=1, stop=rnks.shape[1]+1)
        x_th = np.sort(x_th[x_th < n_up])
        max_auc = np.sum(np.diff(np.append(x_th,n_up)) * x_th)
        acts[:,j] = np.apply_along_axis(auc, 1, rnks, n_up=n_up, max_auc=max_auc)
    
    return acts


def run_aucell(mat, net, source='source', target='target', weight='weight', 
               n_up=None, min_n=5, seed=42, verbose=False, use_raw=True):
    """
    AUCell.
    
    Wrapper to run AUCell.
    
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
    n_up : int
        Number of top ranked features to select as observed features.
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
    Returns aucell activity estimates or stores them in 
    `mat.obsm['aucell_estimate']`.
    """
    
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw)
    
    # Set n_up
    if n_up is None:
        n_up = np.round(0.05*len(c))
    if not 0 < n_up:
        raise ValueError('n_up needs to be a value higher than 0.')
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    net = net.groupby('source')['target'].apply(list)
    
    if verbose:
        print('Running aucell on {0} samples and {1} sources.'.format(m.shape[0], len(net)))
    
    # Run AUCell
    rng = default_rng(seed=seed)
    msk = np.arange(m.shape[1])
    rng.shuffle(msk)
    estimate = aucell(m.A[:,msk], c[msk], net, n_up, verbose=verbose)
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = 'aucell_estimate'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
    else:
        return estimate
