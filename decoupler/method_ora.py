"""
Method ORA.
Code to run the Over Representation Analysis (ORA) method. 
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.stats import rankdata

from .pre import extract, match, rename_net, filt_min_n

from fisher import pvalue

from anndata import AnnData
from tqdm import tqdm


def get_cont_table(obs, exp, n_background=20000):
    """
    Get contingency table for ORA.
    
    Generates a contingency table (TP, FP, FN and TN) for a list of observed
    features against a list of expected features.
    
    Parameters
    ----------
    obs : list, set
        List or set of observed features.
    exp : list, set
        List or set of expected features.
    
    Returns
    -------
    Values for TP, FP, FN and TN.
    """
    
    # Transform to set
    obs, exp = set(obs), set(exp)
    
    # Build table
    TP = len(obs.intersection(exp))
    FP = len(obs.difference(exp))
    FN = len(exp.difference(obs))
    TN = n_background - TP - FP - FN
    
    return TP, FP, FN, TN 


def ora(obs, lexp, n_background=20000):
    """
    Over Representation Analysis (ORA).
    
    Computes ORA to infer regulator activities.
    
    Parameters
    ----------
    obs : list, set
        List of observed features.
    lexp : list, pd.Series
        Iterable of collections of expected features.
    
    Returns
    -------
    Array of uncorrected pvalues.
    """
    
    pvals = [pvalue(*get_cont_table(obs, fset, n_background)).right_tail for fset in lexp]
    
    return pvals


def run_ora(mat, net, source='source', target='target', weight='weight', 
            n_up=None, n_bottom=0, n_background=20000, min_n=5, 
            seed=42, verbose=False, use_raw=True):
    """
    Over Representation Analysis (ORA).
    
    Wrapper to run ORA.
    
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
    n_bottom : int
        Number of bottom ranked features to select as observed features.
    n_background : int
        Integer indicating the background size.
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
    Returns ora activity estimates (-log10(p-values)) and p-values 
    or stores them in `mat.obsm['ora_estimate']` and 
    `mat.obsm['ora_pvals']`.
    """
    
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw)
    
    # Set up/bottom masks
    if n_up is None:
        n_up = np.ceil(0.05*len(c))
    assert 0 <= n_up, 'n_up needs to be a value higher than 0.'
    assert 0 <= n_bottom, 'n_bottom needs to be a value higher than 0.'
    assert 0 <= n_background, 'n_background needs to be a value higher than 0.'
    assert (len(c) - n_up) >= n_bottom, 'n_up and n_bottom overlap, please decrase the value of any of them.'
    n_up_msk = len(c) - n_up
    n_bt_msk = n_bottom + 1
    
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    net = net.groupby('source')['target'].apply(set)
    
    if verbose:
        print('Running ora on {0} samples and {1} sources.'.format(m.shape[0], len(net)))
    
    # Run ORA
    pvals = []
    rng = default_rng(seed=seed)
    msk = np.arange(m.shape[1])
    for i in tqdm(range(m.shape[0]), disable=not verbose):
        rng.shuffle(msk)
        obs = rankdata(m[i].A[0][msk], method='ordinal')
        obs = c[msk][(obs > n_up_msk) | (obs < n_bt_msk)]
        pvals.append(ora(obs, net, n_background=n_background))
        
    # Transform to df
    pvals = pd.DataFrame(pvals, index=r, columns=net.index)
    pvals.name = 'ora_pvals'
    pvals.columns.name = None
    estimate = pd.DataFrame(-np.log10(pvals), index=r, columns=pvals.columns)
    estimate.name = 'ora_estimate'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, pvals
