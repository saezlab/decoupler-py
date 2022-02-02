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
import numba as nb


@nb.njit(nb.f4[:,:](nb.f4[:,:]), parallel=True)
def rankdata(mat):
    for i in nb.prange(mat.shape[0]):
        mat[i] = np.argsort(np.argsort(-mat[i])) + 1
    return mat


@nb.njit(nb.f4[:,:](nb.f4[:,:], nb.i4[:], nb.i4[:], nb.i4), parallel=True)
def auc(mat, net, offsets, n_up):
    starts = np.zeros(offsets.shape[0], dtype=nb.i4)
    starts[1:] = np.cumsum(offsets)[:-1]
    acts = np.zeros((mat.shape[0], offsets.shape[0]), dtype=nb.f4)
    for j in nb.prange(offsets.shape[0]):
        srt = starts[j]
        off = offsets[j] + srt
        fset = net[srt:off]
        x_th = np.arange(start=1, stop=fset.shape[0]+1, dtype=nb.i4)
        x_th = x_th[x_th < n_up]
        max_auc = np.sum(np.diff(np.append(x_th, n_up)) * x_th)
        for i in nb.prange(mat.shape[0]):
            x = mat[i][fset]
            x = np.sort(x[x < n_up])
            y = np.arange(x.shape[0]) + 1
            x = np.append(x,n_up)
            acts[i,j] = np.sum(np.diff(x) * y)/max_auc
    return acts


def aucell(mat, net, n_up):
    """
    AUCell.
    
    Computes AUCell to infer biological activities.
    
    Parameters
    ----------
    mat : np.array
        Input matrix with molecular readouts.
    net : pd.Series
        Series of feature sets as lists.
    n_up : int
        Number of top ranked features to select.
    
    Returns
    -------
    acts : Array of activities.
    """
    
    # Rank data
    mat = rankdata(mat)
    
    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int32)
    net = np.concatenate(net.values)
    
    # Compute AUC per fset
    acts = auc(mat, net, offsets, n_up)
    
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
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)
    
    # Set n_up
    if n_up is None:
        n_up = np.round(0.05*len(c))
    if not 0 < n_up:
        raise ValueError('n_up needs to be a value higher than 0.')
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    
    # Randomize feature order to break ties randomly
    rng = default_rng(seed=seed)
    idx = np.arange(m.shape[1])
    rng.shuffle(idx)
    m, c = m[:,idx], c[idx]
    
    # Transform targets to indxs
    table = {name:i for i,name in enumerate(c)}
    net['target'] = [table[target] for target in net['target']]
    net = net.groupby('source')['target'].apply(lambda x: np.array(x, dtype=np.int32))
    
    if verbose:
        print('Running aucell on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), len(net)))
    
    # Run AUCell
    
    estimate = aucell(m.A, net, n_up)
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = 'aucell_estimate'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
    else:
        return estimate
