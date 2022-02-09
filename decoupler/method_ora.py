"""
Method ORA.
Code to run the Over Representation Analysis (ORA) method. 
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.stats import rankdata

from .pre import extract, match, rename_net, filt_min_n

from fisher import pvalue_npy

from anndata import AnnData
from tqdm import tqdm

import numba as nb


@nb.njit(nb.uint[:,:](nb.i4[:], nb.i4[:], nb.i4[:], nb.i4[:], nb.i4), parallel=True, cache=True)
def get_cont_table(sample, net, starts, offsets, n_background):
    
    sample = set(sample)
    n_fsets = offsets.shape[0]
    table = np.zeros((n_fsets,4), dtype=nb.uint)
    
    for i in nb.prange(n_fsets):
        # Extract feature set
        srt = starts[i]
        off = offsets[i] + srt
        fset = set(net[srt:off])
        
        # Build table
        table[i,0] = len(sample.intersection(fset))
        table[i,1] = len(fset.difference(sample))
        table[i,2] = len(sample.difference(fset))
        table[i,3] = n_background - table[i,0] - table[i,1] - table[i,2]
        
    return table


def ora(mat, net, n_up_msk, n_bt_msk, n_background=20000, verbose=False):
    
    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int32)
    net = np.concatenate(net.values)

    # Define starts to subset offsets
    starts = np.zeros(offsets.shape[0], dtype=np.int32)
    starts[1:] = np.cumsum(offsets)[:-1]

    pvls = np.zeros((mat.shape[0], offsets.shape[0]), dtype=np.float32)
    ranks = np.arange(mat.shape[1], dtype=np.int32)

    for i in tqdm(range(mat.shape[0]), disable=not verbose):
        sample = rankdata(mat[i].A, method='ordinal').astype(np.int32)
        sample = ranks[(sample > n_up_msk) | (sample < n_bt_msk)]

        # Generate Table
        table = get_cont_table(sample, net, starts, offsets, n_background)

        # Estimate pvals
        _, pvls[i], _ = pvalue_npy(table[:,0],table[:,1],table[:,2],table[:,3])
    
    return pvls


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
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)
    
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
    
    # Transform targets to indxs
    table = {name:i for i,name in enumerate(c)}
    net['target'] = [table[target] for target in net['target']]
    net = net.groupby('source')['target'].apply(lambda x: np.array(x, dtype=np.int32))
    
    if verbose:
        print('Running ora on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), len(net)))
    
    # Run ORA
    pvals = ora(m, net, n_up_msk, n_bt_msk, n_background, verbose)
        
    # Transform to df
    pvals = pd.DataFrame(pvals, index=r, columns=net.index)
    pvals.name = 'ora_pvals'
    estimate = pd.DataFrame(-np.log10(pvals), index=r, columns=pvals.columns)
    estimate.name = 'ora_estimate'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, pvals
