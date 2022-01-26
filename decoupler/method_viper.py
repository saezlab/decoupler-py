import numpy as np
import pandas as pd

import scipy.stats.stats
from scipy.stats import rankdata
from scipy.stats import norm

from .pre import extract, match, rename_net, get_net_mat, filt_min_n

from anndata import AnnData


def viper(mat, net, wts=None):
    """
    Virtual Inference of Protein-activity by Enriched Regulon (VIPER).
    
    Computes VIPER to infer regulator activities.
    
    Parameters
    ----------
    mat : np.array
        Input matrix with molecular readouts.
    net : np.array
        Regulatory adjacency matrix.
    
    Returns
    -------
    nes : Array of biological activities.
    """
    
    if wts is None:
        wts = np.zeros(net.shape)
        wts[net != 0] = 1
    
    wts = wts / np.max(wts, axis=0)
    nes = np.sqrt(np.sum(wts**2,axis=0))
    wts = (wts / np.sum(wts, axis=0))
    
    t2 = rankdata(mat, method='average', axis=1) / (mat.shape[1] + 1)
    t1 = np.abs(t2 - 0.5) * 2
    t1 = t1 + (1 - np.max(t1))/2
    msk = np.sum(net != 0, axis=1) >= 1
    t1, t2 = t1[:,msk], t2[:,msk]
    net, wts = net[msk], wts[msk]
    t1 = norm.ppf(t1)
    t2 = norm.ppf(t2)
    sum1 = t2.dot(wts*net)
    sum2 = (1-np.abs(net))
    nes = sum1 * nes
    
    return nes


def run_viper(mat, net, source='source', target='target', weight='weight', min_n=5, 
              verbose=False, use_raw=True):
    """
    Virtual Inference of Protein-activity by Enriched Regulon (VIPER).
    
    Wrapper to run VIPER.
    
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
    use_raw : bool
        Use raw attribute of mat if present.
    
    Returns
    -------
    Returns viper activity estimates and p-values or stores them in 
    `mat.obsm['viper_estimate']` and `mat.obsm['viper_pvals']`.
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
        print('Running viper on {0} samples and {1} sources.'.format(m.shape[0], net.shape[1]))
    
    # Run VIPER
    estimate = viper(m.A, net)
    
    # Get pvalues
    pvals = norm.cdf(-np.abs(estimate)) * 2
    
    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = 'viper_estimate'
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = 'viper_pvals'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, pvals
