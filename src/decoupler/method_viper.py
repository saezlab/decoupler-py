import numpy as np
import pandas as pd

import scipy.stats.stats
from scipy.stats import rankdata
from scipy.stats import norm

from .pre import extract, match, rename_net, get_net_mat, filt_min_n

from anndata import AnnData


def viper(mat, net):
    nes = np.sqrt(np.sum(net**2,axis=0))
    msk = np.sum(net != 0, axis=1) == 1
    wts = (net / np.sum(net, axis=0))[msk]
    net = np.sign(wts)
    t2 = rankdata(mat, method='average', axis=1) / (mat.shape[1] + 1)
    t1 = np.abs(t2 - 0.5) * 2
    t1 = t1 + (1 - np.max(t1))/2
    t1 = norm.ppf(t1[:,msk])
    t2 = norm.ppf(t2[:,msk])
    sum1 = t2.dot(wts)
    sum2 = t1.dot((1 - np.abs(net)) * wts)
    ss = np.sign(sum1)
    ss[ss == 0] = 1
    nes = ((np.abs(sum1) + sum2 * (sum2 > 0)) * ss) * nes
    
    return nes

def run_viper(mat, net, source='source', target='target', weight='weight', min_n=5, verbose=False):
    """
    Virtual Inference of Protein-activity by Enriched Regulon (VIPER).
    
    Wrapper to run VIPER.
    
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
    verbose : bool
        Whether to show progress.
    
    Returns
    -------
    estimate : viper activity estimates.
    pvals : p-values of the obtained activities.
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
        print('Running viper on {0} samples and {1} sources.'.format(m.shape[0], net.shape[1]))
    
    # Run estimate
    estimate = viper(m.A, net.A)
    
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