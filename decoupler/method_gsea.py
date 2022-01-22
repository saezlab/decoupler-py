"""
Method GSEA.
Code to run the Gene Set Enrichment Analysis (GSEA) method. 
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.stats import norm

from .pre import extract, match, rename_net, filt_min_n
from .method_gsva import ks_set

from anndata import AnnData
from tqdm import tqdm


def get_M_I(mat):
    I = np.argsort(-mat, axis=1)
    mat = np.abs(mat)
    return mat, I


def gsea(mat, c, net, times=1000, seed=42, verbose=False):
    """
    Gene Set Enrichment Analysis (GSEA).
    
    Computes GSEA to infer biological activities.
    
    Parameters
    ----------
    mat : np.array
        Input matrix with molecular readouts.
    c : np.array
        Feature (column) names of mat.
    net : pd.Series
        Series of feature sets as lists.
    times : int
        How many random permutations to do.
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.
    
    Returns
    -------
    TODO.
    """
    
    # Get modified m and I matrix
    mat, I = get_M_I(mat)
    
    # Randomize columns
    rng = default_rng(seed=seed)
    msk = np.arange(mat.shape[1])
    
    m_es = np.zeros((mat.shape[0], len(net)))
    m_nes, pvals = None, None
    
    # Run GSEA for each feature set
    for j in tqdm(range(len(net)), disable=not verbose):
        fset = net.iloc[j]
        es = ks_set(mat, I, c, fset)
        # Run times random iterations
        if times > 0:
            res = []
            for i in np.arange(times):
                rng.shuffle(msk)
                res.append(ks_set(mat[:,msk], I[:,msk], c[msk], fset))
                
            # Compute z-score
            res = np.array(res)
            null_mean = np.mean(res, axis=0)
            null_std = np.std(res, ddof=1, axis=0)
            nes = (es - null_mean) / null_std
            if m_nes is None:
                m_nes = np.zeros((mat.shape[0], len(net)))
            m_nes[:,j] = nes
        m_es[:,j] = es
        
    if m_nes is not None:
        # Get pvalues
        pvals = norm.cdf(-np.abs(m_nes)) * 2
        
    return m_es, m_nes, pvals
    
    
def run_gsea(mat, net, source='source', target='target', weight='weight', 
             times=10, min_n=5, seed=42, verbose=False, use_raw=True):
    """
    Gene Set Enrichment Analysis (GSEA).
    
    Wrapper to run GSEA.
    
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
    Returns gsea, norm_gsea activity estimates and p-values or stores
    them in `mat.obsm['gsea_estimate']`, `mat.obsm['gsea_norm']`, and 
    `mat.obsm['gsea_pvals']`.
    """
    
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw)
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    net = net.groupby('source')['target'].apply(list)
    
    if verbose:
        print('Running gsea on {0} samples and {1} sources.'.format(m.shape[0], len(net)))
    
    # Run GSEA
    estimate, norm_e, pvals = gsea(m.A, c, net, times=times, seed=seed, verbose=verbose)
    
    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = 'gsea_estimate'
    norm_e = pd.DataFrame(norm_e, index=r, columns=net.index)
    norm_e.name = 'gsea_norm'
    pvals = pd.DataFrame(pvals, index=r, columns=net.index)
    pvals.name = 'gsea_pvals'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[norm_e.name] = norm_e
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, norm_e, pvals
