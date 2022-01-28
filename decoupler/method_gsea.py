"""
Method GSEA.
Code to run the Gene Set Enrichment Analysis (GSEA) method. 
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.stats import norm

from .pre import extract, match, rename_net, filt_min_n

from anndata import AnnData
from tqdm import tqdm

import numba as nb


def get_M_I(mat):
    I = np.argsort(-mat, axis=1)
    mat = np.abs(mat).astype(np.float64)
    return mat, I


@nb.njit(nb.f8(nb.f8[:], nb.i8))
def std(arr, ddof):
    N = arr.shape[0]
    m = np.mean(arr)
    var = np.sum((arr - m)**2)/(N-ddof)
    sd = np.sqrt(var)
    return sd


@nb.njit(nb.f8(nb.f8[:], nb.i8[:], nb.i8, nb.i8[:], nb.i8[:], nb.i8, nb.f8))
def ks_sample(D, I, n_genes, geneset_mask, fset, n_geneset, dec):
    
    sum_gset = 0.0
    for i in nb.prange(n_geneset):
        sum_gset += D[fset[i]]
    
    mx_value_sign = 0.0
    cum_sum = 0.0
    mx_pos = 0.0
    mx_neg = 0.0
    
    for i in nb.prange(n_genes):
        idx = I[i]
        if geneset_mask[idx] == 1:
            cum_sum += D[idx] / sum_gset
        else:
            cum_sum -= dec
            
        if cum_sum > mx_pos: mx_pos = cum_sum
        if cum_sum < mx_neg: mx_neg = cum_sum 
        
    mx_value_sign = mx_pos + mx_neg
    
    return mx_value_sign


@nb.njit(nb.f8[:](nb.f8[:,:], nb.i8[:,:], nb.i8[:]))
def ks_matrix(D, I, fset):
    n_samples, n_genes = D.shape
    n_geneset = fset.shape[0]
    
    geneset_mask = np.zeros(n_genes, dtype=nb.i8)
    geneset_mask[fset] = 1
    
    dec = 1.0 / (n_genes - n_geneset)
    
    res = np.zeros(n_samples)
    for i in nb.prange(n_samples):
        res[i] = ks_sample(D[i], I[i], n_genes, geneset_mask, fset, n_geneset, dec)
    
    return res


@nb.njit(nb.f8[:](nb.f8[:,:], nb.i8[:,:], nb.i8[:], nb.f8[:], nb.i8))
def ks_perms(D, I, fset, es, times):
    res = np.zeros((times, D.shape[0]))
    msk = np.arange(D.shape[1])
    if times == 0:
        return es
    for i in nb.prange(times):
        np.random.shuffle(msk)
        res[i] = ks_matrix(D, I[:,msk], fset)
    
    null_mean = np.zeros(D.shape[0])
    null_std = np.zeros(D.shape[0])
    for j in nb.prange(D.shape[0]):
        null_mean[j] = res[:,j].mean()
        null_std[j] = std(res[:,j], 1)
        
    nes = (es - null_mean) / null_std
    
    return nes
    
    
@nb.njit(nb.types.UniTuple(nb.f8[:,:],2)(nb.f8[:,:], nb.i8[:,:], nb.i8[:], nb.i8[:], nb.i8, nb.i8), parallel=True)
def ks_sets(D, I, net, offsets, times, seed):
    
    n_samples = D.shape[0]
    n_gsets = offsets.shape[0]
    m_es = np.zeros((n_samples, n_gsets))
    m_nes = np.zeros(m_es.shape)
    
    np.random.seed(seed)
    
    starts = np.zeros(n_gsets, dtype=nb.i8)
    starts[1:] = np.cumsum(offsets)[:-1]
    for j in nb.prange(n_gsets):
        srt = starts[j]
        off = offsets[j] + srt
        fset = net[srt:off]
        es = ks_matrix(D, I, fset)
        m_es[:,j] = es
        m_nes[:,j] = ks_perms(D, I, fset, es, times)
    
    return m_es, m_nes

    
def gsea(mat, net, times=100, seed=42, verbose=False):
    """
    Gene Set Enrichment Analysis (GSEA).
    
    Computes GSEA to infer biological activities.
    
    Parameters
    ----------
    mat : np.array
        Input matrix with molecular readouts.
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
    
    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values
    net = np.concatenate(net.values)
    
    # Get modified m and I matrix
    mat, I = get_M_I(mat)
    
    # Randomize columns
    rng = default_rng(seed=seed)
    msk = np.arange(mat.shape[1])
    
    m_es, m_nes = ks_sets(mat, I, net, offsets, times, seed)
        
    pvals = norm.cdf(-np.abs(m_nes)) * 2
        
    return m_es, m_nes, pvals
    
    
def run_gsea(mat, net, source='source', target='target', weight='weight', 
             times=100, min_n=5, seed=42, verbose=False, use_raw=True):
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
    
    # Transform targets to indxs
    table = dict()
    table = {name:i for i,name in enumerate(c)}
    net['target'] = [table[target] for target in net['target']]
    net = net.groupby('source')['target'].apply(np.array)
    
    if verbose:
        print('Running gsea on {0} samples and {1} sources.'.format(m.shape[0], len(net)))
    
    # Run GSEA
    estimate, norm_e, pvals = gsea(m.A, net, times=times, seed=seed, verbose=verbose)
    
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
