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


@nb.njit(nb.types.Tuple((nb.f4[:,:], nb.i4[:,:]))(nb.f4[:,:]), cache=True)
def get_M_I(mat):
    I = np.zeros(mat.shape, dtype=nb.i4)
    for i in range(mat.shape[0]):
        I[i] = np.argsort(-mat[i])
    mat = np.abs(mat)
    return mat, I


@nb.njit(nb.f4(nb.f4[:], nb.i4), cache=True)
def std(arr, ddof):
    N = arr.shape[0]
    m = np.mean(arr)
    var = np.sum((arr - m)**2)/(N-ddof)
    sd = np.sqrt(var)
    return sd


@nb.njit(nb.f4(nb.f4[:], nb.i4[:], nb.i4, nb.i4[:], nb.i4[:], nb.i4, nb.f4), cache=True)
def ks_sample(D, I, n_genes, geneset_mask, fset, n_geneset, dec):
    
    sum_gset = 0.0
    for i in nb.prange(n_geneset):
        sum_gset += D[fset[i]]
        
    if sum_gset == 0.0:
        return 0.0

    mx_value = 0.0
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

    if np.round(mx_pos, 4) == np.round(-mx_neg, 4):
        mx_value = 0.0
    elif mx_pos > -mx_neg:
        mx_value = mx_pos
    else:
        mx_value = mx_neg
    
    return mx_value


@nb.njit(nb.f4[:](nb.f4[:,:], nb.i4[:,:], nb.i4[:]), cache=True)
def ks_matrix(D, I, fset):
    n_samples, n_genes = D.shape
    n_geneset = fset.shape[0]
    
    geneset_mask = np.zeros(n_genes, dtype=nb.i4)
    geneset_mask[fset] = 1
    
    dec = 1.0 / (n_genes - n_geneset)
    
    res = np.zeros(n_samples, dtype=nb.f4)
    for i in nb.prange(n_samples):
        res[i] = ks_sample(D[i], I[i], n_genes, geneset_mask, fset, n_geneset, dec)
    
    return res


@nb.njit(nb.f4[:](nb.f4[:,:], nb.i4[:,:], nb.i4[:], nb.f4[:], nb.i4[:,:]), cache=True)
def ks_perms(D, I, fset, es, m_msk):
    times = m_msk.shape[0]
    if times == 0:
        return es
    res = np.zeros((times, D.shape[0]), dtype=nb.f4)
    for i in nb.prange(times):
        res[i] = ks_matrix(D, I[:,m_msk[i]], fset)
    
    null_mean = np.zeros(D.shape[0], dtype=nb.f4)
    null_std = np.zeros(D.shape[0], dtype=nb.f4)
    for j in nb.prange(D.shape[0]):
        null_mean[j] = res[:,j].mean()
        null_std[j] = std(res[:,j], 1)
        
    nes = (es - null_mean) / null_std
    
    return nes
    
    
@nb.njit(nb.types.UniTuple(nb.f4[:,:],2)(nb.f4[:,:], nb.i4[:,:], nb.i4[:], nb.i4[:], nb.i4, nb.i4), parallel=True, cache=True)
def nb_gsea(D, I, net, offsets, times, seed):
    
    np.random.seed(seed)
    
    n_samples = D.shape[0]
    n_gsets = offsets.shape[0]
    m_es = np.zeros((n_samples, n_gsets), dtype=nb.f4)
    m_nes = np.zeros(m_es.shape, dtype=nb.f4)
    
    # Generate random shuffling matrix
    m_msk = np.zeros((times, D.shape[1]), dtype=nb.i4)
    idxs = np.arange(D.shape[1])
    for i in range(times):
        np.random.shuffle(idxs)
        m_msk[i] = idxs
    
    starts = np.zeros(n_gsets, dtype=nb.i4)
    starts[1:] = np.cumsum(offsets)[:-1]
    for j in nb.prange(n_gsets):
        
        srt = starts[j]
        off = offsets[j] + srt
        fset = net[srt:off]
        es = ks_matrix(D, I, fset)
        m_es[:,j] = es
        m_nes[:,j] = ks_perms(D, I, fset, es, m_msk)
    
    return m_es, m_nes

    
def gsea(mat, net, times=1000, batch_size = 10000, seed=42, verbose=False):
    
    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int32)
    net = np.concatenate(net.values)
    
    # Get number of batches
    n_samples = mat.shape[0]
    n_fsets = offsets.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    
    # Init empty acts
    es = np.zeros((n_samples, n_fsets), dtype=np.float32)
    nes = np.zeros((n_samples, n_fsets), dtype=np.float32)
    
    for i in tqdm(range(n_batches), disable=not verbose):
        
        # Subset batch
        srt, end = i*batch_size, i*batch_size+batch_size
        sub_mat = mat[srt:end].A
        
        # Get modified m and I matrix
        sub_mat, I = get_M_I(sub_mat)
        
        # Compute GSEA per batch
        es[srt:end], nes[srt:end] = nb_gsea(sub_mat, I, net, offsets, times, seed)
    
    if times != 0:
        pvals = norm.cdf(-np.abs(nes)) * 2
        return es, nes, pvals
    else:
        return es, None, None
    
    
def run_gsea(mat, net, source='source', target='target', weight='weight', 
             times=1000, batch_size = 10000, min_n=5, seed=42, verbose=False, 
             use_raw=True):
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
    batch_size : int
        Size of the samples to use for each batch. Increasing this will consume more 
        memmory but it will run faster.
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
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    
    # Transform targets to indxs
    table = {name:i for i,name in enumerate(c)}
    net['target'] = [table[target] for target in net['target']]
    net = net.groupby('source')['target'].apply(lambda x: np.array(x, dtype=np.int32))
    
    if verbose:
        print('Running gsea on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), len(net)))
    
    # Run GSEA
    estimate, norm_e, pvals = gsea(m, net, times=times, batch_size=batch_size, 
                                   seed=seed, verbose=verbose)
    
    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = 'gsea_estimate'
    if norm_e is not None:
        norm_e = pd.DataFrame(norm_e, index=r, columns=net.index)
        norm_e.name = 'gsea_norm'
        pvals = pd.DataFrame(pvals, index=r, columns=net.index)
        pvals.name = 'gsea_pvals'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        if norm_e is not None:
            mat.obsm[norm_e.name] = norm_e
            mat.obsm[pvals.name] = pvals
    else:
        if pvals is not None:
            return estimate, norm_e, pvals
        else:
            return estimate
