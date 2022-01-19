"""
Method GSVA.
Code to run the Gene Set Variation Analysis (GSVA) method. 
"""

import numpy as np
import pandas as pd

from scipy.stats import norm

from .pre import extract, match, rename_net, filt_min_n

from anndata import AnnData
from tqdm import tqdm


def init_cdfs():
    pre_cdf = norm.cdf(np.arange(pre_res+1) * max_pre / pre_res, loc=0, scale=1)
    
    return pre_cdf


def col_d(x, sigma_factor=4.0):
    size = x.shape[0]
    bw = (np.std(x, ddof=1) / sigma_factor)
    left_tail = (x[:,np.newaxis] - x[np.newaxis,:]) / bw
    cdf = (np.abs(left_tail)/max_pre * pre_res).astype(int)
    cdf = pre_cdf[np.where(cdf > pre_res, -1, cdf)]
    left_tail = np.where(left_tail < 0, 1.0 - cdf, cdf)
    left_tail = np.sum(left_tail,axis=1)/size 
    left_tail = -1.0 * np.log((1.0-left_tail)/left_tail)
    return left_tail


def density(mat):
    global pre_res, max_pre, pre_cdf, precomp_cdf
    pre_res, max_pre = 10000, 10
    pre_cdf = init_cdfs()
    
    D = np.zeros(mat.shape)
    for i in tqdm(range(mat.shape[1])):
        D[:,i] = col_d(mat[:,i])
    return D


def rank_score(idxs, rev_idx):
    return rev_idx[idxs]
       
    
def get_D_I(mat):
    D = density(mat)
    n = D.shape[1]
    rev_idx = np.abs(np.arange(start=n, stop=0, step=-1) - n / 2)
    I = np.argsort(np.argsort(-D, axis=1))
    for j in range(D.shape[0]):
        D[j] = rank_score(I[j], rev_idx)
    # NEED TO ARGSORT BACK AGAIN
    I = np.argsort(I)
    return D, I
    
    
def ks_set(D, I, c, fset, tau = 1, mx_diff = True, abs_rnk = False):
    msk = np.isin(c, fset)
    f_idxs = np.where(np.isin(c, fset))[0]
    s_idxs = np.arange(D.shape[0])
    n_ftrs = len(c)
    n_set = len(fset)
    dec = 1.0 / (n_ftrs - n_set)
    sum_gset = np.sum(np.power(D[:,f_idxs], tau), axis=1)
    mx_value = 0.0
    cum_sum = 0.0
    mx_pos = 0.0
    mx_neg = 0.0

    for j in range(n_ftrs):
        idx = I[:,j]
        cum_sum += np.where(msk[idx], np.power(D[s_idxs,idx], tau) / sum_gset, -dec)
        csum_bgr = cum_sum > mx_pos
        mx_pos = np.where(csum_bgr, cum_sum, mx_pos)
        csum_lwr = cum_sum < mx_neg
        mx_neg = np.where(csum_lwr, cum_sum, mx_neg)

    if mx_diff != 0:
        mx_value = mx_pos + mx_neg
        if abs_rnk != 0:
            mx_value = mx_pos - mx_neg
    else:
        mx_value = np.where(mx_pos > np.abs(mx_neg), mx_pos, mx_neg)
    
    return mx_value
    
    
def run_gsva(mat, net, source='source', target='target', weight='weight', 
             mx_diff = True, abs_rnk = False, min_n=5, verbose=False):
    """
    Gene Set Variation Analysis (GSVA).
    
    Wrapper to run GSVA.
    
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
    mx_diff : bool
        Changes how the enrichment statistic (ES) is calculated. If True (default),
        ES is calculated as the difference between the maximum positive and negative
        random walk deviations. If False, ES is calculated as the maximum positive
        to 0. 
    abs_rnk : bool
        Used when mx_diff = True. If False (default), the enrichment statistic (ES) 
        is calculated taking the magnitude difference between the largest positive 
        and negative random walk deviations. If True, feature sets with features 
        enriched on either extreme (high or low) will be regarded as 'highly' 
        activated.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    verbose : bool
        Whether to show progress. 
    
    Returns
    -------
    estimate : -log10 of the obtained p-values.
    """
    
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat)
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    net = net.groupby('source')['target'].apply(list)
    
    if verbose:
        print('Running gsva on {0} samples and {1} sources.'.format(m.shape[0], len(net)))
    
    # Get feature Density
    D, I = get_D_I(m.A)
    
    # Run GSVA
    estimate = np.zeros((len(r), len(net)))
    for j in tqdm(range(len(net)), disable=not verbose):
        fset = net.iloc[j]
        es = ks_set(D, I, c, fset)
        estimate[:,j] = es
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = 'gsva_estimate'
    
    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
    else:
        return estimate
