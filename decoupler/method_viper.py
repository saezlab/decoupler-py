import numpy as np
import pandas as pd

import scipy.stats.stats
from scipy.stats import rankdata
from scipy.stats import norm

from .pre import extract, match, rename_net, get_net_mat, filt_min_n

from anndata import AnnData
from tqdm import tqdm


def get_tmp_idxs(pval):
    tmp = []
    idxs = []
    for i in range(pval.shape[0]):
        for j in range(pval.shape[1]):
            if i <= j:
                x = pval[i,j]
                if not np.isnan(x):
                    y = pval[j,i]
                    if not np.isnan(y):
                        tmp.append([x, y])
                        idxs.append([i, j])
    tmp = np.array(tmp)
    idxs = np.array(idxs)
    return tmp, idxs
    
    
def get_inter_pvals(nes_i, ss_i, sub_net, n_targets=10):
    
    pval = np.full((sub_net.shape[1], sub_net.shape[1]), np.nan)
    for j in range(sub_net.shape[1]):

        trgt_msk = sub_net[:,j] != 0

        reg = (sub_net[trgt_msk] != 0) * sub_net[trgt_msk,j].reshape(-1,1)

        s2 = ss_i[trgt_msk]
        s2 = rankdata(s2, method='average') / (s2.shape[0]+1) * 2 - 1
        s1 = np.abs(s2) * 2 - 1
        s1 = s1 + (1 - np.max(s1)) / 2
        s1 = norm.ppf(s1/2 + 0.5)
        tmp = np.sign(nes_i[j])
        if tmp == 0: tmp = 1
        s2 = norm.ppf(s2/2 + 0.5) * tmp

        for k in range(sub_net.shape[1]):
            if k == j:
                continue

            reg_k = reg[:,k]
            k_msk = reg_k != 0
            if k_msk.sum() < n_targets:
                continue
            sum1 = np.sum(reg_k * s2)
            ss = np.sign(sum1)
            if ss == 0:
                ss = 1
            ww = np.abs(reg_k) / np.max(np.abs(reg_k))
            pval[j, k] = np.abs(sum1) / np.sum(np.abs(reg_k)) * ss * np.sqrt(np.sum(ww**2))
    
    pval = 1 - norm.cdf(pval)
    
    return pval


def shadow_regulon(nes_i, ss_i, net, reg_sign=0.05, n_targets=10, penalty=20):
    
    # Find significant activities
    pval = (1-norm.cdf(np.abs(nes_i))) * 2
    msk_sign = pval < reg_sign
    
    # Filter by significance
    nes_i = nes_i[msk_sign]
    sub_net = net[:,msk_sign]
    
    # Init likelihood mat
    wts = np.zeros(sub_net.shape)
    wts[sub_net != 0] = 1.0  
    
    if wts.shape[1] < 2:
        return None
        
    # Get significant interatcions between regulators
    pval = get_inter_pvals(nes_i, ss_i, sub_net, n_targets=n_targets)
    
    # Get pairs of regulators
    tmp, idxs = get_tmp_idxs(pval)
    
    if tmp.size == 0:
        return None
    
    pval1 = np.log10(tmp[:,1]) - np.log10(tmp[:,0])
    unique, counts = np.unique(idxs.flatten(), return_counts=True)
    table = dict(zip(unique, counts))
    
    # Modify interactions based on sign of pval1
    pos_idxs = []
    for j in range(tmp.shape[0]):
        p = pval1[j]
        if p > 0:
            x_idx, y_idx = idxs[j]
        else:
            y_idx, x_idx = idxs[j]
        pos_idxs.append(x_idx)

        x = wts[:,x_idx]
        y = wts[:,y_idx]
        x_msk, y_msk = x != 0, y != 0
        msk = x_msk * y_msk
        x[msk] = x[msk] / (1 + np.abs(p))**(penalty/table[x_idx])
        wts[:,x_idx] = x
        
    # Select only regulators with positive pval1
    pos_idxs = np.unique(pos_idxs)
    sub_net = sub_net[:,pos_idxs]
    wts = wts[:,pos_idxs]
    idxs = np.where(msk_sign)[0][pos_idxs]
    
    return sub_net, wts, idxs


def aREA(mat, net, wts=None):
    
    if wts is None:
        wts = np.zeros(net.shape)
        wts[net != 0] = 1
    
    # Normalize net between -1 and 1
    net = net / np.max(np.abs(net), axis=0)
    
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


def viper(mat, net, pleiotropy = True, reg_sign = 0.05, n_targets = 10, 
          penalty = 20, verbose=False):
    """
    Virtual Inference of Protein-activity by Enriched Regulon (VIPER).
    
    Computes VIPER to infer regulator activities.
    
    Parameters
    ----------
    mat : np.array
        Input matrix with molecular readouts.
    net : np.array
        Regulatory adjacency matrix.
    pleiotropy : bool
        Logical, whether correction for pleiotropic regulation should be performed.
    reg_sign : float
        Pleiotropy argument. p-value threshold for considering significant regulators.
    n_targets : int
        Pleiotropy argument. Integer indicating the minimal number of overlaping targets
        to consider for analysis.
    penalty : int
        Number higher than 1 indicating the penalty for the pleiotropic 
        interactions. 1 = no penalty.
    verbose : bool
        Whether to show progress.
    
    Returns
    -------
    nes : Array of biological activities.
    """
    
    # Acivity estimate
    nes = aREA(mat, net)
    
    if pleiotropy:
        for i in tqdm(range(nes.shape[0]), disable=not verbose):  

            # Extract per sample
            ss_i = mat[i]
            nes_i = nes[i]

            # Shadow regulons
            shadow = shadow_regulon(nes_i, ss_i, net, reg_sign=reg_sign, 
                                    n_targets=n_targets, penalty=penalty)
            if shadow is None:
                continue
            else:
                sub_net, wts, idxs = shadow

            # Recompute activity with shadow regulons and update nes
            tmp = aREA(ss_i.reshape(1,-1), sub_net, wts=wts)[0]
            nes[i,idxs] = tmp
    
    return nes


def run_viper(mat, net, source='source', target='target', weight='weight', 
              pleiotropy = True, reg_sign = 0.05, n_targets = 10, penalty = 20, 
              min_n=5, verbose=False, use_raw=True):
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
    pleiotropy : bool
        Logical, whether correction for pleiotropic regulation should be performed.
    reg_sign : float
        Pleiotropy argument. p-value threshold for considering significant regulators.
    n_targets : int
        Pleiotropy argument. Integer indicating the minimal number of overlaping targets
        to consider for analysis.
    penalty : int
        Number higher than 1 indicating the penalty for the pleiotropic 
        interactions. 1 = no penalty.
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
    estimate = viper(m.A, net, pleiotropy=pleiotropy, reg_sign=reg_sign, 
                     n_targets=n_targets, penalty=penalty, verbose=verbose)
    
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
