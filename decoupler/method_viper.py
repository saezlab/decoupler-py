"""
Method VIPER.
Code to run the Virtual Inference of Protein-activity by Enriched Regulon (VIPER) method.
"""

import numpy as np
import pandas as pd

from scipy.stats import rankdata
from scipy.stats import norm

from .pre import extract, match, rename_net, get_net_mat, filt_min_n

from anndata import AnnData
from tqdm import tqdm

import numba as nb


@nb.njit(nb.types.Tuple((nb.f4[:, :], nb.i4[:]))(nb.f4[:, :], nb.i4[:, :], nb.f4[:], nb.i4[:], nb.i4), cache=True)
def get_wts_posidxs(wts, idxs, pval1, table, penalty):
    pos_idxs = np.zeros(idxs.shape[0], dtype=nb.i4)
    for j in nb.prange(idxs.shape[0]):
        p = pval1[j]
        if p > 0:
            x_idx, y_idx = idxs[j]
        else:
            y_idx, x_idx = idxs[j]
        pos_idxs[j] = x_idx

        x = wts[:, x_idx]
        y = wts[:, y_idx]
        x_msk, y_msk = x != 0, y != 0
        msk = x_msk * y_msk
        x[msk] = x[msk] / (1 + np.abs(p))**(penalty/table[x_idx])
        wts[:, x_idx] = x

    return wts, pos_idxs


@nb.njit(nb.types.Tuple((nb.f4[:, :], nb.i4[:, :]))(nb.f4[:, :]), cache=True)
def get_tmp_idxs(pval):

    size = int(np.sum(~np.isnan(pval)) / 2)

    tmp = np.zeros((size, 2), dtype=nb.f4)
    idxs = np.zeros((size, 2), dtype=nb.i4)

    k = 0
    for i in nb.prange(pval.shape[0]):
        for j in range(pval.shape[1]):
            if i <= j:
                x = pval[i, j]
                if not np.isnan(x):
                    y = pval[j, i]
                    if not np.isnan(y):
                        tmp[k, 0] = x
                        tmp[k, 1] = y
                        idxs[k, 0] = i
                        idxs[k, 1] = j
                        k += 1

    return tmp, idxs


@nb.njit(nb.f4[:](nb.i4, nb.f4[:, :], nb.i4, nb.f4[:]), cache=True)
def fill_pval_mat(j, reg, n_targets, s2):
    n_fsets = reg.shape[1]
    col = np.full(n_fsets, np.nan, dtype=nb.f4)
    for k in nb.prange(n_fsets):
        if k != j:
            k_msk = reg[:, k] != 0
            if k_msk.sum() >= n_targets:
                sum1 = np.sum(reg[:, k] * s2)
                ss = np.sign(sum1)
                if ss == 0:
                    ss = 1
                ww = np.abs(reg[:, k]) / np.max(np.abs(reg[:, k]))
                col[k] = np.abs(sum1) / np.sum(np.abs(reg[:, k])) * ss * np.sqrt(np.sum(ww**2))
    return col


def get_inter_pvals(nes_i, ss_i, sub_net, n_targets):
    pval = np.full((sub_net.shape[1], sub_net.shape[1]), np.nan, dtype=np.float32)
    for j in range(sub_net.shape[1]):

        trgt_msk = sub_net[:, j] != 0

        reg = ((sub_net[trgt_msk] != 0) * sub_net[trgt_msk, j].reshape(-1, 1)).astype(np.float32)

        s2 = ss_i[trgt_msk]
        s2 = rankdata(s2, method='average') / (s2.shape[0]+1) * 2 - 1
        s1 = np.abs(s2) * 2 - 1
        s1 = s1 + (1 - np.max(s1)) / 2
        s1 = norm.ppf(s1/2 + 0.5)
        tmp = np.sign(nes_i[j])
        if tmp == 0:
            tmp = 1
        s2 = (norm.ppf(s2/2 + 0.5) * tmp).astype(np.float32)

        pval[j] = fill_pval_mat(j, reg, n_targets, s2)

    pval = 1 - norm.cdf(pval)

    return pval.astype(np.float32)


def shadow_regulon(nes_i, ss_i, net, reg_sign=1.96, n_targets=10, penalty=20):

    # Find significant activities
    msk_sign = np.abs(nes_i) > reg_sign

    # Filter by significance
    nes_i = nes_i[msk_sign]
    sub_net = net[:, msk_sign]

    # Init likelihood mat
    wts = np.zeros(sub_net.shape, dtype=np.float32)
    wts[sub_net != 0] = 1.0

    if wts.shape[1] < 2:
        return None

    # Get significant interatcions between regulators
    pval = get_inter_pvals(nes_i, ss_i, sub_net, n_targets=n_targets)

    # Get pairs of regulators
    tmp, idxs = get_tmp_idxs(pval)

    if tmp.size == 0:
        return None

    pval1 = np.log10(tmp[:, 1]) - np.log10(tmp[:, 0])
    unique, counts = np.unique(idxs.flatten(), return_counts=True)

    table = np.zeros(unique.max()+1, dtype=np.int32)
    table[unique] = counts

    # Modify interactions based on sign of pval1
    wts, pos_idxs = get_wts_posidxs(wts, idxs, pval1, table, penalty)

    # Select only regulators with positive pval1
    pos_idxs = np.unique(pos_idxs)
    sub_net = sub_net[:, pos_idxs]
    wts = wts[:, pos_idxs]
    idxs = np.where(msk_sign)[0][pos_idxs]

    return sub_net, wts, idxs


def aREA(mat, net, wts=None):

    if wts is None:
        wts = np.zeros(net.shape)
        wts[net != 0] = 1

    # Normalize net between -1 and 1
    net = net / np.max(np.abs(net), axis=0)

    wts = wts / np.max(wts, axis=0)
    nes = np.sqrt(np.sum(wts**2, axis=0))
    wts = (wts / np.sum(wts, axis=0))

    mat = rankdata(mat, method='average', axis=1) / (mat.shape[1] + 1)
    t1 = np.abs(mat - 0.5) * 2
    t1 = t1 + (1 - np.max(t1))/2
    msk = np.sum(net != 0, axis=1) >= 1
    t1, mat = t1[:, msk], mat[:, msk]
    net, wts = net[msk], wts[msk]
    t1 = norm.ppf(t1)
    mat = norm.ppf(mat)
    sum1 = mat.dot(wts*net)
    sum2 = t1.dot((1-np.abs(net)) * wts)
    tmp = (np.abs(sum1) + sum2 * (sum2 > 0)) * np.sign(sum1)
    nes = tmp * nes

    return nes


def viper(mat, net, pleiotropy=True, reg_sign=0.05, n_targets=10, penalty=20, batch_size=10000, verbose=False):

    # Get number of batches
    n_samples = mat.shape[0]
    n_features, n_fsets = net.shape
    n_batches = int(np.ceil(n_samples / batch_size))

    if verbose:
        print('Infering activities on {0} batches.'.format(n_batches))

    # Init empty acts
    nes = np.zeros((n_samples, n_fsets), dtype=np.float32)
    for i in tqdm(range(n_batches), disable=not verbose):

        # Subset batch
        srt, end = i*batch_size, i*batch_size+batch_size
        tmp = mat[srt:end].A

        # Compute VIPER for batch
        nes[srt:end] = aREA(tmp, net)

    if pleiotropy:
        if verbose:
            print('Computing pleiotropy correction.')
        reg_sign = norm.ppf(1-(reg_sign / 2))
        for i in tqdm(range(nes.shape[0]), disable=not verbose):

            # Extract per sample
            ss_i = mat[i].A[0]
            nes_i = nes[i]

            # Shadow regulons
            shadow = shadow_regulon(nes_i, ss_i, net, reg_sign=reg_sign, n_targets=n_targets, penalty=penalty)
            if shadow is None:
                continue
            else:
                sub_net, wts, idxs = shadow

            # Recompute activity with shadow regulons and update nes
            tmp = aREA(ss_i.reshape(1, -1), sub_net, wts=wts)[0]
            nes[i, idxs] = tmp

    # Get pvalues
    pvals = norm.cdf(-np.abs(nes)) * 2

    return nes, pvals


def run_viper(mat, net, source='source', target='target', weight='weight', pleiotropy=True, reg_sign=0.05, n_targets=10,
              penalty=20, batch_size=10000, min_n=5, verbose=False, use_raw=True):
    """
    Virtual Inference of Protein-activity by Enriched Regulon (VIPER).

    VIPER (Alvarez et al., 2016) estimates biological activities by performing a three-tailed enrichment score calculation. For
    further information check the supplementary information of the decoupler mansucript or the original publication.

    Alvarez M.J.et al. (2016) Functional characterization of somatic mutations in cancer using network-based inference of
    protein activity. Nat. Genet., 48, 838–847.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    net : DataFrame
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
        Number higher than 1 indicating the penalty for the pleiotropic interactions. 1 = no penalty.
    batch_size : int
        Size of the batches to use. Increasing this will consume more memmory but it will run faster.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.

    Returns
    -------
    estimate : DataFrame
        VIPER scores. Stored in `.obsm['viper_estimate']` if `mat` is AnnData.
    pvals : DataFrame
        Obtained p-values. Stored in `.obsm['viper_pvals']` if `mat` is AnnData.
    """

    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    sources, targets, net = get_net_mat(net)

    # Match arrays
    net = match(c, targets, net)

    if verbose:
        print('Running viper on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0],
                                                                                              len(c), net.shape[1]))

    # Run VIPER
    estimate, pvals = viper(m, net, pleiotropy=pleiotropy, reg_sign=reg_sign, n_targets=n_targets, penalty=penalty,
                            batch_size=batch_size, verbose=verbose)

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
