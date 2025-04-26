from typing import Tuple

import numpy as np
import scipy.stats as sts
from tqdm.auto import tqdm
import numba as nb

from decoupler._log import _log
from decoupler._Method import MethodMeta, Method


@nb.njit(cache=True)
def _get_wts_posidxs(
    wts: np.ndarray,
    idxs: np.ndarray,
    pval1: np.ndarray,
    table: np.ndarray,
    penalty: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_idxs = np.zeros(idxs.shape[0])
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


@nb.njit(cache=True)
def _get_tmp_idxs(
    pval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    size = int(np.sum(~np.isnan(pval)) / 2)
    tmp = np.zeros((size, 2))
    idxs = np.zeros((size, 2))
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


@nb.njit(cache=True)
def _fill_pval_mat(
    j: int,
    reg: np.ndarray,
    n_targets: int,
    s2: np.ndarray,
) -> np.ndarray:
    n_fsets = reg.shape[1]
    col = np.full(n_fsets, np.nan)
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


def _get_inter_pvals(
    nes_i: np.ndarray,
    ss_i: np.ndarray,
    sub_net: np.ndarray,
    n_targets: int,
) -> np.ndarray:
    pval = np.full((sub_net.shape[1], sub_net.shape[1]), np.nan)
    for j in range(sub_net.shape[1]):
        trgt_msk = sub_net[:, j] != 0
        reg = ((sub_net[trgt_msk] != 0) * sub_net[trgt_msk, j].reshape(-1, 1))
        s2 = ss_i[trgt_msk]
        s2 = sts.rankdata(s2, method='average') / (s2.shape[0]+1) * 2 - 1
        s1 = np.abs(s2) * 2 - 1
        s1 = s1 + (1 - np.max(s1)) / 2
        s1 = sts.norm.ppf(s1/2 + 0.5)
        tmp = np.sign(nes_i[j])
        if tmp == 0:
            tmp = 1
        s2 = (sts.norm.ppf(s2/2 + 0.5) * tmp)
        pval[j] = _fill_pval_mat(j, reg, n_targets, s2)
    pval = 1 - sts.norm.cdf(pval)
    return pval


def _shadow_regulon(
    nes_i: np.ndarray,
    ss_i: np.ndarray,
    net: np.ndarray,
    reg_sign: float = 1.96,
    n_targets: int | float = 10,
    penalty: int | float = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    pval = _get_inter_pvals(nes_i, ss_i, sub_net, n_targets=n_targets)
    # Get pairs of regulators
    tmp, idxs = _get_tmp_idxs(pval)
    if tmp.size == 0:
        return None
    pval1 = np.log10(tmp[:, 1]) - np.log10(tmp[:, 0])
    unique, counts = np.unique(idxs.flatten(), return_counts=True)
    table = np.zeros(unique.max()+1, dtype=np.int64)
    table[unique] = counts
    # Modify interactions based on sign of pval1
    wts, pos_idxs = _get_wts_posidxs(wts, idxs, pval1, table, penalty)
    # Select only regulators with positive pval1
    pos_idxs = np.unique(pos_idxs)
    sub_net = sub_net[:, pos_idxs]
    wts = wts[:, pos_idxs]
    idxs = np.where(msk_sign)[0][pos_idxs]
    return sub_net, wts, idxs


def _aREA(
    mat: np.ndarray,
    net: np.ndarray,
    wts: None = None
) -> np.ndarray:
    if wts is None:
        wts = np.zeros(net.shape)
        wts[net != 0] = 1
    # Normalize net between -1 and 1
    net = net / np.max(np.abs(net), axis=0)
    wts = wts / np.max(wts, axis=0)
    nes = np.sqrt(np.sum(wts**2, axis=0))
    wts = (wts / np.sum(wts, axis=0))
    mat = sts.rankdata(mat, method='average', axis=1) / (mat.shape[1] + 1)
    t1 = np.abs(mat - 0.5) * 2
    t1 = t1 + (1 - np.max(t1))/2
    msk = np.sum(net != 0, axis=1) >= 1
    t1, mat = t1[:, msk], mat[:, msk]
    net, wts = net[msk], wts[msk]
    t1 = sts.norm.ppf(t1)
    mat = sts.norm.ppf(mat)
    sum1 = mat.dot(wts*net)
    sum2 = t1.dot((1-np.abs(net)) * wts)
    tmp = (np.abs(sum1) + sum2 * (sum2 > 0)) * np.sign(sum1)
    nes = tmp * nes
    return nes


def _func_viper(
    mat: np.ndarray,
    adj: np.ndarray,
    pleiotropy: bool = True,
    reg_sign: float = 0.05,
    n_targets: int | float = 10,
    penalty: int | float = 20,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    # Get number of batches
    n_samples = mat.shape[0]
    n_features, n_fsets = adj.shape
    m = f'viper - calculating {n_fsets} scores across {n_samples} observations'
    _log(m, level='info', verbose=verbose)
    # Compute score
    nes = _aREA(mat, adj)
    if pleiotropy:
        m = f'viper - refining scores based on pleiotropy'
        _log(m, level='info', verbose=verbose)
        reg_sign = sts.norm.ppf(1-(reg_sign / 2))
        for i in tqdm(range(nes.shape[0]), disable=not verbose):
            # Extract per sample
            ss_i = mat[i]
            nes_i = nes[i]
            # Shadow regulons
            shadow = _shadow_regulon(nes_i, ss_i, adj, reg_sign=reg_sign, n_targets=n_targets, penalty=penalty)
            if shadow is None:
                continue
            else:
                sub_net, wts, idxs = shadow
            # Recompute activity with shadow regulons and update nes
            tmp = aREA(ss_i.reshape(1, -1), sub_net, wts=wts)[0]
            nes[i, idxs] = tmp
    # Get pvalues
    pvals = sts.norm.cdf(-np.abs(nes)) * 2
    return nes, pvals


_viper = MethodMeta(
    name='viper',
    func=_func_viper,
    stype='numerical',
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference='https://doi.org/10.1038/ng.3593',
)
viper = Method(_method=_viper)
