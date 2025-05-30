from typing import Tuple

import numpy as np
import scipy.stats as sts
from tqdm.auto import tqdm
import numba as nb

from decoupler._docs import docs
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
    pos_idxs = np.zeros(idxs.shape[0], dtype=np.int_)
    for j in range(idxs.shape[0]):
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
    idxs = np.zeros((size, 2), dtype=np.int_)
    k = 0
    for i in range(pval.shape[0]):
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
    s1: np.ndarray,
    s2: np.ndarray,
) -> np.ndarray:
    n_fsets = reg.shape[1]
    col = np.full(n_fsets, np.nan)
    for k in nb.prange(n_fsets):
        if k != j:
            k_msk = reg[:, k] != 0
            nhits = k_msk.sum()
            if nhits > n_targets:
                sum1 = np.sum(reg[:, k] * s2)
                ss = np.sign(sum1)
                if ss == 0:
                    ss = 1
                sum2 = np.sum((1- np.abs(reg[k_msk, k])) * s1[k_msk])
                ww = np.ones(nhits)
                col[k] = (np.abs(sum1) + sum2 * (sum2 > 0)) / ww.size * ss * np.sqrt(ww.size)
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
        pval[j] = _fill_pval_mat(j=j, reg=reg, n_targets=n_targets, s1=s1, s2=s2)
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
    wts = np.zeros(sub_net.shape)
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
    table = np.zeros(int(unique.max()) + 1, dtype=np.int_)
    table[unique.astype(np.int_)] = counts
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


@docs.dedent
def _func_viper(
    mat: np.ndarray,
    adj: np.ndarray,
    pleiotropy: bool = True,
    reg_sign: float = 0.05,
    n_targets: int | float = 10,
    penalty: int | float = 20,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Virtual Inference of Protein-activity by Enriched Regulon analysis (VIPER) :cite:`viper`.

    This approach first ranks features based on their absolute values and computes a one-tail score.

    .. math::

        \begin{align}
        w &= \frac{w}{max(|w|)} \\
        l_{orig} &= 1_{w \neq 0} \\
        l &= \frac{l_{orig}}{\sum_{i=1}^{k} \frac{l_i}{max(l_{orig})}max(l_{orig})} \\
        q^{norm} &= \Phi^{-1}(2|q-0.5| + (1 + max(|q-0.5|))) \\
        S_1 &= \sum_{i=1}^{k}q_i^{norm}l_i(1-|w_i|) \\
        \end{align}
        
    Where:

    - :math:`w \in [-1, +1]` is a vector of interaction weights across features
    - :math:`l \in [0, 1]` is a vector of interaction likelihoods across features
    - :math:`q \in [0, 1]` is a vector of quantiles based on the molecular readouts across features
    - :math:`k` is the number of features in :math:`q`
    - :math:`\Phi^{-1}` is is the inverse of the cumulative distribution function of the standard normal distribution
    - :math:`q^{norm} \in [-\infty,+\infty]` are the z-scores of the deviation of quantiles from 0.5

    :math:`S_1` encodes for the magnitude of the enrichment score, irrespective of the interaction signs in ``net``.

    Then, :math:`q` are z-transformed and weighted by their interaction strength and likelihood.

    .. math::

        S_2 = \sum_{i=1}^{k}w_il_i(\Phi^{-1}(q_i))

    In this case, :math:`S_2` takes the direction (sign) of interactions into consideration.

    Afterwards, a summary score :math:`S_3` is obtained.

    .. math::

        S_3 = 
        \begin{cases}
        (|S_2| + S_1)  \times \mathrm{sgn}(S_2) & \text{if } S_1 > 0 \\
        S_2 & \text{if } S_1 < 0
        \end{cases}

    An enrichment score :math:`ES` is obtained by comparing :math:`S_3` to a
    null model generated through an analytical approach that shuffles features.

    .. math::

        ES = S_3\sqrt{\sum_{i=1}^{k}l_{orig,i}^{2}}
        
    Together with a :math:`p_{value}`

    .. math::

        p_{value} = \Phi(ES)

    Additionaly, computing multiple sources simultaneously, a pleiotropic correction is employed.

    In brief, all possible pairs of sources AB are generated under two conditions:
    
    1. both A and B are significantly enriched (p < ``reg_sign=0.05``)
    2. they share at least ``n_targets=10`` features

    Subsequently, a :math:`ES` and its associated :math:`p_{value}` is computed for
    both A (:math:`pA`) and B (:math:`pB`) based only on the shared features.
    Then the pleiotropy score (:math:`PS`) is computed.

    .. math::

        PS = 
        \begin{cases}
        \frac{1}{(1+|\log_{10}(pB) - \log_{10}(pA)|)^{\frac{20}{n_a}}} \text{ if } pA < pB \\
        \frac{1}{(1+|\log_{10}(pA) - \log_{10}(pB)|)^{\frac{20}{n_b}}} \text{ if } pA > pB
        \end{cases}
    
    Where:

    - :math:`n_a` is the number of test pairs involving the source A
    - :math:`n_b` is the number of test pairs involving the source B

    This score is used to update :math:`l_{orig}`.

    .. math::

        l_{orig, i} = 
        \begin{cases}
        PS \times 1_{\{i \in A\}} \text{ if } pA < pB \\
        PS \times 1_{\{i \in B\}} \text{ if } pA > pB
        \end{cases}

    A new :math:`ES` and :math:`p_{value}` are calculated following all
    the previous steps but using the updated :math:`l_{orig}`
    
    %(yestest)s

    %(params)s
    
    pleiotropy
        Whether correction for pleiotropic regulation should be performed.
    reg_sign
        If ``pleiotropy``, p-value threshold for considering significant regulators.
    n_targets
        If ``pleiotropy``, integer indicating the minimal number of overlaping targets to consider for analysis.
    penalty
        If ``pleiotropy``, number higher than 1 indicating the penalty for the pleiotropic interactions. 1 = no penalty.

    %(returns)s
    """
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
            tmp = _aREA(ss_i.reshape(1, -1), sub_net, wts=wts)[0]
            nes[i, idxs] = tmp
    # Get pvalues
    pvals = 2 * sts.norm.sf(np.abs(nes))
    return nes, pvals


_viper = MethodMeta(
    name='viper',
    desc='Virtual Inference of Protein-activity by Enriched Regulon analysis (VIPER)',
    func=_func_viper,
    stype='numerical',
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference='https://doi.org/10.1038/ng.3593',
)
viper = Method(_method=_viper)
