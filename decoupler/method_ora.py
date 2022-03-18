"""
Method ORA.
Code to run the Over Representation Analysis (ORA) method.
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng

from scipy.stats import rankdata
from math import log, exp, lgamma

from .pre import extract, rename_net, filt_min_n

from anndata import AnnData
from tqdm import tqdm

import numba as nb


@nb.njit(nb.f4(nb.i4, nb.i4, nb.i4, nb.i4), cache=True)
def mlnTest2r(a, ab, ac, abcd):
    if 0 > a or a > ab or a > ac or ab + ac > abcd + a:
        raise ValueError('invalid contingency table')
    a_min = max(0, ab + ac - abcd)
    a_max = min(ab, ac)
    if a_min == a_max:
        return 0.
    p0 = lgamma(ab + 1) + lgamma(ac + 1) + lgamma(abcd - ac + 1) + lgamma(abcd - ab + 1) - lgamma(abcd + 1)
    pa = lgamma(a + 1) + lgamma(ab - a + 1) + lgamma(ac - a + 1) + lgamma(abcd - ab - ac + a + 1)
    if ab * ac > a * abcd:
        sl = 0.
        for i in range(a - 1, a_min - 1, -1):
            sl_new = sl + exp(pa - lgamma(i + 1) - lgamma(ab - i + 1) - lgamma(ac - i + 1) - lgamma(abcd - ab - ac + i + 1))
            if sl_new == sl:
                break
            sl = sl_new
        return -log(1. - max(0, exp(p0 - pa) * sl))
    else:
        sr = 1.
        for i in range(a + 1, a_max + 1):
            sr_new = sr + exp(pa - lgamma(i + 1) - lgamma(ab - i + 1) - lgamma(ac - i + 1) - lgamma(abcd - ab - ac + i + 1))
            if sr_new == sr:
                break
            sr = sr_new
        return max(0, pa - p0 - log(sr))


@nb.njit(nb.f4(nb.i4, nb.i4, nb.i4, nb.i4), cache=True)
def test1r(a, b, c, d):
    """
    Code adapted from:
    https://github.com/painyeph/FishersExactTest/blob/master/fisher.py
    """

    return exp(-mlnTest2r(a, a + b, a + c, a + b + c + d))


@nb.njit(nb.f4[:](nb.i4[:], nb.i4[:], nb.i4[:], nb.i4[:], nb.i4), parallel=True, cache=True)
def get_pvals(sample, net, starts, offsets, n_background):

    # Init vals
    sample = set(sample)
    n_fsets = offsets.shape[0]
    pvals = np.zeros(n_fsets, dtype=nb.f4)
    for i in nb.prange(n_fsets):

        # Extract feature set
        srt = starts[i]
        off = offsets[i] + srt
        fset = set(net[srt:off])

        # Build table
        a = len(sample.intersection(fset))
        b = len(fset.difference(sample))
        c = len(sample.difference(fset))
        d = n_background - a - b - c

        # Store
        pvals[i] = test1r(a, b, c, d)

    return pvals


def ora(mat, net, n_up_msk, n_bt_msk, n_background=20000, verbose=False):

    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int32)
    net = np.concatenate(net.values)

    # Define starts to subset offsets
    starts = np.zeros(offsets.shape[0], dtype=np.int32)
    starts[1:] = np.cumsum(offsets)[:-1]

    # Init empty
    pvls = np.zeros((mat.shape[0], offsets.shape[0]), dtype=np.float32)
    ranks = np.arange(mat.shape[1], dtype=np.int32)
    for i in tqdm(range(mat.shape[0]), disable=not verbose):

        # Find ranks
        sample = rankdata(mat[i].A, method='ordinal').astype(np.int32)
        sample = ranks[(sample > n_up_msk) | (sample < n_bt_msk)]

        # Estimate pvals
        pvls[i] = get_pvals(sample, net, starts, offsets, n_background)

    return pvls


def run_ora(mat, net, source='source', target='target', n_up=None, n_bottom=0, n_background=20000, min_n=5, seed=42,
            verbose=False, use_raw=True):
    """
    Over Representation Analysis (ORA).

    ORA measures the overlap between the target feature set and a list of most altered molecular features in `mat`.
    The most altered molecular features can be selected from the top and or bottom of the molecular readout distribution, by
    default it is the top 5% positive values. With these, a contingency table is build and a one-tailed Fisher’s exact test is
    computed to determine if a regulator’s set of features are over-represented in the selected features from the data.
    The resulting score, `ora_estimate`, is the minus log10 of the obtained p-value.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData
        instance.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    n_up : int
        Number of top ranked features to select as observed features.
    n_bottom : int
        Number of bottom ranked features to select as observed features.
    n_background : int
        Integer indicating the background size.
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
    estimate : DataFrame
        ORA scores, which are the -log(p-values). Stored in `.obsm['ora_estimate']` if `mat` is AnnData.
    pvals : DataFrame
        Obtained p-values. Stored in `.obsm['ora_pvals']` if `mat` is AnnData.
    """

    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Set up/bottom masks
    if n_up is None:
        n_up = np.ceil(0.05*len(c))
    if not 0 <= n_up:
        raise ValueError('n_up needs to be a value higher than 0.')
    if not 0 <= n_bottom:
        raise ValueError('n_bottom needs to be a value higher than 0.')
    if not 0 <= n_background:
        raise ValueError('n_background needs to be a value higher than 0.')
    if not (len(c) - n_up) >= n_bottom:
        raise ValueError('n_up and n_bottom overlap, please decrase the value of any of them.')
    n_up_msk = len(c) - n_up
    n_bt_msk = n_bottom + 1

    # Transform net
    net = rename_net(net, source=source, target=target, weight=None)
    net = filt_min_n(c, net, min_n=min_n)

    # Randomize feature order to break ties randomly
    rng = default_rng(seed=seed)
    idx = np.arange(m.shape[1])
    rng.shuffle(idx)
    m, c = m[:, idx], c[idx]

    # Transform targets to indxs
    table = {name: i for i, name in enumerate(c)}
    net['target'] = [table[target] for target in net['target']]
    net = net.groupby('source')['target'].apply(lambda x: np.array(x, dtype=np.int32))
    if verbose:
        print('Running ora on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), len(net)))

    # Run ORA
    pvals = ora(m, net, n_up_msk, n_bt_msk, n_background, verbose)

    # Transform to df
    pvals = pd.DataFrame(pvals, index=r, columns=net.index)
    pvals.name = 'ora_pvals'
    estimate = pd.DataFrame(-np.log10(pvals), index=r, columns=pvals.columns)
    estimate.name = 'ora_estimate'

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, pvals
