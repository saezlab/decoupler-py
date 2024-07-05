"""
Method ORA.
Code to run the Over Representation Analysis (ORA) method.
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.sparse import csr_matrix

from scipy.stats import rankdata
from math import log, exp, lgamma

from .pre import extract, rename_net, filt_min_n, return_data
from .utils import p_adjust_fdr

from tqdm import tqdm

import numba as nb


@nb.njit(nb.f8(nb.i8, nb.i8, nb.i8, nb.i8), cache=True)
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


@nb.njit(nb.f8(nb.i8, nb.i8, nb.i8, nb.i8), cache=True)
def test1r(a, b, c, d):
    """
    Code adapted from:
    https://github.com/painyeph/FishersExactTest/blob/master/fisher.py
    """

    return exp(-mlnTest2r(a, a + b, a + c, a + b + c + d))


@nb.njit(nb.types.Tuple((nb.i8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.b1[:, :]))
         (nb.i8[:], nb.i8[:], nb.i8[:], nb.i8[:], nb.i8, nb.i8), parallel=True, cache=True)
def get_pvals(sample, net, starts, offsets, n_background, n_table):

    # Init vals
    nfeatures = sample.size
    sample = set(sample)
    n_fsets = offsets.shape[0]
    sizes = np.zeros(n_fsets, dtype=nb.i8)
    overlap_r = np.zeros(n_fsets, dtype=nb.f8)
    odds_r = np.zeros(n_fsets, dtype=nb.f8)
    pvals = np.zeros(n_fsets, dtype=nb.f8)
    overlaps = np.zeros((n_fsets, n_table), dtype=nb.b1)
    for i in nb.prange(n_fsets):

        # Extract feature set
        srt = starts[i]
        off = offsets[i] + srt
        fset = set(net[srt:off])

        # Build table
        overlap = np.array(list(sample.intersection(fset)), dtype=nb.i8)
        a = len(overlap)
        b = len(fset.difference(sample))
        c = len(sample.difference(fset))
        d = n_background - a - b - c

        # Store
        size = len(fset)
        sizes[i] = size
        overlaps[i][overlap] = True
        overlap_r[i] = a / size
        # Haldane-Anscombe correction
        odds_r[i] = ((a + 0.5) * (n_background - size + 0.5)) / ((size + 0.5) * (nfeatures - a + 0.5))
        pvals[i] = test1r(a, b, c, d)

    return sizes, overlap_r, odds_r, pvals, overlaps


def ora(mat, net, n_up_msk, n_bt_msk, n_background=20000, verbose=False):

    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int64)
    net = np.concatenate(net.values)

    # Define starts to subset offsets
    starts = np.zeros(offsets.shape[0], dtype=np.int64)
    starts[1:] = np.cumsum(offsets)[:-1]
    n_samples, n_features = mat.shape

    # Init empty
    pvls = np.zeros((n_samples, offsets.shape[0]), dtype=np.float64)
    ranks = np.arange(n_features, dtype=np.int64)
    for i in tqdm(range(n_samples), disable=not verbose):

        if isinstance(mat, csr_matrix):
            row = mat[i].toarray()[0]
        else:
            row = mat[i]

        # Find ranks
        sample = rankdata(row, method='ordinal').astype(np.int64)
        sample = ranks[(sample > n_up_msk) | (sample < n_bt_msk)]

        # Estimate pvals
        _, _, _, pvls[i], _ = get_pvals(sample, net, starts, offsets, n_background, n_features)

    return pvls


def extract_c(df):
    if isinstance(df, pd.DataFrame):
        c = np.unique(df.index.values.astype('U'))
    elif isinstance(df, list):
        c = np.array(df, dtype='U')
    elif isinstance(df, np.ndarray):
        c = df.astype('U')
    elif isinstance(df, pd.Index):
        c = df.values.astype('U')
    else:
        raise ValueError("df must be a dataframe with significant features as indexes, or a list/array of features.")
    return c


def get_ora_df(df, net, source='source', target='target', n_background=20000, verbose=False):
    """
    Wrapper to run ORA for results of differential analysis (long format dataframe).

    Parameters
    ----------
    df : DataFrame, list, ndarray
        Long format DataFrame with significant features to be tested as indexes, or a list/ndarray with significant features.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    n_background : int
        Integer indicating the background size. If not specified the background is the targets of ``net``.
    verbose : bool
        Whether to show progress.

    Returns
    -------
    results : DataFrame
        Results of ORA.
    """

    # Extract feature names
    df = df.copy()
    c = extract_c(df)

    # Transform net
    net = rename_net(net, source=source, target=target, weight=None)

    # Generate background
    unq_net = np.unique(net['target'].values.astype('U'))
    if n_background is None:
        n_background = unq_net.size
        # Filter
        msk = np.isin(c, unq_net)
        c = c[msk]
        if c.size == 0:
            raise ValueError("""No features in df match with the target features of net. Check that df contains enough
            features or that you have specified the correct 'target' column in net.""")
    elif not isinstance(n_background, int):
        raise ValueError("n_background must be a positive integer or None.")

    # Transform targets to indxs
    all_f = np.unique(np.hstack([unq_net, c]))
    table = {name: i for i, name in enumerate(all_f)}
    net['target'] = [table[target] for target in net['target']]
    idxs = np.array([table[name] for name in c], dtype=np.int64)
    net = net.groupby('source', observed=True)['target'].apply(lambda x: np.array(x, dtype=np.int64))
    if verbose:
        print('Running ora on df with {0} targets for {1} sources with {2} background features.'.format(len(c), len(net),
                                                                                                        n_background))
    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int64)
    terms = net.index.values.astype('U')
    net = np.concatenate(net.values)

    # Define starts to subset offsets
    starts = np.zeros(offsets.shape[0], dtype=np.int64)
    starts[1:] = np.cumsum(offsets)[:-1]
    n_features = all_f.size

    # Estimate pvals
    sizes, overlap_r, odds_r, pvls, overlap = get_pvals(idxs, net, starts, offsets, n_background, n_features)

    # Cover limit float
    msk = pvls != 0.
    min_p = np.min(pvls[msk])
    pvls[~msk] = min_p

    # Transform to df
    res = []
    for i in range(terms.size):
        if overlap_r[i] > 0:
            res.append([terms[i], sizes[i], overlap_r[i], pvls[i], odds_r[i], ';'.join(all_f[overlap[i]])])
    res = pd.DataFrame(res, columns=['Term', 'Set size', 'Overlap ratio', 'p-value', 'Odds ratio', 'Features'])
    res.insert(4, 'FDR p-value', p_adjust_fdr(res['p-value'].values))
    res.insert(6, 'Combined score', -np.log(res['p-value'].values) * res['Odds ratio'].values)

    return res


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
    n_up : int, None
        Number of top ranked features to select as observed features. By default is the top 5% of positive features.
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
    net = net.groupby('source', observed=True)['target'].apply(lambda x: np.array(x, dtype=np.int64))
    if verbose:
        print('Running ora on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), len(net)))

    # Run ORA
    pvals = ora(m, net, n_up_msk, n_bt_msk, n_background, verbose)

    # Transform to df
    pvals = pd.DataFrame(pvals, index=r, columns=net.index)
    pvals.name = 'ora_pvals'
    estimate = pd.DataFrame(-np.log10(pvals), index=r, columns=pvals.columns)
    estimate.name = 'ora_estimate'

    return return_data(mat=mat, results=(estimate, pvals))
