"""
Method GSEA.
Code to run the Gene Set Enrichment Analysis (GSEA) method.
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.sparse import csr_matrix

from .pre import extract, rename_net, filt_min_n, return_data
from .utils import p_adjust_fdr

from tqdm.auto import tqdm

import numba as nb


@nb.njit(nb.types.Tuple((nb.f4, nb.i8, nb.f4[:]))(nb.f4[:], nb.i8[:], nb.b1[:], nb.f4), cache=True, error_model='numpy')
def compute_es_per_rank(row, rnks, set_msk, dec):

    # Init empty
    mx_value = 0.0
    cum_sum = 0.0
    mx_pos = 0.0
    mx_neg = 0.0
    j_pos = 0
    j_neg = 0
    es = np.zeros(rnks.size, dtype=nb.f4)

    # Compute norm
    sum_set = np.sum(np.abs(row[set_msk]))
    if sum_set == 0.:
        return 0., 0, np.zeros(rnks.size, dtype=nb.f4)

    # Compute ES
    for i in rnks:
        if set_msk[i]:
            cum_sum += np.abs(row[i]) / sum_set
            es[i] = cum_sum
        else:
            cum_sum -= dec
            es[i] = cum_sum

        # Update max scores and idx
        if cum_sum > mx_pos:
            mx_pos = cum_sum
            j_pos = i
        if cum_sum < mx_neg:
            mx_neg = cum_sum
            j_neg = i

    # Determine if pos or neg are more enriched
    if mx_pos > -mx_neg:
        mx_value = mx_pos
        j = j_pos
    else:
        mx_value = mx_neg
        j = j_neg

    return mx_value, j, es


@nb.njit(nb.f4(nb.f4[:], nb.i8), cache=True, error_model='numpy')
def std(arr, ddof):
    N = arr.shape[0]
    m = np.mean(arr)
    var = np.sum((arr - m)**2) / (N - ddof)
    sd = np.sqrt(var)
    return sd


@nb.njit(nb.types.UniTuple(nb.f4, 2)(nb.f4[:], nb.i8[:], nb.b1[:], nb.f4, nb.f4, nb.i8, nb.i8), cache=True,
         error_model='numpy')
def compute_nes_per_rank(row, rnks, set_msk, dec, es, times, seed):

    # Keep old set_msk upstream
    set_msk = set_msk.copy()

    # Compute null
    np.random.seed(seed)
    null = np.zeros(times, dtype=nb.f4)
    for i in range(times):
        np.random.shuffle(set_msk)
        null[i], _, _ = compute_es_per_rank(row, rnks, set_msk, dec)

    # Compute NES
    pos_null_msk = null >= 0.
    neg_null_msk = null < 0.

    pos_null_sum = pos_null_msk.sum()
    neg_null_sum = neg_null_msk.sum()

    if (es >= 0) and (pos_null_sum > 0):
        pval = (null[pos_null_msk] >= es).sum() / pos_null_sum
        pos_null_mean = null[pos_null_msk].mean()
        nes = es / pos_null_mean
    elif (es < 0) and (neg_null_sum > 0):
        pval = (null[neg_null_msk] <= es).sum() / neg_null_sum
        neg_null_mean = null[neg_null_msk].mean()
        nes = -es / neg_null_mean
    else:
        nes = np.inf
        pval = np.inf
    return nes, pval


@nb.njit(nb.types.Tuple((nb.f4[:], nb.f4[:], nb.f4[:], nb.i8[:], nb.f4[:], nb.f4[:], nb.b1[:, :]))
         (nb.f4[:], nb.i8[:], nb.i8[:], nb.i8[:], nb.i8, nb.i8, nb.b1), parallel=True, cache=True, error_model='numpy')
def nb_gsea(row, net, starts, offsets, times, seed, ratios):

    # Get dims
    n_features, n_fsets = row.size, offsets.shape[0]

    # Sort features
    idx = np.argsort(-row)
    row = row[idx]

    # Get ranks
    rnks = np.arange(n_features)

    # Init empty
    es = np.zeros(n_fsets, dtype=nb.f4)
    nes = np.zeros(n_fsets, dtype=nb.f4)
    pval = np.zeros(n_fsets, dtype=nb.f4)
    sizes = np.zeros(n_fsets, dtype=nb.i8)
    hits_r = np.zeros(n_fsets, dtype=nb.f4)
    rnks_r = np.zeros(n_fsets, dtype=nb.f4)
    le_msk = np.zeros((n_fsets, rnks.size), dtype=nb.b1)
    for j in nb.prange(n_fsets):

        # Extract fset
        srt = starts[j]
        nf_in_set = offsets[j]
        off = nf_in_set + srt
        fset = net[srt:off]

        # Get decending penalty
        dec = 1.0 / (n_features - nf_in_set)

        # Get msk
        set_msk = np.zeros(n_features, dtype=nb.b1)
        set_msk[fset] = True
        set_msk = set_msk[idx]

        # Compute es per feature
        estimate, idx_max, v_es = compute_es_per_rank(row, rnks, set_msk, dec)
        es[j] = estimate

        # Compute nes
        if times > 0:
            nes[j], pval[j] = compute_nes_per_rank(row, rnks, set_msk, dec, estimate, times, seed)

        # Compute Tag % and Feature %
        if ratios:
            size = set_msk.sum()
            sizes[j] = size
            if estimate >= 0:
                sub_set_msk = set_msk[:idx_max + 1]
                rnks_r[j] = (idx_max + 1) / v_es.size
                hits_r[j] = sub_set_msk.sum() / size
                set_msk[idx_max + 1:] = False
                le_msk[j] = set_msk[np.argsort(idx)]
            else:
                sub_set_msk = set_msk[idx_max:]
                rnks_r[j] = (v_es.size - idx_max) / v_es.size
                hits_r[j] = sub_set_msk.sum() / size
                set_msk[:idx_max] = False
                le_msk[j] = set_msk[np.argsort(idx)]

    return es, nes, pval, sizes, hits_r, rnks_r, le_msk


def gsea(mat, net, times=1000, seed=42, verbose=False):

    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int64)
    n_gsets = offsets.shape[0]
    net = np.concatenate(net.values)

    # Define starts to subset offsets
    starts = np.zeros(n_gsets, dtype=np.int64)
    starts[1:] = np.cumsum(offsets)[:-1]

    # Get dims
    n_samples = mat.shape[0]
    n_fsets = offsets.shape[0]

    # Init empty acts
    es = np.zeros((n_samples, n_fsets), dtype=np.float32)
    nes = np.zeros((n_samples, n_fsets), dtype=np.float32)
    pvals = np.zeros((n_samples, n_fsets), dtype=np.float32)

    for i in tqdm(range(n_samples), disable=not verbose):

        if isinstance(mat, csr_matrix):
            row = mat[i].toarray()[0]
        else:
            row = mat[i]

        # Compute GSEA per row
        es[i], nes[i], pvals[i], _, _, _, _ = nb_gsea(row, net, starts, offsets, times, seed, False)

    if times != 0:
        return es, nes, pvals
    else:
        return es, None, None


def get_gsea_df(df, stat, net, source='source', target='target', times=1000, min_n=5, seed=42, verbose=False):
    """
    Wrapper to run GSEA for results of differential analysis (long format dataframe).

    Parameters
    ----------
    df : DataFrame
        Long format DataFrame with features to be ranked. Assumes features are indexes.
    stat : str
        Name of the column containing the ranking metric.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    times : int
        How many random permutations to do.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.

    Returns
    -------
    results : DataFrame
        Results of GSEA.
    """

    # Extract
    df = df.copy()
    cols = df.columns
    if stat not in cols:
        raise ValueError('Column name "{0}" for stat not found in df. Please specify a valid column.'.format(stat))
    m = df[stat].values.astype(np.float32)
    c = df.index.values.astype('U')

    # Transform net
    net = rename_net(net, source=source, target=target, weight=None)
    net = filt_min_n(c, net, min_n=min_n)

    # Randomize feature order to break ties randomly
    rng = default_rng(seed=seed)
    idx = np.arange(m.size)
    rng.shuffle(idx)
    m, c = m[idx], c[idx]

    # Transform targets to indxs
    table = {name: i for i, name in enumerate(c)}
    net['target'] = [table[target] for target in net['target']]
    net = net.groupby('source', observed=True)['target'].apply(lambda x: np.array(x, dtype=np.int64))

    if verbose:
        print('Running gsea on df with {0} targets for {1} sources.'.format(len(c), len(net)))

    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int64)
    n_gsets = offsets.shape[0]
    terms = net.index
    net = np.concatenate(net.values)

    # Define starts to subset offsets
    starts = np.zeros(n_gsets, dtype=np.int64)
    starts[1:] = np.cumsum(offsets)[:-1]

    # Compute GSEA per row
    es, nes, pvals, sizes, hits_r, rnks_r, le_msk = nb_gsea(m, net, starts, offsets, times, seed, True)

    # Build final df
    res = []
    for i in range(es.size):
        res.append([terms[i], es[i], nes[i], pvals[i], sizes[i], hits_r[i], rnks_r[i], ';'.join(c[le_msk[i]])])
    res = pd.DataFrame(res, columns=['Term', 'ES', 'NES', 'NOM p-value', 'Set size', 'Tag %', 'Rank %', 'Leading edge'])
    res.insert(4, 'FDR p-value', p_adjust_fdr(res['NOM p-value'].values))

    return res


def run_gsea(mat, net, source='source', target='target', times=1000, batch_size=10000, min_n=5, seed=42, verbose=False,
             use_raw=True):
    """
    Gene Set Enrichment Analysis (GSEA).

    GSEA (Aravind et al., 2005) starts by transforming the input molecular readouts in `mat` to ranks for each sample. Then,
    an enrichment score `gsea_estimate` is calculated by walking down the list of features, increasing a running-sum statistic
    when a feature in the target feature set is encountered and decreasing it when it is not. The final score is the maximum
    deviation from zero encountered in the random walk. Finally, a normalized score `gsea_norm`, can be obtained by computing
    the z-score of the estimate compared to a null distribution obtained from N random permutations.

    Aravind S. et al. (2005) Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression
    profiles. PNAS. 102, 43.

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
    times : int
        How many random permutations to do.
    batch_size : int
        Deprecated argument.
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
        GSEA scores. Stored in `.obsm['gsea_estimate']` if `mat` is AnnData.
    norm : DataFrame
        Normalized GSEA scores. Stored in `.obsm['gsea_norm']` if `mat` is AnnData.
    pvals : DataFrame
        Obtained p-values. Stored in `.obsm['gsea_pvals']` if `mat` is AnnData.
    """

    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

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
        print('Running gsea on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), len(net)))

    # Run GSEA
    estimate, norm_e, pvals = gsea(m, net, times=times, seed=seed, verbose=verbose)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = 'gsea_estimate'
    if norm_e is not None:
        norm_e = pd.DataFrame(norm_e, index=r, columns=net.index)
        norm_e.name = 'gsea_norm'
        pvals = pd.DataFrame(pvals, index=r, columns=net.index)
        pvals.name = 'gsea_pvals'

    return return_data(mat=mat, results=(estimate, norm_e, pvals))
