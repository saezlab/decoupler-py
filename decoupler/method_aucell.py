"""
Method AUCell.
Code to run the AUCell method.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from numpy.random import default_rng
from tqdm import tqdm

from .pre import extract, rename_net, filt_min_n

from anndata import AnnData
import numba as nb


@nb.njit(nb.f4[:](nb.f4[:], nb.i8[:], nb.i8[:], nb.i8[:], nb.i8, nb.i8), parallel=True, cache=True)
def nb_aucell(row, net, starts, offsets, n_up, n_fsets):

    # Rank row
    row = np.argsort(np.argsort(-row)) + 1

    # Empty acts
    es = np.zeros(n_fsets, dtype=nb.f4)

    # For each feature set
    for j in nb.prange(n_fsets):

        # Extract feature set
        srt = starts[j]
        off = offsets[j] + srt
        fset = net[srt:off]

        # Compute max AUC for fset
        x_th = np.arange(1, stop=fset.shape[0]+1, dtype=nb.i8)
        x_th = x_th[x_th < n_up]
        max_auc = np.sum(np.diff(np.append(x_th, n_up)) * x_th)

        # Compute AUC
        x = row[fset]
        x = np.sort(x[x < n_up])
        y = np.arange(x.shape[0]) + 1
        x = np.append(x, n_up)

        # Update acts matrix
        es[j] = np.sum(np.diff(x) * y) / max_auc

    return es


def aucell(mat, net, n_up, verbose):

    # Get dims
    n_samples = mat.shape[0]
    n_fsets = net.shape[0]

    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int64)
    net = np.concatenate(net.values)

    # Define starts to subset offsets
    starts = np.zeros(n_fsets, dtype=np.int64)
    starts[1:] = np.cumsum(offsets)[:-1]

    es = np.zeros((n_samples, n_fsets), dtype=np.float32)
    for i in tqdm(range(mat.shape[0]), disable=not verbose):

        if isinstance(mat, csr_matrix):
            row = mat[i].A[0]
        else:
            row = mat[i]

        # Compute AUC per row
        es[i] = nb_aucell(row, net, starts, offsets, n_up, n_fsets)

    return es


def run_aucell(mat, net, source='source', target='target', n_up=None, min_n=5, seed=42, verbose=False, use_raw=True):
    """
    AUCell.

    AUCell (Aibar et al., 2017) uses the Area Under the Curve (AUC) to calculate whether a set of targets is enriched within
    the molecular readouts of each sample. To do so, AUCell first ranks the molecular features of each sample from highest to
    lowest value, resolving ties randomly. Then, an AUC can be calculated using by default the top 5% molecular features in the
    ranking. Therefore, this metric, `aucell_estimate`, represents the proportion of abundant molecular features in the target
    set, and their relative abundance value compared to the other features within the sample.

    Aibar S. et al. (2017) Scenic: single-cell regulatory network inference and clustering. Nat. Methods, 14, 1083–1086.

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
    n_up : int
        Number of top ranked features to select as observed features. If not specified it will be equal to the 5% of the
        number of features.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.

    estimate : DataFrame
        AUCell scores. Stored in `.obsm['aucell_estimate']` if `mat` is AnnData.
    """

    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Set n_up
    if n_up is None:
        n_up = int(np.ceil(0.05*len(c)))
    else:
        n_up = int(np.ceil(n_up))
        n_up = np.min([n_up, c.size])  # Limit n_up to max features
    if not 0 < n_up:
        raise ValueError('n_up needs to be a value higher than 0.')

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
        print('Running aucell on mat with {0} samples and {1} targets for {2} sources.'.format(m.shape[0], len(c), len(net)))

    # Run AUCell
    estimate = aucell(m, net, n_up, verbose)
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = 'aucell_estimate'

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
    else:
        return estimate
