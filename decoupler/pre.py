"""
Preprocessing functions.
Functions to preprocess the data before running any method.
"""

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from anndata import AnnData


def extract(mat, use_raw=True, verbose=False, dtype=np.float32):
    """
    Processes different input types so that they can be used downstream.

    Parameters
    ----------
    mat : list, pd.DataFrame or AnnData
        List of [matrix, samples, features], dataframe (samples x features) or an AnnData instance.
    use_raw : bool
        Use `raw` attribute of `adata` if present.
    dtype : type
        Type of float used.

    Returns
    -------
    m : csr_matrix
        Sparse matrix containing molecular readouts or statistics.
    r : ndarray
        Array of sample names.
    c : ndarray
        Array of feature names.
    """

    if type(mat) is list:
        m, r, c = mat
        m = csr_matrix(m)
        r = np.array(r)
        c = np.array(c)
    elif type(mat) is pd.DataFrame:
        m = csr_matrix(mat.values)
        r = mat.index.values
        c = mat.columns.values
    elif type(mat) is AnnData:
        if use_raw:
            if mat.raw is None:
                raise ValueError("Received `use_raw=True`, but `mat.raw` is empty.")
            m = mat.raw.X
            c = mat.raw.var.index.values
        else:
            m = csr_matrix(mat.X)
            c = mat.var.index.values
        r = mat.obs.index.values

    else:
        raise ValueError("""mat must be a list of [matrix, samples, features], dataframe (samples x features) or an AnnData
        instance.""")

    # Filter empty features (at least in 3 samples)
    if m.shape[0] <= 3:
        msk = np.sum(m != 0, axis=0).A1 != 0
        n_empty_samples = m.shape[0]
    else:
        msk = np.sum(m != 0, axis=0).A1 >= 3
        n_empty_samples = m.shape[0] - 3

    n_empty_features = np.sum(~msk)
    if n_empty_features > 0:
        if verbose:
            print("{0} features of mat are empty in {1} samples, they will be ignored.".format(n_empty_features,
                                                                                               n_empty_samples))
        m, c = m[:, msk], c[msk]

    # Check for non finite values
    if np.any(~np.isfinite(m.data)):
        raise ValueError("""mat contains non finite values (nan or inf), please set them to 0 or remove them.""")

    # Sort genes
    msk = np.argsort(c)

    return m[:, msk].astype(dtype), r.astype('U'), c[msk].astype('U')


def filt_min_n(c, net, min_n=5):
    """
    Removes sources of a `net` with less than min_n targets.

    First it filters target features in `net` that are not in `mat` and then removes sources with less than `min_n` targets.

    Parameters
    ----------
    c : ndarray
        Column names of `mat`.
    net : DataFrame
        Network in long format.
    min_n : int
        Minimum of targets per source. If less, sources are removed.

    Returns
    -------
    net : DataFrame
        Filtered net in long format.
    """

    # Find shared targets between mat and net
    msk = np.isin(net['target'].values.astype('U'), c)
    net = net.iloc[msk]

    # Count unique sources
    sources, counts = np.unique(net['source'].values.astype('U'), return_counts=True)

    # Find sources with more than min_n targets
    msk = np.isin(net['source'].values.astype('U'), sources[counts >= min_n])

    # Filter
    net = net[msk]

    if net.shape[0] == 0:
        raise ValueError("""No sources with more than min_n={0} targets. Make sure mat and net have shared target features or
        reduce the number assigned to min_n""".format(min_n))

    return net


def match(c, r, net):
    """
    Matches `mat` with a regulatory adjacency matrix.

    Parameters
    ----------
    c : ndarray
        Column names of `mat`.
    r : ndarray
        Row  names of `net`.
    net : ndarray
        Regulatory adjacency matrix.

    Returns
    -------
    regX : ndarray
        Matching regulatory adjacency matrix.
    """

    # Init empty regX
    regX = np.zeros((c.shape[0], net.shape[1]), dtype=np.float32)

    # Match genes from mat, else are 0s
    idxs = np.searchsorted(c, r)
    regX[idxs] = net

    return regX


def rename_net(net, source='source', target='target', weight='weight'):
    """
    Renames input network to match decoupler's format (source, target, weight).

    Parameters
    ----------
    net : DataFrame
        Network in long format.
    source : str
        Column name where to extract source features.
    target : str
        Column name where to extract target features.
    weight : str, None
        Column name where to extract features' weights. If no weights are available, set to None.

    Returns
    -------
    net : DataFrame
        Renamed network.
    """

    # Check if names are in columns
    msg = 'Column name "{0}" not found in net. Please specify a valid column.'
    assert source in net.columns, msg.format(source)
    assert target in net.columns, msg.format(target)
    if weight is not None:
        assert weight in net.columns, msg.format(weight) + """Alternatively, set to None if no weights are available."""
    else:
        net = net.copy()
        net['weight'] = 1.0
        weight = 'weight'

    # Rename
    net = net.rename(columns={source: 'source', target: 'target', weight: 'weight'})

    # Sort
    net = net.reindex(columns=['source', 'target', 'weight'])

    # Check if duplicated
    is_d = net.duplicated(['source', 'target']).sum()
    if is_d > 0:
        raise ValueError('net contains repeated edges, please remove them.')

    return net


def get_net_mat(net):
    """
    Transforms a given network to a regulatory adjacency matrix (targets x sources).

    Parameters
    ----------
    net : DataFrame
        Network in long format.

    Returns
    -------
    sources : ndarray
        Array of source names.
    targets : ndarray
        Array of target names.
    X : ndarray
        Array of interactions bewteen sources and targets (target x source).
    """

    # Pivot df to a wider format
    X = net.pivot(columns='source', index='target', values='weight')
    X[np.isnan(X)] = 0

    # Store node names and weights
    sources = X.columns.values
    targets = X.index.values
    X = X.values

    return sources.astype('U'), targets.astype('U'), X.astype(np.float32)
