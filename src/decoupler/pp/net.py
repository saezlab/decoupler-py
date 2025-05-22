from typing import Tuple

import pandas as pd
import numpy as np
import scipy.stats as sts
from tqdm.auto import tqdm
import numba as nb

from decoupler._docs import docs
from decoupler._log import _log
from decoupler._datatype import DataType
from decoupler.pp.data import extract


def read_gmt(
    path: str,
) -> pd.DataFrame:
    """
    Read a GMT file and return the feature sets as a network.

    Parameters
    ----------
    path
        Path to GMT file containing feature sets.

    Returns
    -------
    Gene sets as ``pd.DataFrame``.
    """
    # Init empty df
    df = []
    # Read line per line
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip().split()
            # Extract gene set name
            set_name = line[0]
            # For each gene add an entry (skip link in [1])
            genes = line[2:]
            for gene in genes:
                df.append([set_name, gene])
    # Transform to df
    df = pd.DataFrame(df, columns=['source', 'target'])
    return df


def _validate_net(
    net = pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    assert isinstance(net, pd.DataFrame), 'net must be a DataFrame'
    assert {'source', 'target'}.issubset(net.columns), \
    "DataFrame must have 'source' and 'target' columns\n \
    If present but with a different names use:\n \
    net = net.rename(columns={'...' : 'source', '...': 'target'})"
    assert not net.duplicated(subset=['source', 'target']).any(), \
    "net has duplicate rows, use:\n \
    net = net.drop_duplicates(subset=['source', 'target'])"
    if 'weight' not in net.columns:
        vnet = net[['source', 'target']].copy()
        vnet['weight'] = 1.
        m = "weight not found in net.columns, adding it as:\nnet['weight'] = 1"
        _log(m, level='warn', verbose=verbose)
    else:
        vnet = net[['source', 'target', 'weight']].copy()
    vnet['source'] = vnet['source'].astype('U')
    vnet['target'] = vnet['target'].astype('U')
    vnet['weight'] = vnet['weight'].astype(float)
    return vnet


@docs.dedent
def prune(
    features: np.ndarray,
    net: pd.DataFrame,
    tmin: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Removes sources of a ``net`` with less than ``tmin`` targets shared with ``mat``.

    Parameters
    ----------
    %(features)s
    %(net)s
    %(tmin)s
    %(verbose)s

    Returns
    -------
    Filtered net in long format.
    """
    # Validate
    vnet = _validate_net(net, verbose=verbose)
    features = set(features)
    assert isinstance(tmin, (int, float)) and tmin >= 0, 'tmin must be numeric and >= 0'
    # Find shared targets between mat and net
    msk = vnet['target'].isin(features)
    vnet = vnet.loc[msk]
    # Find unique sources with tmin
    sources = vnet['source'].value_counts()
    sources = set(sources[sources >= tmin].index)
    # Filter
    msk = vnet['source'].isin(sources)
    vnet = vnet[msk]
    assert not vnet.empty, \
    f'No sources with more than tmin={tmin} targets after\n \
    filtering by shared features in mat.\n \
    Make sure mat and net have shared target features or\n \
    reduce the number assigned to tmin'
    return vnet


def _adj(
    net: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Pivot df to a wider format
    X = net.pivot(columns='source', index='target', values='weight').fillna(0)
    # Store node names and weights
    sources = X.columns.values.astype('U')
    targets = X.index.values.astype('U')
    X = X.values.astype(float)
    return sources, targets, X


def _order(
    features: np.ndarray,
    targets: np.ndarray,
    adjmat: np.ndarray,
) -> np.ndarray:
    # Init empty madjmat
    madjmat = np.zeros((len(features), adjmat.shape[1]))
    # Create an index array for rows of features corresponding to targets
    features_dict = {gene: i for i, gene in enumerate(features)}
    idxs = [features_dict[gene] for gene in targets if gene in features_dict]
    assert len(idxs) > 0, 'No overlap found between features and targets'
    # Populate madjmat using advanced indexing
    madjmat[idxs, :] = adjmat[: len(idxs), :]
    return madjmat


@docs.dedent
def adjmat(
    features: np.ndarray,
    net: pd.DataFrame,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a network in long format into a regulatory adjacency matrix (targets x sources).

    Parameters
    ----------
    %(net)s

    Returns
    -------
    Returns the source names (columns), target names (rows), and the adjacency matrix of weights.
    """
    # Extract adj mat
    sources, targets, adjm = _adj(net=net)
    # Sort adjmat to match features
    adjm = _order(features, targets, adjm)
    m = f'Network adjacency matrix has {targets.size} unique features and {sources.size} unique sources'
    _log(m, level='info', verbose=verbose)
    return sources, targets, adjm


@docs.dedent
def idxmat(
    features: np.ndarray,
    net: pd.DataFrame,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Indexes and returns feature sets as a decomposed sparse matrix.

    Parameters
    ----------
    %(features)s
    %(net)s

    Returns
    -------
    List of sources, concatenated indexes, starts and offsets.
    """
    # Transform targets to indxs
    table = {name: i for i, name in enumerate(features)}
    net['idx_target'] = [table[target] for target in net['target']]
    # Find sets
    cnct = (
        net
        .groupby('source', observed=True)
        ['idx_target']
        .apply(lambda x: np.array(x, dtype=int))
    )
    net.drop(columns=['idx_target'], inplace=True)
    sources = cnct.index.values.astype('U')
    # Flatten net and get offsets
    offsets = cnct.apply(lambda x: len(x)).values
    cnct = np.concatenate(cnct.values)
    # Define starts to subset offsets
    starts = np.zeros(offsets.shape[0], dtype=int)
    starts[1:] = np.cumsum(offsets)[:-1]
    targets = np.unique(cnct)
    m = f'Network has {targets.size} unique features and {sources.size} unique sources'
    _log(m, level='info', verbose=verbose)
    return sources, cnct, starts, offsets


@nb.njit(cache=True)
def _getset(
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    j: int
) -> np.ndarray:
    srt = starts[j]
    off = srt + offsets[j]
    fset = cnct[srt:off]
    return fset


@docs.dedent
def shuffle_net(
    net: pd.DataFrame,
    target: bool = True,
    weight: bool = False,
    seed: int = 42,
    same_seed: bool = True
) -> pd.DataFrame:
    """
    Shuffle a network to make it random.

    Shuffle a given net by targets, weight or both at the same time.

    If only targets are shuffled, targets will change but the
    distribution of weights for each set will be preserved.

    If only weights are shuffled, targets will be the same but the
    distribution of weights for each set will change.

    If targets and weights are shuffled at the same time,
    both targets and weight distribution will change for each set.

    Parameters
    ----------
    %(net)s
    target
        Whether to shuffle targets.
    weight
        Whether to shuffle weights.
    %(seed)s
    same_seed : bool
        Whether to share seed when shuffling targets and weights.

    Returns
    -------
    Shuffled network.
    """
    # Validate
    assert isinstance(net, pd.DataFrame), 'net must be pandas.DataFrame'
    assert isinstance(target, bool), 'target must be bool'
    assert isinstance(weight, bool), 'weight must be bool'
    assert target or weight, 'If target and weight are both False, nothing is shuffled'
    assert isinstance(same_seed, bool), 'same_seed must be bool'
    assert 'target' in net.columns, 'target must be in net.columns'
    assert 'weight' in net.columns, 'weight must be in net.columns'
    # Make copy of net
    rnet = net.copy()
    # Shuffle
    if target:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(rnet['target'].values)
    if weight:
        rng = np.random.default_rng(seed=seed + int(not same_seed))
        rng.shuffle(rnet['weight'].values)
    return rnet.drop_duplicates(['source', 'target'], keep='first')


@docs.dedent
def net_corr(
    net: pd.DataFrame,
    data: None | DataType = None,
    tmin: int = 5,
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Checks the correlation across the sources in a network.
    If data is also provided, target features will be prunned to
    match the ones in mat.

    Parameters
    ----------
    %(net)s
    %(data)s
    %(tmin)s
    %(verbose)s
    kwargs
        All other keyword arguments are passed to ``decoupler.pp.extract``

    Returns
    -------
    Correlation pairs dataframe.
    """
    net = _validate_net(net, verbose=verbose)
    # If mat is provided
    if data is not None:
        # Extract sparse matrix and array of genes
        kwargs.setdefault('verbose', verbose)
        _, _, c = extract(data=data, **kwargs)
    else:
        c = np.unique(net['target'].values).astype('U')
    net = prune(features=c, net=net, tmin=tmin)
    sources, targets, adj = adjmat(features=c, net=net, verbose=False)
    # Compute corr
    corr = []
    for i, s_a in enumerate(tqdm(sources[:-1], disable=not verbose)):
        idx = np.arange(i + 1)
        A = np.delete(adj, idx, axis=1)
        b = adj[:, i].reshape(-1, 1)
        r, p = sts.pearsonr(A, b)
        for j, s_b in enumerate(sources[i + 1:]):
            corr.append([s_a, s_b, r[j], p[j]])
    corr = pd.DataFrame(corr, columns=['source_a', 'source_b', 'corr', 'pval'])
    corr['padj'] = sts.false_discovery_control(corr['pval'])
    corr['abs_corr'] = corr['corr'].abs()
    corr = (
        corr
        .sort_values(['padj', 'pval', 'abs_corr'], ascending=[False, True, True])
        .reset_index(drop=True)
        .drop(columns=['abs_corr'])
    )
    return corr
