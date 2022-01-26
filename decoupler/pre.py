"""
Preprocessing functions.
Functions to preprocess the data before running any method. 
"""

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from anndata import AnnData

import numba as nb


def extract(mat, use_raw=True, dtype=np.float32):
    """
    Processes different input types so that they can be used downstream. 
    
    Parameters
    ----------
    mat : list, pd.DataFrame or AnnData
        List of [matrix, samples, features], dataframe (samples x features) or an AnnData
        instance.
    use_raw : bool
        Use `raw` attribute of `adata` if present.
    dtype : type
        Type of float used.
    
    Returns
    -------
    m : sparse matrix
    r : array of samples
    c : array of features
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
        raise ValueError("""mat must be a list of [matrix, samples, features], 
        dataframe (samples x features) or an AnnData instance.""")
    
    # Sort genes
    msk = np.argsort(c)
    
    return m[:,msk].astype(dtype), r.astype('U'), c[msk].astype('U')


@nb.njit(parallel=True)
def isin(matrix, index_to_remove):
    # Faster implementation of np.isin
    # Taken from https://stackoverflow.com/questions/53046473/numpy-isin-performance-improvement

    out=np.empty(matrix.shape[0],dtype=nb.boolean)
    for i in nb.prange(matrix.shape[0]):
        if (matrix[i] == index_to_remove):
            out[i]=False
        else:
            out[i]=True

    return out


def filt_min_n(c, net, min_n=5):
    """
    Removes sources of a `net` with less than min_n targets.
    
    First it filters target genes in `net` that are not in `mat` and
    then removes sources with less than `min_n` targets. 
    
    Parameters
    ----------
    c : narray
        Column names of `mat`.
    net : pd.DataFrame
        Network in long format.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    
    Returns
    -------
    net : Filtered net.
    """
    
    # Find shared targets between mat and net
    msk = isin(net['target'].values.astype('U'), c)
    net = net.loc[msk]
    
    # Count unique sources
    sources, counts = np.unique(net['source'].values.astype('U'), return_counts=True)
    
    # Find sources with more than min_n targets
    msk = isin(net['source'].values.astype('U'), sources[counts >= min_n])
    
    return net[msk]


def match(c, r, net):
    """
    Match expression matrix with a regulatory adjacency matrix.
    
    Parameters
    ----------
    c : narray
        Column names of `mat`.
    r : narray
        Row  names of `net`.
    net : csr_matrix
        Regulatory adjacency matrix.
    
    Returns
    -------
    regX : Matching regulatory adjacency matrix.
    """
    
    # Init empty regX
    regX = np.zeros((c.shape[0], net.shape[1]))
    
    # Match genes from mat, else are 0s
    idxs = np.searchsorted(c,r)
    regX[idxs] = net
    
    return regX


def rename_net(net, source='source', target='target', weight='weight'):
    """
    Renames input network to match decoupleR's format (source, target, weight).
    
    Parameters
    ----------
    net : pd.DataFrame
        Network in long format.
    source : str
        Column name where to extract source features.
    target : str
        Column name where to extract target features.
    weight : str
        Column name where to extract features' weights. 
    
    Returns
    -------
    net : Renamed pd.DataFrame network.
    """
    
    # Check if names are in columns
    msg = 'Column name "{0}" not found in net. Please specify a valid column.'
    assert source in net.columns, msg.format(source)
    assert target in net.columns, msg.format(target)
    if weight is not None:
        assert weight in net.columns, msg.format(weight)
    else:
        import sys
        print("weight column not provided, will be set to 1s.", file=sys.stderr) 
        net = net.copy()
        net['weight'] = 1.0
        weight = 'weight'
    
    # Rename
    net = net.rename(columns={source: 'source', target: 'target', weight: 'weight'})
    # Sort
    net = net.reindex(columns=['source', 'target', 'weight'])
    
    return net


def get_net_mat(net):
    """
    Transforms a given network to an adjacency matrix (target x source).
    
    Parameters
    ----------
    net : pd.DataFrame
        Network in long format.
    
    Returns
    -------
    sources : Array of source names.
    targets : Array of target names.
    X : Matrix of interactions bewteen sources and targets (target x source).
    """

    # Pivot df to a wider format
    X = net.pivot(columns='source', index='target', values='weight')
    X[np.isnan(X)] = 0

    # Store node names and weights
    sources = X.columns.values
    targets = X.index.values
    X = X.values
    
    return sources.astype('U'), targets.astype('U'), X
