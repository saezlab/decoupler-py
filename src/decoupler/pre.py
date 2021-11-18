"""
Preprocessing utility functions.
Functions to preprocess the data before running any method. 
"""

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from anndata import AnnData


def extract(mat, dtype=np.float32):
    """
    Processes different input types so that they can be used downstream. 
    
    Parameters
    ----------
    mat : list, pd.DataFrame or AnnData
        List of [genes, matrix], dataframe (samples x genes) or an AnnData
        instance.
    
    Returns
    -------
    m : sparse matrix
    c : array of genes
    """
    
    if type(mat) is list:
        m, c = mat
        m = csr_matrix(m)
        c = np.array(c)
    elif type(mat) is pd.DataFrame:
        m = csr_matrix(mat.values)
        c = mat.columns.values
    elif type(mat) is AnnData:
        m = csr_matrix(mat.X)
        c = mat.var.index.values
    else:
        raise ValueError("""mat must be a list of [genes, matrix], 
        dataframe (samples x genes) or an AnnData instance.""")
    
    # Sort genes
    msk = np.argsort(c)
    
    return m[:,msk].astype(dtype), c[msk]


def match(mat, c, r, net):
    """
    Match expression matrix with a regulatory adjacency matrix.
    
    Parameters
    ----------
    mat : csr_matrix
        Gene expression matrix.
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
    for i in range(c.shape[0]):
        for j in range(r.shape[0]):
            if c[i] == r[j]:
                regX[i] = net[j]
                break
    
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
    
    # Rename
    net.rename(columns={source: 'source', target: 'target', weight: 'weight'}, 
               inplace=True)
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
    
    return sources, targets, X