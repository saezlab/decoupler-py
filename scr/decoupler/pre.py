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
    c_msk : Array of indexes to reorder `mat`'s columns.
    r_msk : Array of indexes to reorder `net`'s rows.
    """
    
    # Intersect
    inter = np.sort(list(set(r) & set(c)))
    
    # Match
    c_msk = np.searchsorted(c, inter)
    r_msk = np.searchsorted(r, inter)
    
    return c_msk, r_msk
