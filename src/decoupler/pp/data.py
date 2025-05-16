from typing import Tuple

import pandas as pd
import numpy as np
from numpy.random import default_rng
from anndata import AnnData
import scipy.sparse as sps

from decoupler._docs import docs
from decoupler._log import _log
from decoupler._datatype import DataType


def _extract(
    data: DataType,
    layer: str | None = None,
    raw: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert isinstance(data, (list, pd.DataFrame, AnnData)), \
    'mat must be a list of [matrix, samples, features], pd.DataFrame (samples x features)\n\
    or an AnnData instance'
    assert layer is None or isinstance(layer, str), 'layer must be str or None'
    assert isinstance(raw, bool), 'raw must be bool'
    if isinstance(data, list):
        mat, row, col = data
        mat = np.array(mat)
        row = np.array(row, dtype='U')
        col = np.array(col, dtype='U')
    elif isinstance(data, pd.DataFrame):
        mat = data.values.astype(float)
        row = data.index.values.astype('U')
        col = data.columns.values.astype('U')
    elif isinstance(data, AnnData):
        row = data.obs_names.values.astype('U')
        if raw:
            assert data.raw, 'Received `raw=True`, but `mat.raw` is empty.'
            mat = data.raw.X.astype(float)
            col = data.raw.var_names.values.astype('U')
        else:
            col = data.var_names.values.astype('U')
            if layer:
                mat = data.layers[layer].astype(float)
            else:
                mat = data.X.astype(float)
    return mat, row, col


def _validate_mat(
    mat: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    empty: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert isinstance(empty, bool), 'empty must be bool'
    # Accept any sparse format but transform to csr
    if sps.issparse(mat) and not isinstance(mat, sps.csr_matrix):
        mat = sps.csr_matrix(mat)
    # Check for empty features
    if isinstance(mat, sps.csr_matrix):
        msk_col = mat.getnnz(axis=0) == 0
    else:
        msk_col = np.count_nonzero(mat, axis=0) == 0
    n_empty_col = np.sum(msk_col)
    if n_empty_col > 0 and empty:
        m = f'{n_empty_col} features of mat are empty, they will be removed'
        _log(m, level='warn', verbose=verbose)
        col = col[~msk_col]
        mat = mat[:, ~msk_col]
    # Check for repeated features
    assert not np.any(col[1:] == col[:-1]), \
    'mat contains repeated feature names, please make them unique'
    # Check for empty samples
    if isinstance(mat, sps.csr_matrix):
        msk_row = mat.getnnz(axis=1) == 0
    else:
        msk_row = np.count_nonzero(mat, axis=1) == 0
    n_empty_row = np.sum(msk_row)
    if n_empty_row > 0 and empty:
        m = f'{n_empty_row} observations of mat are empty, they will be removed'
        _log(m, level='warn', verbose=verbose)
        row = row[~msk_row]
        mat = mat[~msk_row]
    # Check for non finite values
    assert not np.any(~np.isfinite(mat.data)), \
    'mat contains non finite values (nan or inf), set them to 0 or remove them'
    return mat, row, col


def break_ties(
    mat: np.ndarray,
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # Randomize feature order to break ties randomly
    rng = default_rng(seed=0)
    idx = np.arange(features.size)
    idx = rng.choice(idx, features.size, replace=False)
    mat, features = mat[:, idx], features[idx]
    return mat, features


@docs.dedent
def extract(
    data: DataType,
    layer: str | None = None,
    raw: bool = False,
    empty: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts matrix, rownames and colnames from data.

    Parameters
    ----------
    %(data)s
    %(layer)s
    %(raw)s
    %(empty)s
    %(verbose)s

    Returns
    -------
    Matrix, rownames and colnames from data.
    """
    # Extract
    mat, row, col = _extract(data=data, layer=layer, raw=raw)
    m = f'Extracted omics mat with {row.size} rows (observations) and {col.size} columns (features)'
    _log(m, level='info', verbose=verbose)
    # Validate
    mat, row, col = _validate_mat(mat=mat, row=row, col=col, empty=empty, verbose=verbose)
    # Randomly sort features
    mat, col = break_ties(mat=mat, features=col)
    return mat, row, col
