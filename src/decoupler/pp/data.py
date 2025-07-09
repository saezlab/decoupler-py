import numpy as np
import pandas as pd
import scipy.sparse as sps
from anndata import AnnData
from numpy.random import default_rng
from tqdm.auto import tqdm

from decoupler._datatype import DataType
from decoupler._docs import docs
from decoupler._log import _log


def _extract(
    data: DataType,
    layer: str | None = None,
    raw: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert isinstance(data, list | pd.DataFrame | AnnData), (
        "mat must be a list of [matrix, samples, features], pd.DataFrame (samples x features)\n\
    or an AnnData instance"
    )
    assert layer is None or isinstance(layer, str), "layer must be str or None"
    assert isinstance(raw, bool), "raw must be bool"
    if isinstance(data, list):
        mat, row, col = data
        mat = np.array(mat)
        row = np.array(row, dtype="U")
        col = np.array(col, dtype="U")
    elif isinstance(data, pd.DataFrame):
        mat = data.values.astype(float)
        row = data.index.values.astype("U")
        col = data.columns.values.astype("U")
    elif isinstance(data, AnnData):
        row = data.obs_names.values.astype("U")
        if raw:
            assert data.raw, "Received `raw=True`, but `mat.raw` is empty."
            mat = data.raw.X.astype(float)
            col = data.raw.var_names.values.astype("U")
        else:
            col = data.var_names.values.astype("U")
            if layer:
                mat = data.layers[layer]
            else:
                mat = data.X
            if not data.isbacked:
                mat = mat.astype(float)
    return mat, row, col


def _validate_mat(
    mat: np.ndarray, row: np.ndarray, col: np.ndarray, empty: bool = True, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert isinstance(empty, bool), "empty must be bool"
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
        m = f"{n_empty_col} features of mat are empty, they will be removed"
        _log(m, level="warn", verbose=verbose)
        col = col[~msk_col]
        mat = mat[:, ~msk_col]
    # Check for repeated features
    assert not np.any(col[1:] == col[:-1]), "mat contains repeated feature names, please make them unique"
    # Check for empty samples
    if isinstance(mat, sps.csr_matrix):
        msk_row = mat.getnnz(axis=1) == 0
    else:
        msk_row = np.count_nonzero(mat, axis=1) == 0
    n_empty_row = np.sum(msk_row)
    if n_empty_row > 0 and empty:
        m = f"{n_empty_row} observations of mat are empty, they will be removed"
        _log(m, level="warn", verbose=verbose)
        row = row[~msk_row]
        mat = mat[~msk_row]
    # Check for non finite values
    assert not np.any(~np.isfinite(mat.data)), (
        "mat contains non finite values (nan or inf), set them to 0 or remove them"
    )
    return mat, row, col


def _validate_backed(
    mat,
    row: np.ndarray,
    col: np.ndarray,
    empty: bool = True,
    verbose: bool = False,
    bsize: int = 250_000,
) -> np.ndarray:
    nbatch = int(np.ceil(row.size / bsize))
    msk_col = np.zeros((nbatch, mat.shape[1]), dtype=bool)
    for i in tqdm(range(nbatch), disable=not verbose):
        srt, end = i * bsize, i * bsize + bsize
        bmat = mat[srt:end]
        if sps.issparse(bmat):
            msk_col[i] = bmat.getnnz(axis=0) == 0
        else:
            msk_col[i] = np.count_nonzero(bmat, axis=0) == 0
        has_nonfin = np.any(~np.isfinite(bmat.data))
        assert not has_nonfin, "mat contains non finite values (nan or inf), set them to 0 or remove them"
    msk_col = np.logical_and.reduce(msk_col, axis=0)
    n_empty_col = np.sum(msk_col)
    if n_empty_col > 0 and empty:
        m = f"{n_empty_col} features of mat are empty, they will be removed"
        _log(m, level="warn", verbose=verbose)
    else:
        msk_col[:] = False
    return msk_col


def _break_ties(
    mat: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    bsize: int = 250_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[tuple, np.ndarray, np.ndarray]:
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

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        adata, net = dc.ds.toy()
        X, obs_names, var_names = dc.pp.extract(adata)
    """
    # Extract
    mat, row, col = _extract(data=data, layer=layer, raw=raw)
    m = f"Extracted omics mat with {row.size} rows (observations) and {col.size} columns (features)"
    _log(m, level="info", verbose=verbose)
    # Validate
    isbacked = hasattr(data, "isbacked") and data.isbacked
    if not isbacked:
        mat, row, col = _validate_mat(mat=mat, row=row, col=col, empty=empty, verbose=verbose)
        # Randomly sort features
        mat, col = _break_ties(mat=mat, features=col)
    else:
        msk_col = _validate_backed(mat=mat, row=row, col=col, empty=empty, verbose=verbose, bsize=bsize)
        msk_col = ~msk_col
        mat = (mat, msk_col)
        col = col[msk_col]
    return mat, row, col
