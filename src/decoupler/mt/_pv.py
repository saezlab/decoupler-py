import numba as nb
import numpy as np


@nb.njit(cache=True)
def _fdr_bh_single_row(ps_row, m):
    """Apply Benjamini-Hochberg correction to a single row."""
    # Sort the row and get indices
    order = np.argsort(ps_row)
    ps_sorted = ps_row[order]

    # BH scale: p_(i) * m / i
    ps_bh = np.empty_like(ps_sorted, dtype=np.float64)
    for i in range(m):
        ps_bh[i] = ps_sorted[i] * (m / (i + 1))

    # Reverse cumulative min
    ps_rev = np.empty_like(ps_bh, dtype=np.float64)
    for i in range(m):
        ps_rev[i] = ps_bh[m - 1 - i]

    for j in range(1, m):
        ps_rev[j] = min(ps_rev[j], ps_rev[j - 1])

    # Reverse back
    ps_monotone = np.empty_like(ps_rev, dtype=np.float64)
    for i in range(m):
        ps_monotone[i] = ps_rev[m - 1 - i]

    # Unsort back to original order
    ps_adj = np.empty_like(ps_monotone, dtype=np.float64)
    for i in range(m):
        ps_adj[order[i]] = ps_monotone[i]

    # Clip to [0, 1]
    for i in range(m):
        ps_adj[i] = max(0.0, min(1.0, ps_adj[i]))

    return ps_adj


@nb.njit(parallel=True, cache=True)
def _fdr_bh_parallel(ps, m):
    """Apply Benjamini-Hochberg correction to all rows in parallel."""
    n_rows = ps.shape[0]
    result = np.empty_like(ps, dtype=np.float64)

    for i in nb.prange(n_rows):
        result[i] = _fdr_bh_single_row(ps[i], m)

    return result


def _fdr_bh_axis1_numba(ps):
    """Benjamini–Hochberg adjusted p-values along axis=1 (rows)."""
    ps = np.asarray(ps, dtype=np.float64)
    if ps.ndim != 2:
        raise ValueError("ps must be 2D (n_rows, n_tests) for axis=1.")
    if not np.issubdtype(ps.dtype, np.number):
        raise ValueError("`ps` must be numeric.")
    if not np.all((ps >= 0) & (ps <= 1)):
        raise ValueError("`ps` must be within [0, 1].")

    n_rows, m = ps.shape
    if m <= 1:
        return ps.copy().astype(np.float32)

    # Process each row in parallel
    result = _fdr_bh_parallel(ps, m)
    return result.astype(np.float32)
