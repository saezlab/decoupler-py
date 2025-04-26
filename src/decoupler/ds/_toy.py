from typing import Tuple

import numpy as np
import pandas as pd


def _fillval(
    arr: np.ndarray,
    nvar: int,
    val: float,
) -> np.ndarray:
    # Compute how many zeros to add
    n_missing = nvar - arr.size
    # Pad with zeros if needed
    if n_missing > 0:
        arr = np.pad(arr, (0, n_missing), constant_values=val)
    return arr


def toy(
    nobs: int = 30,
    nvar: int = 20,
    seed: int = 42,
    val: int | float = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a toy mat and net for testing.

    Parameters
    ----------
    nobs
        Number of samples to generate.
    seed
        Random seed to use.

    Returns
    -------
    mat and net examples
    """
    # Network model
    net = pd.DataFrame([
        ['T1', 'G01', 1], ['T1', 'G02', 1], ['T1', 'G03', 0.7],
        ['T2', 'G04', 1], ['T2', 'G06', -0.5], ['T2', 'G07', -3], ['T2', 'G08', -1],
        ['T3', 'G06', 1], ['T3', 'G07', 0.5], ['T3', 'G08', 1],
        ['T4', 'G05', 1.9], ['T4', 'G10', -1.5], ['T4', 'G11', -2], ['T4', 'G09', 3.1],
        ['T5', 'G09', 0.7], ['T5', 'G10', 1.1], ['T5', 'G11', 0.1],
    ], columns=['source', 'target', 'weight'])
    # Simulate two population of samples with different molecular values
    rng = np.random.default_rng(seed=seed)
    n = int(nobs / 2)
    res = nobs % 2
    row_a = _fillval(np.array([8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0]), nvar=nvar, val=val)
    row_b = _fillval(np.array([0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0]), nvar=nvar, val=val)
    row_a = [row_a + np.abs(rng.normal(size=nvar)) for _ in range(n)]
    row_b = [row_b + np.abs(rng.normal(size=nvar)) for _ in range(n + res)]
    mat = np.vstack([row_a, row_b])
    features = ['G{:02d}'.format(i + 1) for i in range(nvar)]
    samples = ['S{:02d}'.format(i + 1) for i in range(nobs)]
    mat = pd.DataFrame(mat, index=samples, columns=features)
    return mat, net
