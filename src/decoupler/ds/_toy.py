from typing import Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

from decoupler._docs import docs
from decoupler._log import _log


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


@docs.dedent
def toy(
    nobs: int = 30,
    nvar: int = 20,
    bval: int | float = 2.,
    pstime: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[AnnData, pd.DataFrame]:
    """
    Generate a toy adata and net for testing.

    Parameters
    ----------
    nobs
        Number of samples to generate.
    nvar
        Number of features to generate.
    bvar
        Background value to set features not associated to any source.
    pstime
        Whether to add simulated pseudotime.
    %(seed)s
    %(verbose)s

    Returns
    -------
    AnnData and net examples.
    """
    # Validate
    assert isinstance(nobs, (int, float)) and nobs >= 2, \
    'nobs must be numeric and >= 2'
    assert isinstance(nvar, (int, float)) and nvar >= 12, \
    'nvar must be numeric and >= 12'
    assert isinstance(bval, (int, float)), \
    'bval must be numeric'
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
    row_a = np.array([8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0])
    row_b = np.array([0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0])
    if nvar > 12:
        m = f'toy - adding background gene expresison for extra vars bval={bval}'
        _log(m, level='info', verbose=verbose)
        row_a = _fillval(row_a, nvar=nvar, val=bval)
        row_b = _fillval(row_b, nvar=nvar, val=bval)
    m = f'toy - adding random noise to .X with seed={seed}'
    _log(m, level='info', verbose=verbose)
    row_a = [row_a + np.abs(rng.normal(size=nvar)) for _ in range(n)]
    row_b = [row_b + np.abs(rng.normal(size=nvar)) for _ in range(n + res)]
    adata = np.vstack([row_a, row_b])
    features = ['G{:02d}'.format(i + 1) for i in range(nvar)]
    samples = ['C{:02d}'.format(i + 1) for i in range(nobs)]
    adata = pd.DataFrame(adata, index=samples, columns=features)
    adata = AnnData(adata)
    adata.obs['group'] = (['A'] * len(row_a)) + (['B'] * len(row_b))
    adata.obs['group'] = adata.obs['group'].astype('category')
    adata.obs['sample'] = rng.choice(['S01', 'S02', 'S03'], size=adata.n_obs, replace=True)
    adata.obs['sample'] = adata.obs['sample'].astype('category')
    if pstime:
        m = f'toy - Adding simulated pseudotime'
        _log(m, level='info', verbose=verbose)
        pst = np.arange(adata.n_obs)
        pst = pst / pst.max()
        adata.X[:, :8] = adata.X[:, :8] + (adata.X[:, :8] * pst.reshape(-1, 1))
        adata.obs['pstime'] = pst
    m = f'toy - generated AnnData with shape={adata.shape}'
    _log(m, level='info', verbose=verbose)
    return adata, net


@docs.dedent
def toy_bench(
    shuffle_r: float = 0.25,
    seed: int = 42,
    verbose: bool = False,
    **kwargs
):
    """
    Generate a toy adata and net for testing the benchmark pipeline.

    Parameters
    ----------
    shuffle_r
        Percentage of the ground truth to randomize.
    %(seed)s
    %(verbose)s
    kwargs
        All other keyword arguments are passed to ``decoupler.ds.toy``.

    Returns
    -------
    AnnData and net examples.
    """
    # Validate
    assert isinstance(shuffle_r, (int, float)) and 0.0 <= shuffle_r <= 1.0, \
    'shuffle_r must be numeric and between 0 and 1'
    # Get toy data
    adata, net = toy(**kwargs)
    # Add grount truth
    adata.obs['source'] = [['T1', 'T2'] if g == 'A' else ['T3', 'T4'] for g in adata.obs['group']]
    adata.obs['class'] = np.tile(['CA', 'CB'], int(np.ceil(adata.obs_names.size / 2)))[:adata.obs_names.size]
    adata.obs['class'] = adata.obs['class'].astype('category')
    adata.obs['type_p'] = 1.
    # Shuffle a percentage of the samples
    idxs = np.arange(adata.obs_names.size)
    n_shuffle = int(np.ceil(idxs.size * shuffle_r))
    if n_shuffle > 0:
        rng = np.random.default_rng(seed=seed)
        m = f'Shuffling {n_shuffle} observations ({shuffle_r * 100:.2f}%).'
        _log(m, level='info', verbose=verbose)
        idxs = rng.choice(idxs, n_shuffle, replace=False)
        r_idxs = rng.choice(idxs, idxs.size, replace=False)
        adata.X[r_idxs, :] = adata.X[idxs, :]
    return adata, net
