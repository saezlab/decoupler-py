from typing import Tuple, Callable
from functools import partial

from anndata import AnnData
import pandas as pd
import numpy as np
import scipy.sparse as sps

from decoupler._log import _log
from decoupler.pp.data import extract


def get_obsm(
    adata: AnnData,
    key: str
) -> AnnData:
    """
    Extracts values stored in ``.obsm`` as a new AnnData object.
    This allows to reuse ``scanpy`` functions to visualise enrichment scores.

    Parameters
    ----------
    adata
        Annotated data matrix with enrichment scores or p-values stored in ``.obsm``.
    key
        ``.obsm`` key to extract.

    Returns
    -------
    New AnnData object with values of the provided key in ``.obsm`` in ``X``.
    """
    obs = adata.obs
    var = pd.DataFrame(index=adata.obsm[key].columns)
    uns = adata.uns
    obsm = adata.obsm
    odata = AnnData(
        X=adata.obsm[key].values,
        obs=obs,
        var=var,
        uns=uns,
        obsm=obsm
    )
    return odata


def swap_layer(
    adata: AnnData,
    key: str,
    X_key: str = 'X',
    inplace: bool = False,
) -> None | AnnData:
    """
    Swaps an ``AnnData.X`` for a given layer key. Generates a new object by default.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    key : str
        ``AnnData.layers`` key to place in ``AnnData.X``.
    X_key : str, None
        ``AnnData.layers`` key where to move and store the original ``AnnData.X``.
        If None, the original ``AnnData.X`` is discarded.
    inplace : bool
        If ``False``, return a copy. Otherwise, do operation inplace and return ``None``.

    Returns
    -------
    If ``inplace=False``, new ``AnnData`` object.
    """
    cdata = None
    if inplace:
        if X_key is not None:
            adata.layers[X_key] = adata.X
        adata.X = adata.layers[key]
    else:
        cdata = adata.copy()
        if X_key is not None:
            cdata.layers[X_key] = cdata.X
        cdata.X = cdata.layers[key]
    return cdata


def _extract_psbulk_inputs(
    adata,
    obs,
    layer,
    raw,
    verbose: bool = False,
):
    # Extract data
    X, _, _ = extract(adata, layer=layer, raw=raw, verbose=verbose)
    obs = adata.obs
    var = adata.var

    return X, obs, var


def _validate_X(
    X: np.ndarray | sps.csr_matrix,
    mode: str = 'sum',
    skip_checks: bool = False
) -> None:
    assert isinstance(skip_checks, bool), 'skip_checks must be bool'
    skip_checks = type(mode) is dict or callable(mode) or skip_checks
    if not skip_checks:
        if isinstance(X, sps.csr_matrix):
            any_neg = (X.data < 0).any()
        else:
            any_neg = (X < 0).any()
        assert not any_neg, 'Provided data contains negative values.\n \
        Check the parameters raw and layers to determine if you are selecting the correct matrix.\n \
        If negative values are to be expected, override this error by setting skip_checks=True.'
        if mode == 'sum':
            if isinstance(X, sps.csr_matrix):
                is_int = float(np.sum(X.data)).is_integer()
            else:
                is_int = float(np.sum(X)).is_integer()
            assert is_int, 'Provided data contains float (decimal) values.\n \
            Check the parameters raw and layers to determine if you are selecting the correct matrix.\n \
            If decimal values are to be expected, override this error by setting skip_checks=True'


def _validate_mode(
    mode: str | Callable = 'sum',
) -> Callable:
    if mode == 'sum':
        func = partial(np.sum, axis=0)
    elif mode == 'mean':
        func = partial(np.mean, axis=0)
    elif mode == 'median':
        func = partial(np.median, axis=0)
    elif callable(mode):
        func = partial(np.apply_along_axis, func1d=mode, axis=0)
    return func


def _sample_group_by(
    sample: str,
    group: str | None,
    obs: pd.DataFrame,
):
    # Use one column if the same
    if sample == group:
        group = None
    if group is None:
        # Filter extra columns in obs
        cols = obs.groupby(sample, observed=True).nunique(dropna=False).eq(other=1).all(axis=0)
        cols = np.hstack([sample, cols[cols].index])
        obs = obs.loc[:, cols]
        # Get unique samples
        smples = obs[sample].unique()
        groups = None
        # Get number of obs
        n_rows = len(smples)
    else:
        # Check if extra grouping is needed
        if type(group) is list:
            joined_cols = '_'.join(group)
            obs[joined_cols] = obs[group[0]].str.cat(obs[group[1:]].astype('U'), sep='_')
            group = joined_cols
        # Filter extra columns in obs
        cols = obs.groupby([sample, group], observed=True).nunique(dropna=False).eq(other=1).all(axis=0)
        cols = np.hstack([sample, group, cols[cols].index])
        obs = obs.loc[:, cols]
        # Get unique samples and groups
        smples = np.unique(obs[sample].values)
        groups = np.unique(obs[group].values)
        # Get number of obs
        n_rows = len(smples) * len(groups)
    return obs, group, smples, groups, n_rows


def _psbulk(
    n_rows: int,
    n_cols: int,
    X: np.ndarray | sps.csr_matrix,
    sample_col: str,
    groups_col: str | None,
    smples: np.ndarray,
    groups: np.ndarray,
    obs: pd.DataFrame,
    new_obs: pd.DataFrame,
    min_cells: int | float,
    min_counts: int | float,
    mode: Callable,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Init empty variables
    psbulk = np.zeros((n_rows, n_cols))
    props = np.zeros((n_rows, n_cols))
    ncells = np.zeros(n_rows)
    counts = np.zeros(n_rows)
    # Iterate for each group and sample
    i = 0
    if groups_col is None:
        for smp in smples:
            # Write new meta-data
            tmp = obs[obs[sample_col] == smp].drop_duplicates().values
            new_obs.loc[smp, :] = tmp
            # Get cells from specific sample
            profile = X[(obs[sample_col] == smp).values]
            if isinstance(X, sps.csr_matrix):
                profile = profile.toarray()
            # Skip if few cells or not enough counts
            ncell = profile.shape[0]
            count = np.sum(profile)
            ncells[i] = ncell
            counts[i] = count
            if ncell < min_cells or np.abs(count) < min_counts:
                m = f"Sample {smp} has fewer observations or counts than expected and will be removed.\n \
                Adjust min_cells or min_counts to include it."
                _log(m, level='warn', verbose=verbose)
                i += 1
                continue
            # Get prop of non zeros
            prop = (profile.astype(bool)).mean(axis=0)
            # Pseudo-bulk
            profile = mode(profile)
            # Append
            props[i] = prop
            psbulk[i] = profile
            i += 1
    else:
        for grp in groups:
            for smp in smples:
                # Write new meta-data
                index = smp + '_' + grp
                tmp = obs[(obs[sample_col] == smp) & (obs[groups_col] == grp)].drop_duplicates().values
                if tmp.shape[0] == 0:
                    tmp = np.full(tmp.shape[1], np.nan)
                new_obs.loc[index, :] = tmp
                # Get cells from specific sample and group
                profile = X[((obs[sample_col] == smp) & (obs[groups_col] == grp)).values]
                if isinstance(X, sps.csr_matrix):
                    profile = profile.toarray()
                # Skip if few cells or not enough counts
                ncell = profile.shape[0]
                count = np.sum(profile)
                ncells[i] = ncell
                counts[i] = count
                if ncell < min_cells or np.abs(count) < min_counts:
                    m = f"{grp} for sample {smp} have fewer observations than min_cells and will be removed.\n \
                    Adjust min_cells or min_counts to include it."
                    _log(m, level='warn', verbose=verbose)
                    i += 1
                    continue
                # Get prop of non zeros
                prop = (profile.astype(bool)).mean(axis=0)
                # Pseudo-bulk
                profile = psbulk_profile(profile, mode=mode)
                # Append
                props[i] = prop
                psbulk[i] = profile
                i += 1
    return psbulk, ncells, counts, props


def pseudobulk(
    adata: AnnData,
    sample_col: str,
    groups_col: str | None,
    layer: str | None = None,
    raw: bool = False,
    empty: bool = True,
    mode: str | Callable | dict = 'sum',
    min_cells: int = 10,
    min_counts: int = 1000,
    skip_checks: bool = False,
    verbose: bool = False,
) -> AnnData:
    # Validate
    assert isinstance(adata, AnnData), 'adata must be an AnnData instance'
    assert isinstance(sample_col, str), 'sample_col must be a str'
    assert isinstance(groups_col, (str, None)), 'sample_col must be str or None'
    assert isinstance(mode, (str, dict)) or callable(mode), 'mode must be str, dict or callable'
    assert isinstance(min_cells, (int, float)) and min_cells > 0., 'min_cells must be numerical and bigger than 0'
    assert isinstance(min_counts, (int, float)) and min_counts > 0., 'min_counts must be numerical and bigger than 0'
    #assert (min_prop is None) == (min_smpls is None), 'If min_prop is None, min_smpls must also be None (and vice versa)'
    #assert min_prop is None or isinstance(min_prop, (int, float)) and 0. <= min_prop <= 1., \
    #'min_props should be numerical and between 0 and 1'
    #assert min_smpls is None or isinstance(min_smpls, (int, float)) and min_smpls > 0, \
    #'min_smpls must be numerical and bigger than 0'
    # Extract data
    X, _, _ = extract(adata, layer=layer, raw=raw, empty=empty, verbose=verbose)
    obs = adata.obs.copy()
    var = adata.var
    # Validate X
    _validate_X(X=X, mode=mode, skip_checks=skip_checks)
    # Format inputs
    obs, groups_col, smples, groups, n_rows = _sample_group_by(sample=sample_col, group=groups_col, obs=obs)
    n_cols = adata.shape[1]
    new_obs = pd.DataFrame(columns=obs.columns)
    if type(mode) is dict:
        psbulks = []
        for l_name in mode:
            func = mode[l_name]
            func = _validate_mode(mode=mode)
            psbulk, ncells, counts, props = _psbulk(
                n_rows=n_rows,
                n_cols=n_cols,
                X=X,
                sample_col=sample_col,
                groups_col=groups_col,
                smples=smples,
                groups=groups,
                obs=obs,
                new_obs=new_obs,
                min_cells=min_cells,
                min_counts=min_counts,
                mode=func,
                verbose=verbose,
            )
            psbulks.append(psbulk)
        layers = {k: v for k, v in zip(mode.keys(), psbulks)}
        layers['psbulk_props'] = props
    else:
        # Compute psbulk
        func = _validate_mode(mode=mode)
        psbulk, ncells, counts, props = _psbulk(
            n_rows=n_rows,
            n_cols=n_cols,
            X=X,
            sample_col=sample_col,
            groups_col=groups_col,
            smples=smples,
            groups=groups,
            obs=obs,
            new_obs=new_obs,
            min_cells=min_cells,
            min_counts=min_counts,
            mode=func,
            verbose=verbose,
        )
        layers = {'psbulk_props': props}
    # Add QC metrics
    new_obs['psbulk_n_cells'] = ncells
    new_obs['psbulk_counts'] = counts
    # Create new AnnData
    psbulk = AnnData(X=psbulk, obs=new_obs, var=var, layers=layers)
    # Place first element of mode dict as X
    if type(mode) is dict:
        swap_layer(psbulk, layer_key=list(mode.keys())[0], X_layer_key=None, inplace=True)
    return psbulk
