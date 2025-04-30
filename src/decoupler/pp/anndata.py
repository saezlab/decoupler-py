from typing import Tuple, Callable
from functools import partial

from anndata import AnnData
import pandas as pd
import numpy as np
import scipy.sparse as sps

from decoupler._docs import docs
from decoupler._log import _log
from decoupler.pp.data import extract


@docs.dedent
def get_obsm(
    adata: AnnData,
    key: str
) -> AnnData:
    """
    Extracts values stored in ``.obsm`` as a new AnnData object.
    This allows to reuse ``scanpy`` functions to visualise enrichment scores.

    Parameters
    ----------
    %(adata)s
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


@docs.dedent
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
    %(adata)s
    key : str
        ``adata.AnnData.layers`` key to place in ``adata.AnnData.X``.
    X_key : str, None
        ``adata.AnnData.layers`` key where to move and store the original ``adata.AnnData.X``.
        If None, the original ``adata.AnnData.X`` is discarded.
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
    verbose: bool = False,
) -> Callable:
    if mode == 'sum':
        func = partial(np.sum, axis=0)
    elif mode == 'mean':
        func = partial(np.mean, axis=0)
    elif mode == 'median':
        func = partial(np.median, axis=0)
    elif callable(mode):
        func = partial(np.apply_along_axis, func1d=mode, axis=0)
    m = f'Using function {func.__name__} to aggregate observations'
    _log(m, level='info', verbose=verbose)
    return func


def _sample_group_by(
    sample: str,
    group: str | None,
    obs: pd.DataFrame,
    verbose: bool = False,
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
    m = f'Generating {n_rows} profiles: {smples.size} samples x {groups.size} groups'
    _log(m, level='info', verbose=verbose)
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
            # Get prop of non zeros
            prop = (profile.astype(bool)).mean(axis=0)
            # Pseudo-bulk
            profile = mode(profile)
            # Append
            props[i] = prop
            psbulk[i] = profile
            i += 1
            m = f'{smp} number of cells={ncell}, number of counts={count}'
            _log(m, level='info', verbose=verbose)
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
                # Get prop of non zeros
                prop = (profile.astype(bool)).mean(axis=0)
                # Pseudo-bulk
                profile = psbulk_profile(profile, mode=mode)
                # Append
                props[i] = prop
                psbulk[i] = profile
                i += 1
                m = f'{grp} {smp} number of cells={ncell}, number of counts={count}'
                _log(m, level='info', verbose=verbose)
    return psbulk, ncells, counts, props


@docs.dedent
def pseudobulk(
    adata: AnnData,
    sample_col: str,
    groups_col: str | None,
    layer: str | None = None,
    raw: bool = False,
    empty: bool = True,
    mode: str | Callable | dict = 'sum',
    skip_checks: bool = False,
    verbose: bool = False,
) -> AnnData:
    """

    Parameters
    ----------
    %(adata)s
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be an AnnData instance'
    assert isinstance(sample_col, str), 'sample_col must be a str'
    assert isinstance(groups_col, (str, None)), 'sample_col must be str or None'
    assert isinstance(mode, (str, dict)) or callable(mode), 'mode must be str, dict or callable'
    #assert isinstance(min_cells, (int, float)) and min_cells > 0., 'min_cells must be numerical and bigger than 0'
    #assert isinstance(min_counts, (int, float)) and min_counts > 0., 'min_counts must be numerical and bigger than 0'
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
    obs, groups_col, smples, groups, n_rows = _sample_group_by(
        sample=sample_col,
        group=groups_col,
        obs=obs,
        verbose=verbose,
    )
    n_cols = adata.shape[1]
    new_obs = pd.DataFrame(columns=obs.columns)
    if type(mode) is dict:
        psbulks = []
        for l_name in mode:
            func = mode[l_name]
            func = _validate_mode(mode=mode, verbose=verbose)
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
                mode=func,
                verbose=verbose,
            )
            psbulks.append(psbulk)
        layers = {k: v for k, v in zip(mode.keys(), psbulks)}
        layers['psbulk_props'] = props
    else:
        # Compute psbulk
        func = _validate_mode(mode=mode, verbose=verbose)
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



def _min_sample_size(
    group: str | None,
    obs: pd.DataFrame,
    large_n: int,
    min_prop: float,
) -> float:
    assert isinstance(group, str) or group is None, 'group must be str or None'
    if group is None:
        min_sample_size = obs.shape[0]
    else:
        min_sample_size = obs[group].value_counts().min()
    if min_sample_size > large_n:
        min_sample_size = large_n + (min_sample_size - large_n) * min_prop
    return min_sample_size


def _cpm_cutoff(
    lib_size: np.ndarray,
    min_count: float,
) -> float:
    median_lib_size = np.median(lib_size)
    cpm_cutoff = min_count / median_lib_size * 1e6
    return cpm_cutoff


def _cpm(
    X: np.ndarray,
    lib_size: float | np.ndarray,
) -> np.ndarray:
    if isinstance(lib_size, (int, float)):
        cpm = X / lib_size * 1e6
    else:
        cpm = X / lib_size.reshape(-1, 1) * 1e6
    return cpm

def _ssize_tcount(
    X: np.ndarray,
    lib_size: float | None = None,
    min_count: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(X, sps.csr_matrix):
        X = X.toarray()
    # Compute lib_size if needed
    if lib_size is None:
        lib_size = np.sum(X, axis=1)
    # CPM cutoff
    cpm_cutoff = _cpm_cutoff(lib_size=lib_size, min_count=min_count)
    # CPM mask
    cpm = _cpm(X=X, lib_size=lib_size)
    sample_size = np.round(np.sum(cpm >= cpm_cutoff.reshape(-1, 1), axis=0))
    total_count = np.sum(X, axis=0)
    return sample_size, total_count


@docs.dedent
def filter_by_expr(
    adata: AnnData,
    group: str | None = None,
    lib_size: float | None = None,
    min_count: int = 10,
    min_total_count: int = 15,
    large_n: int = 10,
    min_prop: float = 0.7
) -> np.ndarray:
    """
    Determine which genes have sufficiently large counts to be retained in a statistical analysis.

    Adapted from the function ``filterByExpr`` of edgeR (https://rdrr.io/bioc/edgeR/man/filterByExpr.html).

    Parameters
    ----------
    %(adata)s
    group
        Name of the ``adata.obs`` column to group by. If None, it assumes that all samples belong to one group.
    lib_size
        Library size. If None, default to the sum of reads per sample.
    min_count
        Minimum count requiered per gene for at least some samples.
    min_total_count
        Minimum total count required per gene across all samples.
    large_n
        Number of samples per group that is considered to be "large".
    min_prop
        Minimum proportion of samples in the smallest group that express the gene.

    Returns
    -------
    Array of genes to be kept.
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be AnnData'
    assert isinstance(lib_size, (int, float)) or lib_size is None, \
    'lib_size must be numeric or None'
    assert isinstance(min_count, (int, float)) and min_count >= 0, \
    'min_count must be numeric and > 0'
    assert isinstance(min_total_count, (int, float)) and min_total_count >= 0, \
    'min_total_count must be numeric and > 0'
    assert isinstance(large_n, (int, float)) and large_n >= 0, \
    'large_n must be numeric and > 0'
    assert isinstance(min_prop, (int, float)) and 1 >= min_prop >= 0, \
    'min_prop must be numeric and between 0 and 1'
    # Extract inputs
    X, _, var_names = extract(adata, empty=False)
    obs = adata.obs
    # Minimum sample size cutoff
    min_sample_size = _min_sample_size(
        obs=obs,
        group=group,
        large_n=large_n,
        min_prop=min_prop,
    )
    # Compute sample size and total count
    sample_size, total_count = _ssize_tcount(
        X=X,
        lib_size=lib_size,
        min_count=min_count,
    )
    # Sample size msk
    keep_cpm = sample_size >= (min_sample_size - 1e-14)
    # Total counts msk
    keep_total_count = total_count >= (min_total_count - 1e-14)
    # Merge msks
    msk = keep_cpm & keep_total_count
    genes = var_names[msk]
    return genes

@docs.dedent
def filter_by_prop(
    adata: AnnData,
    min_prop: float = 0.2,
    min_smpls: int = 2,
) -> np.ndarray:
    """
    Determine which genes are expressed in a sufficient proportion of cells across samples.

    This function selects genes that are sufficiently expressed across cells in each sample and that this condition is
    met across a minimum number of samples.

    Parameters
    ----------
    %(adata)s
    min_prop
        Minimum proportion of cells that express a gene in a sample.
    min_smpls
        Minimum number of samples with bigger or equal proportion of cells with expression than ``min_prop``.

    Returns
    -------
    Array of genes to be kept.
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be AnnData'
    assert 'psbulk_props' in adata.layers.keys(), \
    'psbulk_props must be in adata.layers, use this function afer running decoupler.pp.pseudobulk'
    assert isinstance(min_prop, (int, float)) and 1 >= min_prop >= 0, \
    'min_prop must be numeric and between 0 and 1'
    assert isinstance(min_smpls, (int, float)) and adata.obs_names.size >= min_smpls >= 0, \
    f'min_smpls must be numeric and {adata.obs_names.size} > 0'
    # Extract props
    props = adata.layers['psbulk_props']
    if isinstance(props, pd.DataFrame):
        props = props.values
    # Compute n_smpl
    nsmpls = np.sum(props >= min_prop, axis=0)
    # Set features to 0
    msk = nsmpls >= min_smpls
    genes = adata.var_names[msk].astype('U')
    return genes
