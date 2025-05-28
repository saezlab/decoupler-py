from typing import Tuple, Callable
from functools import partial

from anndata import AnnData
import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy.spatial as scs

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
    %(key)s

    Returns
    -------
    New AnnData object with values of the provided key in ``.obsm`` in ``X``.
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be anndata.AnnData'
    assert isinstance(key, str), 'key must be str'
    assert key in adata.obsm, f'key={key} must be in adata.obsm'
    # Generate new AnnData
    obs = adata.obs
    X_obsm = adata.obsm[key]
    if isinstance(X_obsm, pd.DataFrame):
        var = pd.DataFrame(index=X_obsm.columns)
        X = X_obsm.values
    else:
        var = pd.DataFrame(index=[f'{i}' for i in range(X_obsm.shape[1])])
        X = X_obsm
    uns = adata.uns
    obsm = adata.obsm
    odata = AnnData(
        X=X,
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
    X_key: str | None = 'X',
    inplace: bool = False,
) -> None | AnnData:
    """
    Swaps an ``AnnData.X`` for a given layer key. Generates a new object by default.

    Parameters
    ----------
    %(adata)s
    key
        ``adata.AnnData.layers`` key to place in ``adata.AnnData.X``.
    X_key
        ``adata.AnnData.layers`` key where to move and store the original ``adata.AnnData.X``.
        If None, the original ``adata.AnnData.X`` is discarded.
    inplace
        If ``False``, return a copy. Otherwise, do operation inplace and return ``None``.

    Returns
    -------
    If ``inplace=False``, new ``AnnData`` object.
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be anndata.AnnData'
    assert isinstance(key, str), 'key must be str'
    assert key in adata.layers, f'key={key} must be in adata.layers'
    assert isinstance(X_key, str) or X_key is None, 'X_key must be str or None'
    assert isinstance(inplace, bool), 'inplace must be bool'
    # Move layers
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
        func = partial(np.apply_along_axis, mode, 0)
    m = f'Using function {func.func.__name__} to aggregate observations'
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
    # Handle list columns
    ocols = obs.select_dtypes(include='object').columns
    for ocol in ocols:
        has_list = any(isinstance(x, list) for x in obs[ocol].values)
        if has_list:
            obs[ocol] = obs[ocol].str.join('_')
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
        gsize = 0
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
        gsize = groups.size
    m = f'Generating {n_rows} profiles: {smples.size} samples x {gsize} groups'
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
            m = f'sample={smp}\tcells={ncell}\tcounts={count}'
            _log(m, level='info', verbose=verbose)
            if ncell == 0 or count == 0:
                i +=1
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
                # Get cells from specific sample and group
                profile = X[((obs[sample_col] == smp) & (obs[groups_col] == grp)).values]
                if isinstance(X, sps.csr_matrix):
                    profile = profile.toarray()
                # Skip if few cells or not enough counts
                ncell = profile.shape[0]
                count = np.sum(profile)
                ncells[i] = ncell
                counts[i] = count
                m = f'group={grp}\tsample={smp}\tcells={ncell}\tcounts={count}'
                _log(m, level='info', verbose=verbose)
                # Write new meta-data
                index = smp + '_' + grp
                tmp = obs[(obs[sample_col] == smp) & (obs[groups_col] == grp)].drop_duplicates().values
                if tmp.shape[0] == 0:
                    tmp = obs[obs[sample_col] == smp].drop(columns=groups_col).drop_duplicates()
                    tmp = tmp.head(1)  # Remove extra repeated cat variables
                    tmp[groups_col] = grp
                    tmp = tmp[obs.columns].values
                new_obs.loc[index, :] = tmp
                if ncell == 0 or count == 0:
                    i +=1
                    continue
                # Get prop of non zeros
                prop = (profile.astype(bool)).mean(axis=0)
                # Pseudo-bulk
                profile = mode(profile)
                # Append
                props[i] = prop
                psbulk[i] = profile
                i += 1
    return psbulk, ncells, counts, props


@docs.dedent
def pseudobulk(
    adata: AnnData,
    sample_col: str,
    groups_col: str | None,
    layer: str | None = None,
    raw: bool = False,
    empty: bool = False,
    mode: str | Callable | dict = 'sum',
    skip_checks: bool = False,
    verbose: bool = False,
) -> AnnData:
    """
    Summarizes omic profiles across cells, grouped by sample and optionally by group categories.

    By default this function expects raw integer counts as input and sums them per sample and group (``mode='sum'``),
    but other modes are available.

    This function produces some quality control metrics to assess if is necessary to filter some samples or features.
    The number of cells that belong to each sample is stored in ``adata.obs['psbulk_n_cells']``,
    the total sum of counts per sample in ``.obs['psbulk_counts']``,
    and the proportion of cells that have a non-zero value for a given feature in ``.layers['psbulk_props']``.

    Parameters
    ----------
    %(adata)s
    sample_col
        Column of ``adata.obs`` where to extract the samples names.
    groups_col
        Column of ``adata.obs`` where to extract the groups names. Can be set to ``None`` to ignore groups.
    %(layer)s
    %(raw)s
    %(empty)s
    mode
        How to perform the pseudobulk. Available options are ``sum``, ``mean`` or ``median``. It also accepts callback
        functions, like lambda, to perform custom aggregations. Additionally, it is also possible to provide a dictionary of
        different callback functions, each one stored in a different resulting `.layer`.
        In this case, the result of the first callback function of the dictionary is stored in ``.X`` by default. To switch
        between layers check ``decoupler.swap_layer``.
    skip_checks
        Whether to skip input checks. Set to ``True`` when working with positive and negative data, or when counts are not
        integers and ``mode='sum'``.
    %(verbose)s

    Returns
    -------
    New AnnData object containing summarized pseudobulk profiles by sample and optionally by group.
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be an AnnData instance'
    assert isinstance(sample_col, str), 'sample_col must be a str'
    assert isinstance(groups_col, (str, list)) or groups_col is None, 'groups_col must be str or None'
    assert isinstance(mode, (str, dict)) or callable(mode), 'mode must be str, dict or callable'
    # Extract data
    X, obs, var = extract(adata, layer=layer, raw=raw, empty=empty, verbose=verbose)
    obs = adata.obs.loc[obs].copy()
    var = adata.var.loc[var]
    # Validate X
    _validate_X(X=X, mode=mode, skip_checks=skip_checks)
    # Format inputs
    obs, groups_col, smples, groups, n_rows = _sample_group_by(
        sample=sample_col,
        group=groups_col,
        obs=obs,
        verbose=verbose,
    )
    n_cols = var.index.size
    new_obs = pd.DataFrame(columns=obs.columns)
    if type(mode) is dict:
        psbulks = []
        for l_name in mode:
            func = mode[l_name]
            func = _validate_mode(mode=func, verbose=verbose)
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
    new_obs['psbulk_cells'] = ncells
    new_obs['psbulk_counts'] = counts
    # Make cats
    for col in new_obs.columns:
        if not pd.api.types.is_numeric_dtype(new_obs[col]):
            new_obs[col] = new_obs[col].astype('category')
    # Create new AnnData
    psbulk = AnnData(X=psbulk, obs=new_obs, var=var, layers=layers)
    # Place first element of mode dict as X
    if type(mode) is dict:
        swap_layer(psbulk, key=list(mode.keys())[0], X_key=None, inplace=True)
    return psbulk


@docs.dedent
def filter_samples(
    adata: AnnData,
    min_cells: int | float = 10,
    min_counts: int | float = 1000,
    inplace: bool = True,
) -> None | np.ndarray:
    """
    Remove pseudobulked samples with insufficient number of cells and total counts.

    Parameters
    ----------
    %(adata)s
    %(min_cells)s
    %(min_counts)s
    %(inplace)s

    Returns
    -------
    If ``inplace=False``, array of samples to be kept.
    """
    assert isinstance(adata, AnnData), 'adata must be AnnData'
    cols = {'psbulk_cells', 'psbulk_counts'}
    assert cols.issubset(adata.obs.columns), \
    'psbulk_cells and psbulk_counts must be in obs, ' \
    'run again decoupler.pp.pseudobulk'
    assert isinstance(min_cells, (int, float)) and min_cells > 0., \
    'min_cells must be numerical and bigger than 0'
    assert isinstance(min_counts, (int, float)), \
    'min_counts must be numerical'
    msk_cells = adata.obs['psbulk_cells'] >= min_cells
    msk_counts = adata.obs['psbulk_counts'] >= min_counts
    msk = msk_cells & msk_counts
    obs_names = adata.obs_names[msk].to_list()
    if inplace:
        adata._inplace_subset_obs(obs_names)
    else:
        return np.array(obs_names, dtype='U')


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
    min_prop: float = 0.7,
    inplace: bool = True,
) -> None | np.ndarray:
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
    %(inplace)s

    Returns
    -------
    If ``inplace=False``, array of genes to be kept.
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
    if inplace:
        adata._inplace_subset_var(genes)
    else:
        return genes


@docs.dedent
def filter_by_prop(
    adata: AnnData,
    min_prop: float = 0.2,
    min_smpls: int = 2,
    inplace: bool = True,
) -> None | np.ndarray:
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
    %(inplace)s

    Returns
    -------
    If ``inplace=False``, array of genes to be kept.
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
    genes = adata.var_names[msk]
    if inplace:
        adata._inplace_subset_var(genes)
    else:
        return np.array(genes, dtype='U')


@docs.dedent
def knn(
    adata: AnnData,
    key: str = 'spatial',
    bw: float = 100,
    max_nn: int = 100,
    cutoff: float = 0.1,
) -> None:
    """
    Adds K-Nearest Neighbors similarities based on spatial distances.

    Parameters
    ----------
    %(adata)s
    %(key)s
    bw
        Bandwith of kernel.
    max_nn
        Maximum number of nearest neighbors to consider.
    cutoff
        Values below this number are set to zero.
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be anndata.AnnData'
    assert key in adata.obsm, \
    f'adata.obsm must contain the spatial coordinates in adata.obsm["{key}"]'
    assert isinstance(bw, (int, float)) and bw > 0, 'bw must be numeric and > 0'
    assert isinstance(max_nn, int) and max_nn > 0, 'max_nn must be int and > 0'
    assert isinstance(cutoff, (int, float)) and cutoff > 0, \
    'cutoff must be numeric and > 0'
    # Find NN and their eucl dists
    coords = adata.obsm[key]
    nobs = coords.shape[0]
    tree = scs.KDTree(coords)
    max_nn = np.min([max_nn + 1, adata.n_obs])
    dist, idx = tree.query(coords, k=max_nn, workers=4)
    # Gaussian
    dist = np.exp(-(dist ** 2.0) / (2.0 * bw ** 2.0))
    if cutoff is not None:
        dist = dist * (dist > cutoff)
    # L1 norm
    dist = dist / np.sum(np.abs(dist), axis=1).reshape(-1, 1)
    # Build sparse matrix
    krnl = sps.csr_matrix(
        (dist.ravel(), (np.repeat(np.arange(nobs), max_nn), idx.ravel())),
        shape=(nobs, nobs)
    )
    krnl.eliminate_zeros()
    # Store
    adata.obsp[f'{key}_connectivities'] = krnl


@docs.dedent
def bin_order(
    adata: AnnData,
    order: str,
    names: str | list = None,
    label: str | None = None,
    nbins: int = 100,
) -> pd.DataFrame:
    """
    Bins features along a continuous, ordered process such as pseudotime.

    Parameters
    ----------
    %(adata)s
    %(order)s
    names
        Name of features to bin.
    %(label)s
    %(nbins)s

    Returns
    -------
    DataFrame with sources binned alng a continous ordered proess.
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be anndata.AnnData'
    assert isinstance(order, str) and order in adata.obs.columns, \
    'order must be str and in adata.obs.columns'
    assert isinstance(names, (str, list)) or names is None, \
    'names must be str, list or None'
    assert (isinstance(label, str) and label in adata.obs.columns) or label is None, \
    'label must be str and in adata.obs.columns, or None'
    assert nbins > 1 and isinstance(nbins, int), 'nbins should be higher than 1 and be an integer'
    # Get vars and ordinal variable
    if names is None:
        names = adata.var_names
    elif isinstance(names, str):
        names = [names]
    X = adata[:, names].X
    if sps.issparse(X):
        X = X.toarray()
    y = adata.obs[order].values
    # Make windows
    ymin, ymax = y.min(), y.max()
    bin_edges = np.linspace(start=ymin, stop=ymax, num=nbins + 1)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Prepare label colors
    cols = ['name', 'midpoint', 'value']
    if label is not None:
        adata.obs[label] = pd.Categorical(adata.obs[label])
        if f'{label}_colors' not in adata.uns.keys():
            from matplotlib.colors import to_hex
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('tab10')
            adata.uns[f'{label}_colors'] = [to_hex(cmap(i)) for i in adata.obs[label].sort_values().cat.codes.unique()]
        cols += ['label', 'color']
    dfs = []
    for i, name in enumerate(names):
        # Assign to windows based on order
        df = pd.DataFrame()
        df['value'] = X[:, i].ravel()
        df['name'] = name
        df['order'] = y
        df['window'] = pd.cut(df['order'], bins=bin_edges, labels=False, include_lowest=True, right=True)
        df['midpoint'] = df['window'].map(lambda x: bin_midpoints[int(x)])
        if label is not None:
            df['label'] = adata.obs[label].values
            df['color'] = [adata.uns[f'{label}_colors'][i] for i in adata.obs[label].cat.codes]
        df = df.sort_values('order')
        dfs.append(df)
    df = pd.concat(dfs)
    df = df[cols]
    df = df.rename(columns={'midpoint': 'order'}).reset_index(drop=True)
    return df
