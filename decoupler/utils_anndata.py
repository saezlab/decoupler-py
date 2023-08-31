"""
Utility functions for AnnData objects.
Functions to process AnnData objects.
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse
import pandas as pd
import sys

from anndata import AnnData
from tqdm import tqdm

from .utils import melt, p_adjust_fdr
from .pre import rename_net


def get_acts(adata, obsm_key, dtype=np.float32):
    """
    Extracts activities as AnnData object.

    From an AnnData object with source activities stored in ``.obsm``, generates a new AnnData object with activities in ``X``.
    This allows to reuse many scanpy processing and visualization functions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with activities stored in ``.obsm``.
    obsm_key : str
        ``.osbm`` key to extract.
    dtype : type
        Type of float used.

    Returns
    -------
    acts : AnnData
        New AnnData object with activities in ``X``.
    """

    obs = adata.obs
    var = pd.DataFrame(index=adata.obsm[obsm_key].columns)
    uns = adata.uns
    obsm = adata.obsm

    return AnnData(np.array(adata.obsm[obsm_key]), obs=obs, var=var, uns=uns, obsm=obsm, dtype=dtype)


def swap_layer(adata, layer_key, X_layer_key='X', inplace=False):
    """
    Swaps an ``adata.X`` for a given layer.

    Swaps an AnnData ``X`` matrix with a given layer. Generates a new object by default.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer_key : str
        ``.layers`` key to place in ``.X``.
    X_layer_key : str, None
        ``.layers`` key where to move and store the original ``.X``. If None, the original ``.X`` is discarded.
    inplace : bool
        If ``False``, return a copy. Otherwise, do operation inplace and return ``None``.

    Returns
    -------
    layer : AnnData, None
        If ``inplace=False``, new AnnData object.
    """

    cdata = None
    if inplace:
        if X_layer_key is not None:
            adata.layers[X_layer_key] = adata.X
        adata.X = adata.layers[layer_key]
    else:
        cdata = adata.copy()
        if X_layer_key is not None:
            cdata.layers[X_layer_key] = cdata.X
        cdata.X = cdata.layers[layer_key]

    return cdata


def extract_psbulk_inputs(adata, obs, layer, use_raw):

    # Extract count matrix X
    if layer is not None:
        X = adata.layers[layer]
    elif type(adata) is AnnData:
        if use_raw:
            if adata.raw is None:
                raise ValueError("Received `use_raw=True`, but `mat.raw` is empty.")
            X = adata.raw.X
        else:
            X = adata.X
    else:
        X = adata.values

    # Extract meta-data
    if type(adata) is AnnData:
        obs = adata.obs
        var = adata.var
    else:
        var = pd.DataFrame(index=adata.columns)
        if obs is None:
            raise ValueError('If adata is a pd.DataFrame, obs cannot be None.')

        # Match indexes of X with obs if DataFrame
        idxs = adata.index
        try:
            obs = obs.loc[idxs]
        except KeyError:
            raise KeyError('Indices in obs do not match with mat\'s.')

    # Sort genes
    msk = np.argsort(var.index)
    X = X[:, msk]
    var = var.iloc[msk]

    if issparse(X) and not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    return X, obs, var


def check_X(X, mode='sum', skip_checks=False):
    if isinstance(X, csr_matrix):
        is_finite = np.all(np.isfinite(X.data))
    else:
        is_finite = np.all(np.isfinite(X))
    if not is_finite:
        raise ValueError('Data contains non finite values (nan or inf), please set them to 0 or remove them.')
    skip_checks = type(mode) is dict or callable(mode) or skip_checks
    if not skip_checks:
        if isinstance(X, csr_matrix):
            is_positive = np.all(X.data >= 0)
        else:
            is_positive = np.all(X >= 0)
        if not is_positive:
            raise ValueError("""Data contains negative values. Check the parameters use_raw and layers to
            determine if you are selecting the correct matrix. To override this, set skip_checks=True.
            """)
        if mode == 'sum':
            if isinstance(X, csr_matrix):
                is_integer = float(np.sum(X.data)).is_integer()
            else:
                is_integer = float(np.sum(X)).is_integer()
            if not is_integer:
                raise ValueError("""Data contains float (decimal) values. Check the parameters use_raw and layers to
                determine if you are selecting the correct data, which should be positive integer counts when mode='sum'.
                To override this, set skip_checks=True.
                """)


def format_psbulk_inputs(sample_col, groups_col, obs):
    # Use one column if the same
    if sample_col == groups_col:
        groups_col = None

    if groups_col is None:
        # Filter extra columns in obs
        cols = obs.groupby(sample_col).apply(lambda x: x.apply(lambda y: len(y.unique()) == 1)).all(0)
        obs = obs.loc[:, cols]

        # Get unique samples
        smples = np.unique(obs[sample_col].values)
        groups = None

        # Get number of samples and features
        n_rows = len(smples)
    else:
        # Check if extra grouping is needed
        if type(groups_col) is list:
            obs = obs.copy()
            joined_cols = '_'.join(groups_col)
            obs[joined_cols] = obs[groups_col[0]].str.cat(obs[groups_col[1:]], sep='_')
            groups_col = joined_cols

        # Filter extra columns in obs
        cols = obs.groupby([sample_col, groups_col]).apply(lambda x: x.apply(lambda y: len(y.unique()) == 1)).all(0)
        obs = obs.loc[:, cols]

        # Get unique samples and groups
        smples = np.unique(obs[sample_col].values)
        groups = np.unique(obs[groups_col].values)

        # Get number of samples and features
        n_rows = len(smples) * len(groups)

    return obs, groups_col, smples, groups, n_rows


def psbulk_profile(profile, mode='sum'):
    if mode == 'sum':
        profile = np.sum(profile, axis=0)
    elif mode == 'mean':
        profile = np.mean(profile, axis=0)
    elif mode == 'median':
        profile = np.median(profile, axis=0)
    elif callable(mode):
        profile = np.apply_along_axis(mode, 0, profile)
    else:
        raise ValueError("""mode={0} can be 'sum', 'mean', 'median' or a callable function.""".format(mode))
    return profile


def compute_psbulk(n_rows, n_cols, X, sample_col, groups_col, smples, groups, obs,
                   new_obs, min_cells, min_counts, mode, dtype):

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
            profile = X[obs[sample_col] == smp]
            if isinstance(X, csr_matrix):
                profile = profile.A

            # Skip if few cells or not enough counts
            ncell = profile.shape[0]
            count = np.sum(profile)
            ncells[i] = ncell
            counts[i] = count
            if ncell < min_cells or np.abs(count) < min_counts:
                i += 1
                continue

            # Get prop of non zeros
            prop = np.sum(profile != 0, axis=0) / profile.shape[0]

            # Pseudo-bulk
            profile = psbulk_profile(profile, mode=mode)

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
                profile = X[(obs[sample_col] == smp) & (obs[groups_col] == grp)]
                if isinstance(X, csr_matrix):
                    profile = profile.A

                # Skip if few cells or not enough counts
                ncell = profile.shape[0]
                count = np.sum(profile)
                ncells[i] = ncell
                counts[i] = count
                if ncell < min_cells or np.abs(count) < min_counts:
                    i += 1
                    continue

                # Get prop of non zeros
                prop = np.sum(profile != 0, axis=0) / profile.shape[0]

                # Pseudo-bulk
                profile = psbulk_profile(profile, mode=mode)

                # Append
                props[i] = prop
                psbulk[i] = profile
                i += 1

    return psbulk, ncells, counts, props


def get_pseudobulk(adata, sample_col, groups_col, obs=None, layer=None, use_raw=False, mode='sum', min_cells=10,
                   min_counts=1000, dtype=np.float32, skip_checks=False, min_prop=None, min_smpls=None):
    """
    Summarizes expression profiles across cells per sample and group.

    Generates summarized expression profiles across cells per sample (e.g. sample id) and group (e.g. cell type) based on the
    metadata found in ``.obs``. To ensure a minimum quality control, this function removes genes that are not expressed enough
    across cells (``min_prop``) or samples (``min_smpls``), and samples with not enough cells (``min_cells``) or gene counts
    (``min_counts``).

    By default this function expects raw integer counts as input and sums them per sample and group (``mode='sum'``), but other
    modes are available.

    This function produces some quality control metrics to assess if is necessary to filter some samples. The number of cells
    that belong to each sample is stored in ``.obs['psbulk_n_cells']``, the total sum of counts per sample in
    ``.obs['psbulk_counts']``, and the proportion of cells that express a given gene in ``.layers['psbulk_props'].

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    sample_col : str
        Column of `obs` where to extract the samples names.
    groups_col : str
        Column of `obs` where to extract the groups names. Can be set to ``None`` to ignore groups.
    obs : DataFrame, None
        If provided, metadata dataframe.
    layer : str
        If provided, which element of layers to use.
    use_raw : bool
        Use `raw` attribute of `adata` if present.
    mode : str
        How to perform the pseudobulk. Available options are ``sum``, ``mean`` or ``median``. It also accepts callback
        functions, like lambda, to perform custom aggregations. Additionally, it is also possible to provide a dictionary of
        different callback functions, each one stored in a different resulting `.layer`. In this case, the result of the first
        callback function of the dictionary is stored in ``.X`` by default. To switch between layers check
        ``decoupler.swap_layer``.
    min_cells : int
        Filter to remove samples by a minimum number of cells in a sample-group pair.
    min_counts : int
        Filter to remove samples by a minimum number of summed counts in a sample-group pair.
    dtype : type
        Type of float used.
    skip_checks : bool
        Whether to skip input checks. Set to ``True`` when working with positive and negative data, or when counts are not
        integers.
    min_prop : float
        Filter to remove features by a minimum proportion of cells with non-zero values. Deprecated parameter,
        check ``decoupler.filter_by_prop``.
    min_smpls : int
        Filter to remove genes by a minimum number of samples with non-zero values. Deprecated parameter,
        check ``decoupler.filter_by_prop``.

    Returns
    -------
    psbulk : AnnData
        Returns new AnnData object with unormalized pseudobulk profiles per sample and group. It also returns quality control
        metrics that start with the prefix ``psbulk_``.
    """

    min_cells, min_counts = np.clip(min_cells, 1, None), np.clip(min_counts, 1, None)

    # Extract inputs
    X, obs, var = extract_psbulk_inputs(adata, obs, layer, use_raw)

    # Test if X is correct
    check_X(X, mode=mode, skip_checks=skip_checks)

    # Format inputs
    obs, groups_col, smples, groups, n_rows = format_psbulk_inputs(sample_col, groups_col, obs)
    n_cols = adata.shape[1]
    new_obs = pd.DataFrame(columns=obs.columns)

    if type(mode) is dict:
        psbulks = []
        for l_name in mode:
            func = mode[l_name]
            if not callable(func):
                raise ValueError("""mode requieres a dictionary of layer names and callable functions. The layer {0} does not
                contain one.""".format(l_name))
            else:
                # Compute psbulk
                psbulk, ncells, counts, props = compute_psbulk(n_rows, n_cols, X, sample_col, groups_col, smples, groups, obs,
                                                               new_obs, min_cells, min_counts, func, dtype)
                psbulks.append(psbulk)
        layers = {k: v for k, v in zip(mode.keys(), psbulks)}
        layers['psbulk_props'] = props
    elif type(mode) is str or callable(mode):
        # Compute psbulk
        psbulk, ncells, counts, props = compute_psbulk(n_rows, n_cols, X, sample_col, groups_col, smples, groups, obs,
                                                       new_obs, min_cells, min_counts, mode, dtype)
        layers = {'psbulk_props': props}

    # Add QC metrics
    new_obs['psbulk_n_cells'] = ncells
    new_obs['psbulk_counts'] = counts

    # Create new AnnData
    psbulk = AnnData(psbulk, obs=new_obs, var=var, layers=layers, dtype=dtype)

    # Remove empty samples and features
    msk = psbulk.X == 0
    psbulk = psbulk[~np.all(msk, axis=1), ~np.all(msk, axis=0)].copy()

    # Place first element of mode dict as X
    if type(mode) is dict:
        swap_layer(psbulk, layer_key=list(mode.keys())[0], X_layer_key=None, inplace=True)

    # Filter by genes if not None.
    if min_prop is not None and min_smpls is not None:
        if groups_col is None:
            genes = filter_by_prop(psbulk, min_prop=min_prop, min_smpls=min_smpls)
        else:
            genes = []
            for group in groups:
                g = filter_by_prop(psbulk[psbulk.obs[groups_col] == group], min_prop=min_prop, min_smpls=min_smpls)
                genes.extend(g)
            genes = np.unique(genes)
        psbulk = psbulk[:, genes]

    return psbulk


def get_unq_dict(col, condition, reference):
    unq, counts = np.unique(col, return_counts=True)
    unq_dict = dict()
    for i in range(len(unq)):
        k = unq[i]
        v = counts[i]
        if k == condition:
            unq_dict[k] = v
        if reference == 'rest':
            unq_dict.setdefault('rest', 0)
            unq_dict[reference] += v
        elif k == reference:
            unq_dict[k] = v
    return unq_dict


def check_if_skip(grp, condition_col, condition, reference, unq_dict):
    skip = True
    if reference not in unq_dict and reference != 'rest':
        print('Skipping group "{0}" since reference "{1}" not in column "{2}".'.format(grp, reference, condition_col),
              file=sys.stderr)
    elif condition not in unq_dict:
        print('Skipping group "{0}" since condition "{1}" not in column "{2}".'.format(grp, condition, condition_col),
              file=sys.stderr)
    elif unq_dict[reference] < 2:
        print('Skipping group "{0}" since reference "{1}" has less than 2 samples.'.format(grp, reference), file=sys.stderr)
    elif unq_dict[condition] < 2:
        print('Skipping group "{0}" since condition "{1}" has less than 2 samples.'.format(grp, condition), file=sys.stderr)
    else:
        skip = False
    return skip


def get_contrast(adata, group_col, condition_col, condition, reference=None, method='t-test'):
    """
    Computes Differential Expression Analysis using scanpy's `rank_genes_groups` function between two conditions from
    pseudo-bulk profiles.

    Parameters
    ----------
    adata : AnnData
        Input pseudo-bulk AnnData object.
    group_col : str, None
        Column of `obs` where to extract the groups names, for example cell types. If None, do not group.
    condition_col : str
        Column of `obs` where to extract the condition names, for example disease status.
    condition : str
        Name of the condition to test inside condition_col.
    reference : str
        Name of the reference to use inside condition_col. If 'rest' or None, compare each group to the union of the rest of
        the group.
    method : str
        Method to use for scanpy's `rank_genes_groups` function.

    Returns
    -------
    logFCs : DataFrame
        Dataframe containing log-fold changes per gene.
    p_vals : DataFrame
         Dataframe containing p-values per gene.
    """

    try:
        from scanpy.tl import rank_genes_groups
        from scanpy.get import rank_genes_groups_df
    except Exception:
        raise ImportError('scanpy is not installed. Please install it with: pip install scanpy')

    # Process reference
    if reference is None or reference == 'rest':
        reference = 'rest'
        glst = 'all'
    else:
        glst = [condition, reference]

    if group_col is not None:
        # Find unique groups
        groups = np.unique(adata.obs[group_col].values.astype(str))
    else:
        group_col = 'tmpcol'
        grp = '{0}.vs.{1}'.format(condition, reference)
        adata.obs[group_col] = grp
        groups = [grp]

    # Condition and reference must be different
    if condition == reference:
        raise ValueError('condition and reference cannot be identical.')

    # Init empty logFC and pvals
    logFCs = pd.DataFrame(columns=adata.var.index)
    p_vals = pd.DataFrame(columns=adata.var.index)

    for grp in groups:

        # Sub-set by group
        sub_adata = adata[adata.obs[group_col] == grp].copy()
        sub_adata.obs = sub_adata.obs[[condition_col]]

        # Transform string columns to categories (removes anndata warnings)
        if sub_adata.obs[condition_col].dtype == 'object' or sub_adata.obs[condition_col].dtype == 'category':
            sub_adata.obs[condition_col] = pd.Categorical(sub_adata.obs[condition_col])

        # Run DEA if enough samples
        unq_dict = get_unq_dict(sub_adata.obs[condition_col], condition, reference)
        skip = check_if_skip(grp, condition_col, condition, reference, unq_dict)
        if skip:
            continue
        rank_genes_groups(sub_adata, groupby=condition_col, groups=glst, reference=reference, method=method)

        # Extract DEA results
        df = rank_genes_groups_df(sub_adata, group=condition)
        df[group_col] = grp
        logFC = df.pivot(columns='names', index=group_col, values='logfoldchanges')
        logFCs = pd.concat([logFCs, logFC])
        p_val = df.pivot(columns='names', index=group_col, values='pvals')
        p_vals = pd.concat([p_vals, p_val])

    # Force dtype
    logFCs = logFCs.astype(np.float32)
    p_vals = p_vals.astype(np.float32)

    # Add name
    logFCs.name = 'contrast_logFCs'
    p_vals.name = 'contrast_pvals'

    if group_col is None:
        del adata.obs[group_col]

    return logFCs, p_vals


def get_top_targets(logFCs, pvals, contrast, name=None, net=None, source='source', target='target',
                    weight='weight', sign_thr=1, lFCs_thr=0.0, fdr_corr=True):
    """
    Return significant target features for a given source and contrast. If no name or net are provided, return all significant
    features without subsetting.

    Parameters
    ----------
    logFCs : DataFrame
        Data-frame of logFCs (contrasts x features).
    pvals : DataFrame
        Data-frame of p-values (contrasts x features).
    name : str
        Name of the source.
    contrast : str
        Name of the contrast (row).
    net : DataFrame, None
        Network data-frame. If None, return without subsetting targets by it.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    sign_thr : float
        Significance threshold for adjusted p-values.
    lFCs_thr : float
        Significance threshold for logFCs.

    Returns
    -------
    df : DataFrame
        Dataframe containing the significant features.
    """

    # Check for net
    if net is not None:
        # Rename net
        net = rename_net(net, source=source, target=target, weight=weight)

        # Find targets in net that match with logFCs
        if name is None:
            raise ValueError('If net is given, name cannot be None.')
        targets = net[net['source'] == name]['target'].values
        msk = np.isin(logFCs.columns, targets)

        # Build df
        df = logFCs.loc[[contrast], msk].T.rename({contrast: 'logFCs'}, axis=1)
        df['pvals'] = pvals.loc[[contrast], msk].T
    else:
        df = logFCs.loc[[contrast]].T.rename({contrast: 'logFCs'}, axis=1)
        df['pvals'] = pvals.loc[[contrast]].T

    # Sort
    df = df.sort_values('pvals')

    if fdr_corr:
        # Compute FDR correction
        df['adj_pvals'] = p_adjust_fdr(df['pvals'].values.flatten())
        pval_col = 'adj_pvals'
    else:
        pval_col = 'pvals'

    # Filter by thresholds
    df = df[(np.abs(df['logFCs']) >= lFCs_thr) & (np.abs(df[pval_col]) <= sign_thr)]

    # Format names
    df = df.reset_index().rename({'index': 'name'}, axis=1)
    df['contrast'] = contrast

    # Order columns
    if fdr_corr:
        df = df[['contrast', 'name', 'logFCs', 'pvals', 'adj_pvals']].sort_values('pvals')
    else:
        df = df[['contrast', 'name', 'logFCs', 'pvals']].sort_values('pvals')

    return df


def format_contrast_results(logFCs, pvals):
    """
    Formats the results from get_contrast into a long format data-frame.

    logFCs : DataFrame
        Dataframe of logFCs (contrasts x features).
    pvals : DataFrame
        Dataframe of p-values (contrasts x features).

    Returns
    -------
    df : DataFrame
        DataFrame in long format.
    """

    df = melt([logFCs, pvals]).rename({'sample': 'contrast', 'source': 'name', 'score': 'logFCs'}, axis=1)
    df = df[['contrast', 'name', 'logFCs', 'pvals']].sort_values('contrast').reset_index(drop=True)
    df['adj_pvals'] = df.groupby('contrast').apply(lambda x: p_adjust_fdr(x['pvals'])).explode().values
    df = df.sort_values(['contrast', 'pvals']).reset_index(drop=True)

    return df


def get_filterbyexpr_inputs(adata, obs):
    # Extract inputs
    if type(adata) is AnnData:
        y, obs, var_names = adata.X, adata.obs, adata.var_names.values.astype('U')
    elif type(adata) is pd.DataFrame:
        y, var_names = adata.values, adata.columns.values.astype('U')
        if obs is None:
            obs = pd.DataFrame(index=adata.index)
    else:
        raise ValueError("""adata must be either an AnnData object or a df.""")
    return y, obs, var_names


def get_min_sample_size(group, obs, large_n, min_prop):
    # Minimum effect sample sample size for any of the coefficients
    if group is None:
        min_sample_size = obs.shape[0]
    elif isinstance(group, (str, list, np.ndarray, pd.Series)):
        if isinstance(group, str):
            group = obs[group].values
        elif isinstance(group, pd.Series):
            group = obs.values
        else:
            group = np.array(group)
        _, n = np.unique(group, return_counts=True)
        min_sample_size = np.min(n[n > 0])
    else:
        raise ValueError('group needs to be a column name, or a list/series/array of values.')

    if min_sample_size > large_n:
        min_sample_size = large_n + (min_sample_size - large_n) * min_prop

    return min_sample_size


def get_cpm_cutoff(lib_size, min_count):
    median_lib_size = np.median(lib_size)
    cpm_cutoff = min_count / median_lib_size * 1e6
    return cpm_cutoff


def get_cpm(y, lib_size):
    if type(lib_size) is float or type(lib_size) is int:
        return y / lib_size * 1e6
    else:
        return y / lib_size.reshape(-1, 1) * 1e6


def filter_by_expr(adata, obs=None, group=None, lib_size=None, min_count=10, min_total_count=15, large_n=10, min_prop=0.7):
    """
    Determine which genes have sufficiently large counts to be retained in a statistical analysis.

    Adapted from the function ``filterByExpr`` of edgeR (https://rdrr.io/bioc/edgeR/man/filterByExpr.html).

    Parameters
    ----------
    adata : AnnData
        AnnData obtained after running ``decoupler.get_pseudobulk``.
    obs : DataFrame, None
        If provided, metadata dataframe, only needed if ``adata`` is not an ``AnnData``.
    group : str, None
        Name of the ``.obs`` column to group by. If None, it assumes that all samples belong to one group.
    lib_size : int, float, None
        Library size. If None, default to the sum of reads per sample.
    min_count : int
        Minimum count requiered per gene for at least some samples.
    min_total_count : int
        Minimum total count required per gene across all samples.
    large_n : int
        Number of samples per group that is considered to be "large".
    min_prop : float
        Minimum proportion of samples in the smallest group that express the gene.

    Returns
    -------
    genes : ndarray
        List of genes to be kept.
    """

    # Define limits
    min_count = np.clip(min_count, 0, None)
    min_total_count = np.clip(min_total_count, 0, None)
    large_n = np.clip(large_n, 0, None)
    min_prop = np.clip(min_prop, 0, 1)

    # Extract inputs
    y, obs, var_names = get_filterbyexpr_inputs(adata, obs)

    # Compute lib_size if needed
    if lib_size is None:
        lib_size = np.sum(y, axis=1)

    # Minimum sample size cutoff
    min_sample_size = get_min_sample_size(group, obs, large_n, min_prop)
    min_sample_size -= 1e-14

    # Total count cutoff
    min_total_count -= 1e-14

    # CPM cutoff
    cpm_cutoff = get_cpm_cutoff(lib_size, min_count)

    # CPM mask
    cpm = get_cpm(y, lib_size)
    sample_size = np.sum(cpm >= cpm_cutoff, axis=0)
    keep_cpm = sample_size >= min_sample_size

    # Total counts msk
    keep_total_count = np.sum(y, axis=0) >= min_total_count

    # Merge msks
    msk = keep_cpm & keep_total_count
    genes = var_names[msk]

    return genes


def filter_by_prop(adata, min_prop=0.2, min_smpls=2):
    """
    Determine which genes are expressed in a sufficient proportion of cells across samples.

    This function selects genes that are sufficiently expressed across cells in each sample and that this condition is
    met across a minimum number of samples.

    Parameters
    ----------
    adata : AnnData
        AnnData obtained after running ``decoupler.get_pseudobulk``. It requieres ``.layer['psbulk_props']``.
    min_prop : float
        Minimum proportion of cells that express a gene in a sample.
    min_smpls : int
        Minimum number of samples with bigger or equal proportion of cells with expression than ``min_prop``.

    Returns
    -------
    genes : ndarray
        List of genes to be kept.
    """

    # Define limits
    min_prop = np.clip(min_prop, 0, 1)
    min_smpls = np.clip(min_smpls, 0, adata.shape[0])

    if isinstance(adata, AnnData):
        layer_keys = adata.layers.keys()
        if 'psbulk_props' in list(layer_keys):
            var_names = adata.var_names.values.astype('U')
            props = adata.layers['psbulk_props']
            if isinstance(props, pd.DataFrame):
                props = props.values

            # Compute n_smpl
            nsmpls = np.sum(props >= min_prop, axis=0)

            # Set features to 0
            msk = nsmpls >= min_smpls
            genes = var_names[msk]
            return genes
    raise ValueError("""adata must be an AnnData object that contains the layer 'psbulk_props'. Please check the function
                     decoupler.get_pseudobulk.""")


def rank_sources_groups(adata, groupby, reference='rest', method='t-test_overestim_var'):
    """
    Rank sources for characterizing groups.

    Parameters
    ----------
    adata : AnnData
        AnnData obtained after running ``decoupler.get_acts``.
    groupby: str
        The key of the observations grouping to consider.
    reference: str, list
        Reference group or list of reference groups to use as reference.
    method: str
        Statistical method to use for computing differences between groups. Avaliable methods
        include: ``{'wilcoxon', 't-test', 't-test_overestim_var'}``.

    Returns
    -------
    results: DataFrame with changes in source activity score between groups.
    """

    from scipy.stats import ranksums, ttest_ind_from_stats

    # Get tf names
    features = adata.var.index.values

    # Generate mask for group samples
    groups = np.unique(adata.obs[groupby].values)
    results = []
    for group in groups:

        # Extract group mask
        g_msk = (adata.obs[groupby] == group).values

        # Generate mask for reference samples
        if reference == 'rest':
            ref_msk = ~g_msk
            ref = reference
        elif isinstance(reference, str):
            ref_msk = (adata.obs[groupby] == reference).values
            ref = reference
        else:
            cond_lst = np.array([(adata.obs[groupby] == r).values for r in reference])
            ref_msk = np.sum(cond_lst, axis=0).astype(bool)
            ref = ', '.join(reference)

        assert np.sum(ref_msk) > 0, 'No reference samples found for {0}'.format(reference)

        # Skip if same than ref
        if group == ref:
            continue

        # Test differences
        result = []
        for i in np.arange(len(features)):
            v_group = adata.X[g_msk, i]
            v_rest = adata.X[ref_msk, i]
            assert np.all(np.isfinite(v_group)) and np.all(np.isfinite(v_rest)), \
                "adata contains not finite values, please remove them."
            if method == 'wilcoxon':
                stat, pval = ranksums(v_group, v_rest)
            elif method == 't-test':
                stat, pval = ttest_ind_from_stats(
                    mean1=np.mean(v_group),
                    std1=np.std(v_group, ddof=1),
                    nobs1=v_group.size,
                    mean2=np.mean(v_rest),
                    std2=np.std(v_rest, ddof=1),
                    nobs2=v_rest.size,
                    equal_var=False,  # Welch's
                )
            elif method == 't-test_overestim_var':
                stat, pval = ttest_ind_from_stats(
                    mean1=np.mean(v_group),
                    std1=np.std(v_group, ddof=1),
                    nobs1=v_group.size,
                    mean2=np.mean(v_rest),
                    std2=np.std(v_rest, ddof=1),
                    nobs2=v_group.size,
                    equal_var=False,  # Welch's
                )
            else:
                raise ValueError("Method must be one of {'wilcoxon', 't-test', 't-test_overestim_var'}.")
            mc = np.mean(v_group) - np.mean(v_rest)
            result.append([group, ref, features[i], stat, mc, pval])

        # Tranform to df
        result = pd.DataFrame(
            result,
            columns=['group', 'reference', 'names', 'statistic', 'meanchange', 'pvals']
        )

        # Correct pvalues by FDR
        result.loc[np.isnan(result['pvals']), 'pvals'] = 1
        result['pvals_adj'] = p_adjust_fdr(result['pvals'].values)

        # Sort and save
        result = result.sort_values('statistic', ascending=False)
        results.append(result)

    # Merge
    results = pd.concat(results)

    return results.reset_index(drop=True)


def _check_anova_inputs(data, obs_keys=None, obsm_key=None, use_X=False, layer=None):

    # check that data is not None
    assert data is not None, 'data cannot be None'

    # check whether data is a list, of size 2
    if isinstance(data, list):
        assert len(data) == 2, 'data must be a list of size 2'
        # both elements in data must be pd.Dataframe instances
        assert isinstance(data[0], pd.DataFrame), 'data[0] must be a pd.DataFrame instance'
        assert isinstance(data[1], pd.DataFrame), 'data[1] must be a pd.DataFrame instance'
        # both data[0] and data[1] must have the same index
        assert data[0].index.equals(data[1].index), 'data[0] and data[1] must have the same index'

        dependent_variables = data[0].columns
        explanatory_variables = data[1].columns
        scores = data[0]
        obs = data[1]

    elif hasattr(data, 'obs'):
        assert isinstance(data.obs, pd.DataFrame), 'data.obs must be a pd.DataFrame instance'
        obs = data.obs
        if obsm_key is None and not use_X and layer is None:
            raise ValueError('When providing an AnnData or MuData object, either obsm_key, use_X or layer must be specified')
        elif (obsm_key is not None and use_X) or (obsm_key is not None and layer is not None) or (use_X and layer is not None):
            raise ValueError('When providing an AnnData or MuData object, only one of obsm_key, \
                             use_X or layer must be specified')
        elif obsm_key is not None:
            assert hasattr(data, 'obsm'), 'data must have an .obsm attribute'
            assert obsm_key in data.obsm.keys(), 'obsm_key must be a key in data.obsm'
            column_name = obsm_key.replace('X_', '').replace('pca', 'PC').replace('mofa', 'Factor').replace('umap', 'UMAP')
            scores = pd.DataFrame(data.obsm[obsm_key], index=data.obs.index,
                                  columns=['{0}{1}'.format(column_name, 1 + x) for x in range(data.obsm[obsm_key].shape[1])])
        elif use_X:
            assert hasattr(data, 'X'), 'data must have a .X attribute'
            scores = pd.DataFrame(data.X, index=data.obs.index, columns=data.var.index)
        elif layer is not None:
            assert hasattr(data, 'layers'), 'data must have a .layers attribute'
            assert layer in data.layers.keys(), 'layer must be a key in data.layers'
            scores = pd.DataFrame(data.layers[layer], index=data.obs.index, columns=data.var.index)

        assert scores.index.equals(obs.index), 'scores and obs must have the same index'
        dependent_variables = scores.columns
        explanatory_variables = obs.columns
    else:
        raise TypeError('data must be a list of size 2 or an AnnData or MuData object with .obs dataframe')

    # select only the columns in obs_keys
    if obs_keys is None:
        obs_keys = explanatory_variables
    assert all([var in explanatory_variables for var in obs_keys]), 'obs_keys must be a subset of the obs columns'
    obs = obs.filter(obs_keys, axis=1)
    explanatory_variables = obs_keys

    # check that there are no . in the dependent and explanatory variables
    assert all(['.' not in var for var in dependent_variables]), 'column names of dependent variables cannot contain .'
    assert all(['.' not in var for var in explanatory_variables]), 'column names of explanatory variables \
    (obs) cannot contain .'
    # check there is no overlap of column names between dependent and explanatory variables
    assert len([value for value in dependent_variables if value in explanatory_variables]) == 0, 'dependent and explanatory \
    variables cannot have the same column names'

    # merge scores and obs
    scores = pd.merge(scores, obs, left_index=True, right_index=True)

    return scores, list(dependent_variables), list(explanatory_variables)


def check_if_statsmodels():
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.multitest import multipletests
        return sm, ols, multipletests
    except Exception:
        raise ImportError('statsmodels is not installed. Please install it.')


def format_assoc_results(data, stats_df, inplace, obsm_key, uns_key, use_X, layer):
    if isinstance(data, list):
        return stats_df
    elif inplace:
        # which of obsm_key, use_X, or layer is not None?
        if obsm_key is not None:
            key = obsm_key.replace('X_', '') + '_anova' if uns_key is None else uns_key
        elif use_X:
            key = 'X_anova' if uns_key is None else uns_key
        elif layer is not None:
            key = layer + '_anova' if uns_key is None else uns_key
        if not hasattr(data, 'uns'):
            data.uns = {key: stats_df}
        else:
            data.uns[key] = stats_df
    else:
        return stats_df


def get_metadata_associations(data, obs_keys=None, obsm_key=None, use_X=False, layer=None, uns_key=None, inplace=False,
                              alpha=0.05, method='fdr_bh'):
    """
    Associate the data to sample metadata using ANOVA. The data can be any kind of embedding stored in a layer,
    obsm or X matrix.

    Requires statsmodels to be installed.

    Parameters
    ----------
    data : list, AnnData or MuData
        The input data for ANOVA testing. It can be either a list of two pandas DataFrames [data, obs],
        an AnnData or MuData object.
    obs_keys: list, optional
        Column names of obs (sample metadata) which should be tested. If not provided, all columns in obs will be used.
    obsm_key : str, optional
        A key specifying where in obsm the data is located when providing an AnnData/MuData object. Either
        ``obsm_key``, ``use_X``, or ``layer`` must be specified.
    use_X : bool, optional
        A boolean flag indicating whether to use the data in ``.X`` from the AnnData/MuData object when providing
        the ``data``. Either ``obsm_key``, ``use_X``, or ``layer`` must be specified.
    layer : str, optional
        Which layer to use when providing an AnnData/MuData object. Either ``obsm_key``,
        ``use_X``, or ``layer`` must be specified.
    uns_key : str, optional
        Where results will be stored the AnnData/MuData object.
    inplace : bool, optional
        Whether to store the results in the AnnData or MuData object. If ``False``, the function returns a pandas
        DataFrame with the results.
    alpha : float, optional
        The significance level for multiple testing correction (default is 0.05).
    method : (str, optional):
        The method used for multiple testing correction. It can be one of the methods supported by
        ``statsmodels.stats.multitest.multipletests`` (default is ``fdr_bh``, i.e., Benjamini-Hochberg method).

    Returns
    -------
    results: pd.DataFrame
        DataFrame with ANOVA results. If ``data`` is an AnnData or MuData object and ``inplace`` is ``True``,
        the results are stored in ``data.uns[uns_key]``.
    """

    sm, ols, multipletests = check_if_statsmodels()

    scores, dependent_variables, explanatory_variables = _check_anova_inputs(
        data,
        obs_keys=obs_keys,
        obsm_key=obsm_key,
        use_X=use_X,
        layer=layer
    )

    # for each dependent variable, test the association with the other columns in scores using an ANOVA
    # collect p-values and eta_sq in a dataframe
    for dependent in tqdm(dependent_variables):
        stats = []
        for explainer in explanatory_variables:
            # check the data type of the column
            if scores[explainer].dtype == 'object' or scores[explainer].dtype == 'category':
                # if it is a string, the explanatory variable is a categorical variable
                var_name = 'C({0})'.format(explainer)
            # if it is a float or integer, the explanatory variable is a continuous variable
            elif scores[explainer].dtype == 'float' or scores[explainer].dtype == 'int':
                var_name = '{0}'.format(explainer)
            # else raise an error
            else:
                raise ValueError('Column {0} has an unknown data type. Make sure it is either \
                categorical/object or float/int'.format(explainer))

            # create the formula
            formula = '{0} ~ {1}'.format(dependent, var_name)

            # fit the model
            mod = ols(formula, data=scores.filter([dependent, explainer], axis=1)).fit()
            # get the ANOVA table
            try:
                aov_table = sm.stats.anova_lm(mod, typ=2)
            except ValueError:
                print('WARNING: could not compute ANOVA for {0} and {1}'.format(dependent, explainer))
                row = [explainer, np.nan, np.nan]
            else:
                # compute eta squared from anova table
                eta_sq = aov_table.loc[var_name, 'sum_sq'] / aov_table.sum(axis=0)['sum_sq']
                # eta squared = SS from explanatory variable / SStotal (i.e. sum of all SS from variables plus residual SS)
                row = [explainer, aov_table['PR(>F)'][0], eta_sq]
            stats.append(row)

        stats = pd.DataFrame(stats, columns=['variable', 'pval', 'eta_sq']).set_index('variable')
        stats['factor'] = dependent
        if dependent_variables.index(dependent) == 0:
            stats_df = stats.copy()
        else:
            stats_df = pd.concat([stats_df, stats], axis=0, join='outer', sort=False)

    stats_df = stats_df.reset_index()
    stats_df['p_adj'] = np.stack(
        stats_df
        .groupby('factor')
        .apply(lambda df: multipletests(df['pval'], alpha=alpha, method=method, returnsorted=False)[1])
        .values
    ).flatten()

    return format_assoc_results(data, stats_df, inplace, obsm_key, uns_key, use_X, layer)
