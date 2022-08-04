"""
Utility functions for AnnData objects.
Functions to process AnnData objects.
"""

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import sys

from anndata import AnnData

from .utils import melt, p_adjust_fdr
from .pre import rename_net


def get_acts(adata, obsm_key):
    """
    Extracts activities as AnnData object.

    From an AnnData object with source activities stored in `.obsm`, generates a new AnnData object with activities in `X`.
    This allows to reuse many scanpy processing and visualization functions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with activities stored in .obsm.
    obsm_key
        `.osbm` key to extract.

    Returns
    -------
    acts : AnnData
        New AnnData object with activities in X.
    """

    obs = adata.obs
    var = pd.DataFrame(index=adata.obsm[obsm_key].columns)
    uns = adata.uns
    obsm = adata.obsm

    return AnnData(np.array(adata.obsm[obsm_key]), obs=obs, var=var, uns=uns, obsm=obsm)


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

    if type(X) is not csr_matrix:
        X = csr_matrix(X)

    return X, obs, var


def check_for_raw_counts(X):
    is_finite = np.all(np.isfinite(X.data))
    if not is_finite:
        raise ValueError('Data contains non finite values (nan or inf), please set them to 0 or remove them.')
    is_positive = np.all(X.data > 0)
    if not is_positive:
        raise ValueError("""Data contains negative values. Check the parameters use_raw and layers to
        determine if you are selecting the correct data, which should be positive integer counts.""")
    is_integer = float(np.sum(X.data)).is_integer()
    if not is_integer:
        raise ValueError("""Data contains float (decimal) values. Check the parameters use_raw and layers to
        determine if you are selecting the correct data, which should be positive integer counts.""")


def format_psbulk_inputs(sample_col, groups_col, obs):
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
        # Filter extra columns in obs
        cols = obs.groupby([sample_col, groups_col]).apply(lambda x: x.apply(lambda y: len(y.unique()) == 1)).all(0)
        obs = obs.loc[:, cols]

        # Get unique samples and groups
        smples = np.unique(obs[sample_col].values)
        groups = np.unique(obs[groups_col].values)

        # Get number of samples and features
        n_rows = len(smples) * len(groups)

    return obs, smples, groups, n_rows


def compute_psbulk(psbulk, props, X, sample_col, groups_col, smples, groups, obs, new_obs, min_cells, min_counts, min_prop):

    # Iterate for each group and sample
    i = 0
    if groups_col is None:
        for smp in smples:
            # Write new meta-data
            tmp = obs[obs[sample_col] == smp].drop_duplicates().values
            new_obs.loc[smp, :] = tmp

            # Get cells from specific sample
            profile = X[obs[sample_col] == smp]

            # Skip if few cells or not enough counts
            if profile.shape[0] <= min_cells or np.sum(profile) <= min_counts:
                i += 1
                continue

            # Get prop of non zeros
            prop = np.sum(profile != 0, axis=0) / profile.shape[0]

            # Pseudo-bulk
            profile = np.sum(profile, axis=0)

            # Append
            props[i] = prop > min_prop
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

                # Skip if few cells or not enough counts
                if profile.shape[0] <= min_cells or np.sum(profile) <= min_counts:
                    i += 1
                    continue

                # Get prop of non zeros
                prop = np.sum(profile != 0, axis=0) / profile.shape[0]

                # Pseudo-bulk
                profile = np.sum(profile, axis=0)

                # Append
                props[i] = prop > min_prop
                psbulk[i] = profile
                i += 1


def get_pseudobulk(adata, sample_col, groups_col, obs=None, layer=None, use_raw=False, min_prop=0.2, min_cells=10,
                   min_counts=1000, min_smpls=2):
    """
    Generates an unormalized pseudo-bulk profile per sample and group.

    Sums the counts of the cells belonging the the same sample and group. Genes that are not expressed in at least a proportion
    of cells (`min_prop`) and a number of samples (`min_smpls`) are ignored. This is done to remove noisy genes.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    sample_col : str
        Column of `obs` where to extract the samples names.
    groups_col : str
        Column of `obs` where to extract the groups names.
    obs : DataFrame, None
        If provided, meta-data dataframe.
    layer : str
        If provided, which element of layers to use.
    use_raw : bool
        Use `raw` attribute of `adata` if present.
    min_prop : float
        Minimum proportion of cells with non-zero values.
    min_cells : int
        Minimum number of cells per sample.
    min_counts : int
        Minimum number of counts per sample.
    min_smpls : int
        Minimum number of samples per feature.

    Returns
    -------
    psbulk : AnnData
        Returns new AnnData object with unormalized pseudobulk profiles per sample and group.
    """

    # Use one column if the same
    if sample_col == groups_col:
        groups_col = None

    # Extract inputs
    X, obs, var = extract_psbulk_inputs(adata, obs, layer, use_raw)

    # Test if raw counts are present
    check_for_raw_counts(X)

    # Format inputs
    obs, smples, groups, n_rows = format_psbulk_inputs(sample_col, groups_col, obs)
    n_cols = adata.shape[1]
    new_obs = pd.DataFrame(columns=obs.columns)
    new_var = pd.DataFrame(index=var.index)

    # Init empty variables
    psbulk = np.zeros((n_rows, n_cols))
    props = np.full((n_rows, n_cols), False)

    # Compute psbulk
    compute_psbulk(psbulk, props, X, sample_col, groups_col, smples, groups, obs, new_obs, min_cells, min_counts, min_prop)

    offset = 0
    if groups_col is None:

        # Remove features
        msk = np.sum(props, axis=0) > min_smpls
        psbulk[:, ~msk] = 0

    else:
        for i in range(len(groups)):

            # Get profiles and proportions per group
            prop = props[offset:offset + len(smples)]
            profile = psbulk[offset:offset + len(smples)]

            # Remove features
            msk = np.sum(prop, axis=0) >= min_smpls
            profile[:, ~msk] = 0

            # Append
            psbulk[offset:offset + len(smples)] = profile
            offset += len(smples)

    # Create new AnnData
    psbulk = AnnData(psbulk, obs=new_obs, var=new_var)

    # Remove empty samples
    psbulk = psbulk[~np.all(psbulk.X == 0, axis=1), ~np.all(psbulk.X == 0, axis=0)]

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

    df = melt([logFCs, pvals]).rename({'sample': 'contrast', 'source': 'name', 'score': 'logFC'}, axis=1)
    df = df[['contrast', 'name', 'logFC', 'pval']].sort_values('pval').reset_index(drop=True)

    return df
