"""
Utility functions for AnnData objects.
Functions to process AnnData objects.
"""

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from anndata import AnnData


def extract_psbulk_inputs(adata, obs, layer):

    # Extract count matrix X
    if layer is not None:
        X = adata.layers[layer]
    elif type(adata) is AnnData:
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


def get_pseudobulk(adata, sample_col, groups_col, obs=None, layer=None, min_prop=0.2, min_cells=10, min_counts=1000,
                   min_smpls=2):
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
    obs : pd.DataFrame, None
        If provided, meta-data dataframe.
    layer : str
        If provided, which element of layers to use.
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
    Returns new AnnData object with unormalized pseudobulk profiles per sample
    and group.
    """

    # Use one column if the same
    if sample_col == groups_col:
        groups_col = None

    # Extract inputs
    X, obs, var = extract_psbulk_inputs(adata, obs, layer)

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


def get_contrast(adata, group_col, condition_col, condition, reference, method='t-test'):
    """
    Computes simple contrast between two conditions from pseudo-bulk profiles. After genertaing pseudo-bulk profiles with
    `dc.get_pseudobulk`, computes Differential Expression Analysis using scanpy's `rank_genes_groups` function between two
    conditions.

    Parameters
    ----------
    adata : AnnData
        Input pseudo-bulk AnnData object.
    group_col : str
        Column of `obs` where to extract the groups names.
    condition_col : str
        Column of `obs` where to extract the condition names.
    condition : str
        Name of the condition to test inside condition_col.
    reference : str
        Name of the reference to use inside condition_col. If 'rest', compare each group to the union of the rest of the group.
    method : str
        Method to use for scanpy's `rank_genes_groups` function.

    Returns
    -------
    Returns logfoldchanges and p-values per gene.
    """

    try:
        from scanpy.tl import rank_genes_groups
        from scanpy.get import rank_genes_groups_df
    except Exception:
        raise BaseException('scanpy is not installed. Please install it with: pip install scanpy')

    # Find unique groups
    groups = np.unique(adata.obs[group_col].values.astype(str))

    # Process reference
    if reference is None:
        reference = 'rest'

    # Init empty logFC and pvals
    logFCs = pd.DataFrame(columns=adata.var.index)
    p_vals = pd.DataFrame(columns=adata.var.index)

    for grp in groups:

        # Sub-set by group
        sub_adata = adata[adata.obs[group_col] == grp].copy()

        # Subset condition_col
        sub_adata.obs = sub_adata.obs[[condition_col]]

        # Transform string columns to categories (removes anndata warnings)
        if sub_adata.obs[condition_col].dtype == 'object' or sub_adata.obs[condition_col].dtype == 'category':
            sub_adata.obs[condition_col] = pd.Categorical(sub_adata.obs[condition_col])

        # Run DEA if enough samples
        unq, counts = np.unique(sub_adata.obs[condition_col], return_counts=True)
        is_ref_in_unq = True
        if reference is not None or reference != 'rest':
            is_ref_in_unq = reference in unq
        if np.min(counts) >= 2 and condition in unq and is_ref_in_unq:
            rank_genes_groups(sub_adata, groupby=condition_col, groups=[condition, reference], reference=reference,
                              method=method)
        else:
            continue

        # Extract DEA results
        df = rank_genes_groups_df(sub_adata, group=condition)
        df[group_col] = grp
        logFC = df.pivot(columns='names', index=group_col, values='logfoldchanges')
        logFCs = logFCs.append(logFC)
        p_val = df.pivot(columns='names', index=group_col, values='pvals')
        p_vals = p_vals.append(p_val)

    return logFCs, p_vals
