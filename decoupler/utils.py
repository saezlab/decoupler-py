"""
Utility functions.
Functions of general utility used in multiple places.
"""

import numpy as np
import pandas as pd

from .pre import extract, rename_net, get_net_mat, filt_min_n

from anndata import AnnData


def m_rename(m, name):
    # Rename
    m = m.rename({'index': 'sample', 'variable': 'source'}, axis=1)

    # Assign score or pval
    if 'pval' in name:
        m = m.rename({'value': 'pval'}, axis=1)
    else:
        m = m.rename({'value': 'score'}, axis=1)

    return m


def melt(df):
    """
    Function to generate a long format dataframe similar to the one obtained in the R implementation of decoupler.

    Parameters
    ----------
    df : dict, tuple, list or DataFrame
        Output of decouple, of an individual method or an individual dataframe.

    Returns
    -------
    m : DataFrame
        Melted long format dataframe.
    """

    # If input is result from decoule function
    if type(df) is list or type(df) is tuple:
        df = {k.name: k for k in df}
    if type(df) is dict:
        # Get methods run
        methods = np.unique([k.split('_')[0] for k in df])

        res = []
        for methd in methods:
            for k in df:
                # Melt estimates
                if methd in k and 'pvals' not in k:
                    m = df[k].reset_index().melt(id_vars='index')
                    m = m_rename(m, k)
                    if 'estimate' not in k:
                        name = methd + '_' + k.split('_')[1]
                    else:
                        name = methd
                    m['method'] = name
                    # Extract pvals from this method
                    if methd+'_pvals' in df:
                        pvals = df[methd+'_pvals'].reset_index().melt(id_vars='index')['value'].values
                    else:
                        pvals = np.full(m.shape[0], np.nan)
                    m['pval'] = pvals

                    res.append(m)

        # Concat results
        m = pd.concat(res)

    # If input is an individual dataframe
    elif type(df) is pd.DataFrame:

        # Melt
        name = df.name
        m = df.reset_index().melt(id_vars='index')

        # Rename
        m = m_rename(m, name)

    else:
        raise ValueError('Input type {0} not supported.'.format(type(df)))

    return m


def show_methods():
    """
    Shows available methods.
    The first column correspond to the function name in decoupler and the second to the method's full name.

    Returns
    -------
    df : DataFrame
        Dataframe with the available methods.
    """

    import decoupler

    df = []
    lst = dir(decoupler)
    for m in lst:
        if m.startswith('run_'):
            name = getattr(decoupler, m).__doc__.split('\n')[1].lstrip()
            df.append([m, name])
    df = pd.DataFrame(df, columns=['Function', 'Name'])

    return df


def check_corr(net, source='source', target='target', weight='weight', mat=None, min_n=5, use_raw=True):
    """
    Checks the correlation across the regulators in a network. If a mat is also provided, target features will be prunned to
    match the ones in mat.

    Parameters
    ----------
    net : DataFrame
        Network in long format.
    source : str
        Column name with source nodes.
    target : str
        Column name with target nodes.
    weight : str
        Column name with weights.
    mat : list, DataFrame or AnnData
        Optional. If given, target features be filtered if they are
        not in mat.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    use_raw : bool
        Use raw attribute of mat if present.

    Returns
    -------
    corr : DataFrame
        Correlation pairs dataframe.
    """

    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)

    # If mat is provided
    if mat is not None:
        # Extract sparse matrix and array of genes
        _, _, c = extract(mat, use_raw=use_raw)
    else:
        c = np.unique(net['target'].values).astype('U')

    net = filt_min_n(c, net, min_n=min_n)
    sources, targets, net = get_net_mat(net)

    # Compute corr
    corr = np.round(np.corrcoef(net, rowvar=False), 4)

    # Filter upper diagonal
    corr = pd.DataFrame(np.triu(corr, k=1), index=sources, columns=sources).reset_index()
    corr = corr.melt(id_vars='index').rename({'index': 'source1', 'variable': 'source2', 'value': 'corr'}, axis=1)
    corr = corr[corr['corr'] != 0]

    # Sort by abs value
    corr = corr.iloc[np.argsort(np.abs(corr['corr'].values))[::-1]].reset_index(drop=True)

    return corr


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


def get_toy_data(n_samples=24, seed=42):
    """
    Generate a toy `mat` and `net` for testing.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed to use.

    Returns
    -------
    mat : DataFrame
        `mat` example.
    net : DataFrame
        `net` example.
    """

    from numpy.random import default_rng

    # Network model
    net = pd.DataFrame([['T1', 'G01', 1], ['T1', 'G02', 1], ['T1', 'G03', 0.7], ['T2', 'G06', 1], ['T2', 'G07', 0.5],
                        ['T2', 'G08', 1], ['T3', 'G06', -0.5], ['T3', 'G07', -3], ['T3', 'G08', -1], ['T3', 'G11', 1]],
                       columns=['source', 'target', 'weight'])

    # Simulate two population of samples with different molecular values
    rng = default_rng(seed=seed)
    n_features = 12
    n = int(n_samples/2)
    res = n_samples % 2
    row_a = np.array([8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0])
    row_b = np.array([0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0])
    row_a = [row_a + np.abs(rng.normal(size=n_features)) for _ in range(n)]
    row_b = [row_b + np.abs(rng.normal(size=n_features)) for _ in range(n+res)]

    mat = np.vstack([row_a, row_b])
    features = ['G{:02d}'.format(i+1) for i in range(n_features)]
    samples = ['S{:02d}'.format(i+1) for i in range(n_samples)]
    mat = pd.DataFrame(mat, index=samples, columns=features)

    return mat, net


def summarize_acts(acts, groupby, obs=None, var=None, mode='mean', min_std=1.0):
    """
    Summarizes activities obtained per group by their mean or median and removes features that do not change across samples.

    Parameters
    ----------
    acts : AnnData or DataFrame
        Activities obtained after running a method.
    groupby : str
        Column name of obs to use for grouping.
    obs : DataFrame
        None or a data-frame with sample meta-data.
    var : DataFrame
        None or a data-frame with feature meta-data.
    mode : str
        Wheter to use mean or median to summarize.
    min_std : float
        Minimum std to filter out features. Only features with enough variability will be returned. Decrease it to return more
        features.

    Returns
    -------
    summary : DataFrame
        Dataframe with summaried actvities per group.
    """

    # Extract acts, obs and features
    if type(acts) is AnnData:
        if obs is not None or var is not None:
            raise ValueError('If acts is AnnData, obs and var need to be None.')
        obs = acts.obs[groupby].values.astype('U')
        features = acts.var.index.values.astype('U')
        acts = acts.X
    else:
        obs = obs[groupby].values.astype('U')
        features = var.index.astype('U')

    # Get sizes
    groups = np.unique(obs)
    n_groups = len(groups)
    n_features = acts.shape[1]

    # Init empty mat
    summary = np.zeros((n_groups, n_features), dtype=np.float32)

    for i in range(n_groups):
        msk = obs == groups[i]
        if mode == 'mean':
            summary[i] = np.mean(acts[msk], axis=0, where=np.isfinite(acts[msk]))
        elif mode == 'median':
            summary[i] = np.median(acts[msk], axis=0)
        else:
            raise ValueError('mode can only be either mean or median.')

    # Filter by min_std
    min_std = np.abs(min_std)
    msk = np.std(summary, axis=0, ddof=1) > min_std

    # Transform to df
    summary = pd.DataFrame(summary[:, msk], columns=features[msk], index=groups)

    return summary


def assign_groups(summary):
    """
    Assigns group labels based on summary activities. The maximum positive value is used for assigment.

    Parameters
    ----------
    summary : DataFrame
        Dataframe with summaried actvities per group

    Returns
    -------
    annot_dict : dict
        Dictionary with the group that had the maximum positive activity.
    """

    # Extract from summary
    obs = np.unique(summary.index.values.astype('U'))
    groups = np.unique(summary.columns.values.astype('U'))
    summary = summary.values

    # Get lens
    n_obs = len(obs)

    # Find max value and assign
    annot_dict = dict()
    for i in range(n_obs):
        o = obs[i]
        mx = np.max(summary[i])
        idx = np.where(summary[i] == mx)[0][0]
        annot_dict[o] = groups[idx]

    return annot_dict


def get_top_targets(logFCs, pvals, name, contrast, net, source='source', target='target', weight='weight', sign_thr=1,
                    lFCs_thr=0.0):
    """
    Return significant target features for a given source and contrast.

    Parameters
    ----------
    logFCs : DataFrame
        Data-frame of logFCs (contrasts x features).
    pvals : DataFrame
        Data-frame of p-values (contrasts x features).
    name : str
        Name of the source to plot.
    contrast : str
        Name of the contrast (row) to plot.
    net : DataFrame
        Network data-frame.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    sign_thr : float
        Significance threshold for p-values.
    lFCs_thr : float
        Significance threshold for logFCs.

    Returns
    -------
    df : DataFrame
        Dataframe containing the significant targets of a given source.
    """

    # Rename net
    net = rename_net(net, source=source, target=target, weight=weight)

    # Find targets in net that match with logFCs
    targets = net[net['source'] == name]['target'].values
    msk = np.isin(logFCs.columns, targets)

    # Build df
    df = logFCs.loc[[contrast], msk].T.rename({contrast: 'logFC'}, axis=1)
    df['pval'] = pvals.loc[[contrast], msk].T
    df = df.sort_values('pval')

    # Filter by thresholds
    df = df[(np.abs(df['logFC']) > lFCs_thr) & (np.abs(df['pval']) < sign_thr)]

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
