"""
Utility functions.
Functions of general utility used in multiple places.
"""

import numpy as np
from numpy.random import default_rng
from scipy.sparse import csr_matrix
import pandas as pd

from .pre import extract, rename_net, get_net_mat, filt_min_n

from anndata import AnnData
from tqdm import tqdm


def m_rename(m, name):
    # Rename
    m = m.rename({'index': 'sample', 'variable': 'source'}, axis=1)

    # Assign score or pval
    if 'pvals' in name:
        m = m.rename({'value': 'pvals'}, axis=1)
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
                    m['pvals'] = pvals

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


def get_toy_data(n_samples=24, seed=42):
    """
    Generate a toy mat and net for testing.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed to use.

    Returns
    -------
    mat : DataFrame
        mat example.
    net : DataFrame
        net example.
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


def summarize_acts(acts, groupby, obs=None, mode='mean', min_std=1.0):
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
        if obs is not None:
            raise ValueError('If acts is AnnData, obs needs to be None.')
        obs = acts.obs[groupby].values.astype('U')
        features = acts.var.index.values.astype('U')
        acts = acts.X
    else:
        obs = obs[groupby].values.astype('U')
        features = acts.columns.astype('U')
        acts = acts.values

    # Get sizes
    groups = np.unique(obs)
    n_groups = len(groups)
    n_features = acts.shape[1]

    # Init empty mat
    summary = np.zeros((n_groups, n_features), dtype=np.float32)

    for i in range(n_groups):
        msk = obs == groups[i]
        f_msk = np.isfinite(acts[msk])
        for i in range(n_groups):
            msk = obs == groups[i]
            grp_acts = acts[msk]
            for j in range(n_features):
                ftr_acts = grp_acts[:, j]
                f_msk = np.isfinite(ftr_acts)
                if mode == 'mean':
                    summary[i, j] = np.mean(ftr_acts[f_msk])
                elif mode == 'median':
                    summary[i, j] = np.median(ftr_acts[f_msk])
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


def p_adjust_fdr(p):
    """
    Benjamini-Hochberg p-value correction for multiple hypothesis testing.

    Parameters
    ----------
    p : ndarray, list
        Array or list of p-values to correct.

    Returns
    -------
    corr_p : ndarray
        Array of corrected p-values.
    """

    # Code adapted from: https://stackoverflow.com/a/33532498/8395875
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    corr_p = q[by_orig]

    return corr_p


def dense_run(func, mat, net, source='source', target='target', weight='weight', min_n=5, verbose=False, use_raw=True,
              args={}, estimate_loc=0):
    """
    Run a method without zero values.

    This function runs any method in decoupler (see `dc.show_methods()`) in a dense manner, meaning that all zero vales
    are removed for each sample. Since this is sample dependent, parallelization is not available most of the time and
    running times might increase. This function is useful to test what effect does null imputation do to the inference of
    activities.

    Parameters
    ----------
    func : function
        Function to call a decoupler method, check `dc.show_methods()` for the full list.
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights (if needed).
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.
    args : dict
        A dict of argument to pass to func.
    estimate_loc : int
        Which index is the desired estimate. Only relevant for methods that return more than
        one estime like wmean, wsum or gsea.

    Returns
    -------
    estimate : DataFrame
        Estimate scores. Stored in `.obsm['*_estimate']` if `mat` is AnnData.
    pvals : DataFrame
        Obtained p-values. Stored in `.obsm['*_pvals']` if `mat` is AnnData.
    """

    # Extract sparse matrix and array of features
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)
    fname = func.__name__.split('run_')[1]

    if verbose:
        print('Dense run of {0} on mat with {1} samples and {2} potential targets.'.format(fname, m.shape[0], len(c)))

    net = rename_net(net, source=source, target=target, weight=weight)

    acts, pvals = [], []
    for i in tqdm(range(m.shape[0]), disable=not verbose):

        i_r = r[[i]]

        # Extract single sample
        if isinstance(m, csr_matrix):
            sample = m[i].A[0]
            idx = sample.indices
        else:
            sample = m[i]
            idx = sample != 0.

        # Remove zeros
        i_c = c[idx]
        sample = sample[idx][np.newaxis, :]

        # Run activity
        row = [sample, i_r, i_c]

        # Overwrite min_n, verbose and use_raw
        args['min_n'], args['verbose'], args['use_raw'] = min_n, False, use_raw

        # Check if weight method or not
        is_weighted = 'weight' in func.__code__.co_varnames

        # Test if it can run
        try:
            sub_net = filt_min_n(i_c, net, min_n=min_n)
            skip = False
        except ValueError:
            act = pd.DataFrame([], index=i_r)
            pval = act.copy()
            acts.append(act)
            pvals.append(pval)
            skip = True
        if skip:
            continue

        # Run method
        if is_weighted:
            act = func(mat=row, net=sub_net, source=source, target=target, weight=weight, **args)
        else:
            act = func(mat=row, net=sub_net, source=source, target=target, **args)

        # Split
        if type(act) is tuple:
            act, pval = act[estimate_loc], act[-1]
        else:
            pval = act.copy()
            pval.loc[:, :] = np.nan

        # Append
        acts.append(act)
        pvals.append(pval)

    # Join
    acts = pd.concat(acts, join='outer')
    pvals = pd.concat(pvals, join='outer')

    # Name
    acts.name = '{0}_estimate'.format(fname)
    pvals.name = '{0}_pvals'.format(fname)

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[acts.name] = acts
        mat.obsm[pvals.name] = pvals
    else:
        return acts, pvals


def shuffle_net(net, target=None, weight=None, seed=42, same_seed=True):
    """
    Shuffle network to make it random.

    Shuffle a given net by targets, weight or both at the same time.
    If only targets are shuffled, targets will change but the distirbution of weights for each footprint will be preserved.
    If only weights are shuffled, targets will be the same but the distirbution of weights for each footprint will change.
    If targets and weights are shuffled at the same time, both targets and weight distirbtion will change for each footprint.

    Parameters
    ----------
    net : DataFrame
        Network in long format.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    seed : int
        Random seed to use.
    same_seed : bool
        Whether to share seed between targets and weights if both are not None.

    Returns
    -------
    rnet : DataFrame
        Shuffled network.
    """

    # Make copy of net
    rnet = net.copy()

    # Shuffle
    if target is None and weight is None:
        raise ValueError('If target and weight are None, nothing is shuffled.')

    if target is not None:
        if target not in rnet.columns:
            raise ValueError('Colum target="{0}" not in rnet. Specify a valid column name.'.format(target))
        else:
            rng = default_rng(seed=seed)
            rng.shuffle(rnet[target].values)

    if weight is not None:
        if weight not in rnet.columns:
            raise ValueError('Colum weight="{0}" not in rnet. Specify a valid column name.'.format(weight))
        else:
            if not same_seed:
                seed = seed + 1
            rng = default_rng(seed=seed)
            rng.shuffle(rnet[weight].values)

    return rnet


def read_gmt(path):
    """
    Read a GMT file and return a ``pd.DataFrame``.

    Parameters
    ----------
    path : str
        Path to GMT file containing gene sets.

    Returns
    -------
    df : DataFrame
        Gene sets as `pd.DataFrame`.
    """

    # Init empty df
    df = []

    # Read line per line
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip().split()

            # Extract gene set name
            set_name = line[0]

            # For each gene add an entry (skip link in [1])
            genes = line[2:]
            for gene in genes:
                df.append([set_name, gene])

    # Transform to df
    df = pd.DataFrame(df, columns=['source', 'target'])

    return df

def annotate_enrichment(enrichment, net, queried_gene_list):
    """
    Adds n_genes, lead_genes and lead_genes_count
    to an enrichment result, provided it includes a geneset field.
    
    Parameters
    ----------
    queried_gene_list : list of genes originally queried
    
    enrichment : data.frame
        Pandas data frame produced by one of the enrichment methods
        
    net : data.frame
        Pandas data frame with the net used for the above enrichment. 
        It should contain geneset and genesymbol fields
        
    Returns
    -------
    df : DataFrame
        Melted data frame based on enrichment, with additional fields
        `n_genes`, `lead_genes`, `lead_genes_count`.
        
    """
    qgl = queried_gene_list
    # melt the enrichment dataframe
    enrichment_melted = pd.melt(enrichment, value_vars=enrichment.columns, var_name='geneset', value_name='p-value')
    # add a column with the number of genes in each geneset
    enrichment_melted['n_genes'] = enrichment_melted['geneset'].map(net.groupby('geneset')['genesymbol'].count())
    # add a column with the number of lead genes in high_exp that belong to that msigdb geneset
    enrichment_melted['lead_genes'] = enrichment_melted['geneset'].map(lambda x: ", ".join(set(net[net['geneset'] == x].genesymbol.unique().tolist()).intersection(set(qgl))) )
    # add a column with the lead genes in high_exp that belong to that msigdb geneset
    enrichment_melted['lead_genes_count'] = enrichment_melted['geneset'].map(lambda x: len(set(net[net['geneset'] == x].genesymbol.unique().tolist()).intersection(set(qgl))))
    return enrichment_melted
    
    
