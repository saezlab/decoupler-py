"""
Utility functions to benchmark methods and nets.
Functions to benchmark methods and nets using perturbation experiments.
"""

import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy.sparse import csr_matrix

from .utils import get_toy_data
from .pre import match
from .metrics import metric_auroc, metric_auprc, metric_mcauroc, metric_mcauprc


def get_toy_benchmark_data(n_samples=24, seed=42, shuffle_perc=0.25):
    """
    Generate a toy mat, net and obs for testing the benchmark pipeline.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed to use.
    shuffle_perc : float
        Percentage of the ground truth to randomize.

    Returns
    -------
    mat : DataFrame
        mat example.
    net : DataFrame
        net example.
    obs : DataFrame
        obs example.
    """

    # Get toy data
    mat, net = get_toy_data(n_samples=n_samples, seed=seed)

    # Simulate 2 populations of perturbations
    obs = pd.DataFrame(columns=['group'])
    n_samples = mat.shape[0]
    n = int(n_samples/2)
    res = n_samples % 2
    obs['perturb'] = [['T1', 'T2'] for _ in range(n)] + [['T3', 'T4'] for _ in range(n+res)]
    obs['group'] = np.tile(['CA', 'CB'], n+res)[: obs['perturb'].size]
    obs['sign'] = 1
    obs.index = mat.index.copy()

    # Shuffle a percentage of the samples
    idxs = np.arange(mat.shape[0])
    rng = default_rng(seed=seed)
    idxs = rng.choice(idxs, int(idxs.size * shuffle_perc), replace=False)
    r_idxs = rng.choice(idxs, idxs.size, replace=False)
    mat.iloc[r_idxs] = mat.iloc[idxs].values

    return mat, net, obs


def show_metrics():
    """
    Shows available evaluation metrics.
    The first column correspond to the function name in decoupler and the second to the metrics's full name.

    Returns
    -------
    df : DataFrame
        Dataframe with the available metrics.
    """

    import decoupler

    df = []
    lst = dir(decoupler)
    for m in lst:
        if m.startswith('metric_'):
            name = getattr(decoupler, m).__doc__.split('\n')[1].lstrip()
            df.append([m, name])
    df = pd.DataFrame(df, columns=['Function', 'Name'])

    return df


def validate_metrics(metrics):

    # Check if not list
    if type(metrics) is str:
        metrics = [metrics]

    # Retrieve available metrics
    a_metrics = [metric.split('metric_')[1] for metric in show_metrics()['Function']]

    # Check if given metrics exist
    for metric in metrics:
        if metric not in a_metrics:
            raise ValueError("""Metric {0} not available, please run show_metrics() to see the list of available
            metrics.""".format(metric))


def compute_metric(act, grt, metric, pi0=0.5, n_iter=1000, seed=42):

    if metric == 'auroc':
        scores = metric_auroc(grt, act)
    elif metric == 'auprc':
        scores = metric_auprc(grt, act, pi0=pi0)
    elif metric == 'mcauroc':
        scores = metric_mcauroc(grt, act, n_iter=n_iter, seed=seed)
    elif metric == 'mcauprc':
        scores = metric_mcauprc(grt, act, n_iter=n_iter, seed=seed)

    # Output must be list
    if type(scores) is not np.ndarray:
        scores = np.array([scores])

    return scores


def append_by_experiment(df, grpby_i, grp, act, grt, srcs, mthds, metrics, min_exp=5, pi0=0.5,
                         n_iter=1000, seed=42):

    # Flatten act by method
    act, grt = act.reshape(-1, act.shape[-1]).T, grt.flatten()

    # Compute per method and metric
    for m in range(len(mthds)):
        mth = mthds[m]
        for metric in metrics:
            # identify activity scores with NAs in each method
            act_i = act[m]
            nan_mask = np.isnan(act_i)
            # Remove NAs from activity matrix and ground truth
            act_i = act_i[~nan_mask]
            grt_i = grt[~nan_mask]
            # Compute Class Imbalance
            ci = np.sum(grt_i) / len(grt_i)
            # Compute metrics
            scores = compute_metric(act_i, grt_i, metric, pi0=pi0, n_iter=n_iter, seed=seed)
            for score in scores:
                row = [grpby_i, grp, None, mth, metric, score, ci]
                df.append(row)


def append_by_source(df, grpby_i, grp, act, grt, srcs, mthds, metrics, min_exp=5, pi0=0.5,
                     n_iter=1000, seed=42):
    
    for m in range(len(mthds)):
        mth = mthds[m]
        act_i = act[:,:,m]
        nan_mask = np.isnan(act_i)
        
        grt_i = grt.copy()
        grt_i[nan_mask]=np.nan

        # Remove sources with less than min_exp
        src_msk = np.sum(grt_i > 0., axis=0) >= min_exp
        act_i, grt_i = act[:, src_msk, :], grt_i[:, src_msk]
        srcs_method = srcs[src_msk]

        # Compute per source, method and metric
        for s in range(len(srcs_method)):
            src = srcs_method[s]
            tmp_grt = grt_i[:, s]
            nan_mask = np.isnan(tmp_grt)
    
            grt_source = tmp_grt[~nan_mask] 
            act_source = act_i[:, s, m][~nan_mask]

            # Compute Class Imbalance
            ci = np.sum(grt_source) / len(grt_source)
            if ci != 0. and ci != 1.:
                for metric in metrics:
                    scores = compute_metric(act_source, grt_source, metric, pi0=pi0, n_iter=n_iter, seed=seed)
                    for score in scores:
                        row = [grpby_i, grp, src, mth, metric, score, ci]
                        df.append(row)


def append_metrics_scores(df, grpby_i, grp, act, grt, srcs, mthds, metrics, by, min_exp=5, pi0=0.5,
                          n_iter=1000, seed=42):

    if not min_exp > 0:
        raise ValueError('Argument min_exp must be bigger than 0.')

    if by == 'experiment':
        append_by_experiment(df, grpby_i, grp, act, grt, srcs, mthds, metrics, min_exp=min_exp, pi0=pi0,
                             n_iter=n_iter, seed=seed)

    elif by == 'source':
        append_by_source(df, grpby_i, grp, act, grt, srcs, mthds, metrics, min_exp=min_exp, pi0=pi0,
                         n_iter=n_iter, seed=seed)


def adjust_sign(mat, v_sign):
    v_sign = v_sign.reshape(-1, 1)
    if isinstance(mat, csr_matrix):
        mat = mat.multiply(v_sign).tocsr()
    else:
        mat = mat * v_sign
    return mat


def build_acts_tensor(res):

    # Get unique methods
    mthds = [m for m in res.keys() if '_pvals' not in m]

    # Extract dimensions
    exps = res[mthds[0]].index.values
    srcs = res[mthds[0]].columns.values

    # Build acts tensor and sort by exps and srcs
    n_exp, n_src, n_mth = len(exps), len(srcs), len(mthds)
    acts = np.zeros((n_exp, n_src, n_mth))
    for i, m in enumerate(mthds):
        acts[:, :, i] = res[m].values
    msk = np.argsort(srcs)
    acts = acts[:, msk]
    srcs = srcs[msk]
    msk = np.argsort(exps)
    exps = exps[msk]

    return acts, exps, srcs, mthds


def build_grts_mat(obs, exps, srcs):

    # Explode nested perturbs and pivot into mat
    grts = obs.explode('perturb').pivot(columns='perturb', values='sign').fillna(0.)

    # Sort by columns (srcs) and by rows (exps)
    msk = np.argsort(grts.columns)
    grts = grts.loc[exps].iloc[:, msk]

    # Remove cols that are not in res srcs
    msk = np.isin(grts.columns.values, srcs)
    grts = grts.loc[:, msk]

    return grts


def unique_obs(col):

    # Gets unique categories from a column with both lists and elements.

    # Init empty cats
    cats = set()

    for row in col:

        # Check if col elements are lists
        if type(row) is list:
            for r in row:
                if r not in cats:
                    cats.add(r)
        else:
            if row not in cats:
                cats.add(row)

    return np.sort(list(cats))


def build_msks_tensor(obs, groupby):

    # If groupby
    if groupby is not None:

        # Init empty lsts
        msks = []
        grps = []
        grpbys = []
        for grpby_i in groupby:

            # Handle nested groupbys
            if type(grpby_i) is list:
                grpby_i = np.sort(grpby_i)
                grpby_name = '|'.join(grpby_i)
                if grpby_i.size > 1:
                    obs[grpby_name] = obs[grpby_i[0]].str.cat(obs[grpby_i[1:]], sep='|')
                grpby_i = grpby_name

            # Find msk in obs based on groupby
            grps_j = unique_obs(obs[grpby_i].values)
            msk_i = []
            grps_i = []
            for grp in grps_j:
                m = np.array([grp in lst for lst in obs[grpby_i]])
                msk_i.append(m)
                grps_i.append(grp)

            # Append
            msks.append(msk_i)
            grpbys.append(grpby_i)
            grps.append(grps_i)

    else:
        msks = None
        grpbys = None
        grps = None

    return msks, grpbys, grps


def format_acts_grts(res, obs, groupby):

    # Build acts tensor and sort by exps and srcs
    acts, exps, srcs, mthds = build_acts_tensor(res)

    # Make sure obs and acts match by exps idxs
    obs = obs.loc[exps]

    # Build sorted and filtered grts mat
    grts = build_grts_mat(obs, exps, srcs)

    # Match to same srcs between acts and grts
    grts = match(srcs, grts.columns, grts.T).T

    # Build msks tensor
    msks, grpbys, grps = build_msks_tensor(obs, groupby)

    return acts, grts, msks, srcs, mthds, grpbys, grps


def rename_obs(obs, perturb, sign):

    # Check if names are in columns
    msg = 'Column name "{0}" not found in obs. Please specify a valid column.'
    assert perturb in obs.columns, msg.format(perturb)

    # Check that they are not the same
    if perturb == sign:
        raise ValueError("perturb={0} and sign={1} cannot have the same value.".format(perturb, sign))

    # Validate sign
    if type(sign) is str:
        assert sign in obs.columns, msg.format(sign)
        unq = np.sort(np.unique(obs[sign].values))
        lbl = np.array([-1, 1])
        msg = '`sign` values can only be -1 or 1, got {0}.'
        assert np.all(np.isin(unq, lbl)), msg.format(list(unq))
    elif sign == 1 or sign == -1:
        obs = obs.copy()
        obs['sign'] = sign
        sign = 'sign'
    else:
        raise ValueError("If sign is not a column name, it must be 1 or -1.")

    # Rename
    obs = obs.rename(columns={perturb: 'perturb', sign: 'sign'})

    return obs


def check_groupby(obs, groupby, perturb, by):

    if groupby is not None:
        if type(groupby) is str:
            groupby = [groupby]

        for grp_i in groupby:
            if type(grp_i) is str:
                grp_i = [grp_i]
            # For each group inside each groupby
            for grp_j in grp_i:

                # Check if perturb is in groupby when by=source
                msg = 'perturb="{0}" column cannot be in groupby if by="source". Please remove it.'
                assert not (perturb == grp_j and by == 'source'), msg.format(perturb)

                # Assert that columns exist in obs
                msg = 'Column name "{0}" not found in obs. Please specify a valid column.'
                assert grp_j in obs.columns, msg.format(grp_j)

                # Assert that column doesn't contain "|"
                msg = "Column names cannot contain the character \"|\", please rename column {0}.".format(grp_j)
                assert '|' not in grp_j

    return groupby
