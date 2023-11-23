"""
Functions to benchmark methods and nets.
Functions to benchmark methods and nets using perturbation experiments.
"""

import numpy as np
import pandas as pd

from .decouple import decouple
from .utils_anndata import extract_psbulk_inputs
from .pre import rename_net, filt_min_n
from .utils_benchmark import format_acts_grts, append_metrics_scores, adjust_sign
from .utils_benchmark import validate_metrics, check_groupby, rename_obs


def get_performances(res, obs, groupby, by, metrics, min_exp=5, pi0=0.5, n_iter=1000,
                     seed=42, verbose=False):

    # Return acts, grts and msks tensors
    acts, grts, msks, srcs, mthds, grpbys, grps = format_acts_grts(res, obs, groupby)

    # Init empty df
    df = []
    if msks is not None:
        n_grpbys = len(msks)
        for i in range(n_grpbys):
            msk_i = msks[i]
            grpby_i = grpbys[i]
            grps_i = grps[i]
            n_grps = len(grps_i)
            if verbose:
                print('Computing metrics for groupby {0}...'.format(grpby_i))
            for j in range(n_grps):
                msk = msk_i[j]
                grp = grps_i[j]
                n = np.sum(msk)

                # If enough exps, subset by group
                if n >= min_exp:
                    act, grt = acts[msk, :, :], grts[msk, :]

                    # Special case when groupby == perturb, remove extra grts
                    if grp in srcs:
                        m = grp == srcs
                        grt[:, ~m] = 0.

                    # Compute and append scores to df
                    append_metrics_scores(df, grpby_i, grp, act, grt, srcs, mthds, metrics, by, min_exp=min_exp,
                                          pi0=pi0, n_iter=n_iter, seed=seed)
    else:
        n_exp = acts.shape[0]
        if n_exp >= min_exp:

            # Compute and append scores to df
            if verbose:
                print('Computing metrics...')
            append_metrics_scores(df, None, None, acts, grts, srcs, mthds, metrics, by, min_exp=min_exp,
                                  pi0=pi0, n_iter=n_iter, seed=seed)

    # Format df
    df = pd.DataFrame(df, columns=['groupby', 'group', 'source', 'method', 'metric', 'score', 'ci'])

    return df


def format_benchmark_inputs(mat, obs, perturb, sign, net, groupby, by, f_expr=True, f_srcs=False,
                            source='source', target='target', weight='weight', min_n=5,
                            verbose=False, use_raw=True, decouple_kws={}):

    # Extract inputs
    if verbose:
        print("Extracting inputs...")
    mat, obs, var = extract_psbulk_inputs(mat, obs, layer=None, use_raw=use_raw)

    # Format groupby
    groupby = check_groupby(obs, groupby, perturb, by)

    # Rename obs
    obs = rename_obs(obs, perturb, sign)

    # Rename net
    if verbose:
        print("Formating net...")
    net = rename_net(net, source=decouple_kws['source'], target=decouple_kws['target'], weight=decouple_kws['weight'])
    net = filt_min_n(var.index.values.astype('U'), net, min_n=decouple_kws['min_n'])

    # Remove experiments without sources in net
    if f_expr:
        msk = np.full((obs['perturb'].size, ), False)
        srcs = net['source'].values.astype('U')
        for i, src in enumerate(obs['perturb']):
            msk[i] = np.any(np.isin(src, srcs))
        if verbose:
            n = np.sum(~msk)
            print("{0} experiments without sources in net, they will be removed.".format(n))
        mat, obs = mat[msk], obs.loc[msk]

    # Remove sources without experiments in obs
    if f_srcs:
        msk = np.isin(net['source'].values, obs['perturb'].values.ravel())
        if verbose:
            n = np.sum(~msk)
            print("{0} sources without experiments in obs, they will be removed.".format(n))
        net = net.loc[msk]

    return mat, obs, var, net, groupby


def _benchmark(mat, obs, net, perturb, sign, metrics=['auroc', 'auprc'], groupby=None, by='experiment', f_expr=True,
               f_srcs=False, min_exp=5, pi0=0.5, n_iter=1000, seed=42, verbose=True, use_raw=True, decouple_kws={}):

    # Format inputs
    mat, obs, var, net, groupby = format_benchmark_inputs(mat, obs, perturb, sign, net, groupby, by, f_expr=f_expr,
                                                          f_srcs=f_srcs, verbose=verbose, use_raw=use_raw,
                                                          decouple_kws=decouple_kws)

    # Adjust sign
    mat = adjust_sign(mat, obs['sign'].values)
    obs['sign'] = 1

    # Reset net names args
    decouple_kws['source'] = 'source'
    decouple_kws['target'] = 'target'
    decouple_kws['weight'] = 'weight'

    # Run prediction
    if verbose:
        print('Running methods...')
    res = decouple([mat, obs.index, var.index], net, verbose=verbose, **decouple_kws)

    # Compute metrics
    if verbose:
        print('Calculating metrics...')
    df = get_performances(res, obs, groupby, by, metrics, min_exp=min_exp, pi0=pi0,
                          n_iter=n_iter, seed=seed, verbose=verbose)
    if verbose:
        print('Done.')

    return df


def benchmark(mat, obs, net, perturb, sign, metrics=['auroc', 'auprc', 'mcauroc', 'mcauprc', 'rank', 'nrank'], groupby=None,
              by='experiment', f_expr=True, f_srcs=False, min_exp=5, pi0=0.5, n_iter=1000, seed=42,
              verbose=True, use_raw=True, decouple_kws={}):
    """
    Benchmark methods or networks on a given set of perturbation experiments using activity inference with decoupler.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    obs : DataFrame or None
        Metadata containing the perturbed targets and the sign of the perturbation. If mat is AnnData, use mat.obs
        attribute instead.
    net : DataFrame, dict
        Network in long format. Can be dictionary of nets, where key is the name and value is the long format DataFrame.
    perturb : str
        Column name in obs with perturbed sources.
    sign : str, int
        Column name in obs with sign of the perturbation. Can be set to 1 or -1 if all experiments are overexpression or
        knockouts, respectively.
    metrics : list, str
        Performance metric(s) to compute. See the description of get_performance for more details.
    groupby : list, str, None
        Performance metrics(s) can be computed per groups if enough experiments are available.
    by : str
        Whether to evaluate performances at the "experiment" or at the "source" level.
    f_expr : bool
        Whether to filter out experiments whose perturbed sources are not in the given net. Defaults to True.
    f_srcs : bool
        Whether to fitler out sources in net for which there are not perturbation data. Defaults to False.
    min_exp : int
        Minimum of perturbation experiments per group.
    pi0 : float
        Reference ratio for calibrated metrics. Corresponds to the baseline/reference class inbalance to which
        to set the metric.
    n_iter : int
        Number of downsampling iterations used for the 'mcroc' and 'mcprc' metrics.
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.
    decouple_kws : dict
        Parameters for the decoupler.decouple function. If more than one net, use a nested dictionary where the main
        key is the network name and the value is a dictionary with the requiered arguments.

    Returns
    -------
    df : DataFrame
        DataFrame containing the metrics' scores.
    """

    # Init default args
    default_kws = {'source': 'source', 'target': 'target', 'weight': 'weight', 'min_n': 5}

    # Validate by
    if by not in ['experiment', 'source']:
        raise ValueError('Argument `by` has to be either "experiment" or "source".')

    # Validate metrics
    validate_metrics(metrics)

    # Validate pi0
    if pi0 is not None:
        if pi0 < 0 or pi0 > 1:
            raise ValueError('Argument `pi0` needs to be between 0 and 1.')

    # Run benchmark per net
    if type(net) is not dict:

        # Update decouple args
        decouple_kws = {**default_kws, **decouple_kws}

        # Run benchmark
        df = _benchmark(mat, obs, net, perturb, sign, metrics, groupby, by, f_expr, f_srcs, min_exp, pi0,
                        n_iter, seed, verbose, use_raw, decouple_kws)
    else:
        df = []
        for net_name in net:

            if verbose:
                print('Using {0} network...'.format(net_name))

            # Update decouple args
            decouple_kws.setdefault(net_name, {})
            decouple_kws[net_name] = {**default_kws, **decouple_kws[net_name]}

            # Run benchmark
            tmp = _benchmark(mat, obs, net[net_name], perturb, sign, metrics, groupby, by, f_expr, f_srcs,
                             min_exp, pi0, n_iter, seed, verbose, use_raw, decouple_kws[net_name])
            tmp['net'] = net_name
            df.append(tmp)

        # Merge all results
        df = pd.concat(df)

    return df
