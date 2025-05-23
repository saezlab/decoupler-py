from typing import Tuple

import pandas as pd
import numpy as np
from anndata import AnnData

from decoupler._docs import docs
from decoupler._log import _log
from decoupler.pp.net import prune
from decoupler.mt._methods import _methods
from decoupler.mt._decouple import decouple
from decoupler.bm._pp import _validate_groupby, _validate_obs, _filter, _sign
from decoupler.bm.metric import dict_metric


def _testsign(
    adata: AnnData,
    mth: str,
    test: bool,
    thr: float,
) -> np.ndarray:
    assert isinstance(thr, (int, float)) and 0. <= thr <= 1., \
    'thr must be numeric and between 0 and 1'
    if test:
        sign = adata.obsm[f'padj_{mth}'].values <= thr
        sign = (adata.obsm[f'score_{mth}'].values > 0) & sign
    else:
        q = np.quantile(adata.obsm[f'score_{mth}'].values, 1 - thr, axis=1).reshape(-1, 1)
        sign = adata.obsm[f'score_{mth}'].values > q
    return sign


def _tensor_scores(
    adata: AnnData,
    thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    # Get unique methods
    has_test = {m.name: m.test for m in _methods}
    has_test = has_test | {'consensus': True}
    mthds = [k.replace('score_', '') for k in adata.obsm if k.startswith('score_')]
    # Extract dimensions
    exps = adata.obs_names
    srcs = adata.obsm[f'score_{mthds[0]}'].columns.values
    # Build acts tensor and sort by exps and srcs
    n_exp, n_src, n_mth = len(exps), len(srcs), len(mthds)
    scores = np.zeros((n_exp, n_src, n_mth))
    signs = np.zeros(scores.shape, dtype=np.bool_)
    for i, mth in enumerate(mthds):
        scores[:, :, i] = adata.obsm[f'score_{mth}'].values
        signs[:, :, i] = _testsign(adata=adata, mth=mth, test=has_test[mth], thr=thr)
    return scores, signs, srcs, mthds


def _tensor_truth(
    obs: pd.DataFrame,
    srcs: np.ndarray
) -> pd.DataFrame:
    # Explode nested perturbs and pivot into mat
    grts = obs.explode('source').pivot(columns='source', values='type_p').notna().astype(float).fillna(0.)
    miss_srcs = srcs[~np.isin(srcs, grts.columns)]
    miss_srcs = pd.DataFrame(0, index=grts.index, columns=miss_srcs)
    grts = pd.concat([grts, miss_srcs], axis=1)
    grts = grts.loc[:, srcs].values
    return grts


def _unique_obs(
    col: np.ndarray,
) -> np.ndarray:
    # Gets unique categories from a column with both lists and elements.
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


def _mask_grps(
    obs: pd.DataFrame,
    groupby: None | list,
    verbose: float,
) -> Tuple[list, list, list]:
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
            m = f'benchmark - grouping by {grpby_i}'
            _log(m, level='info', verbose=verbose)
            # Find msk in obs based on groupby
            grps_j = _unique_obs(obs[grpby_i].values)
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
        m = f'benchmark - running without grouping'
        _log(m, level='info', verbose=verbose)
        msks = None
        grpbys = None
        grps = None
    return msks, grpbys, grps


def _metric(
    y_score: np.ndarray,
    y_true: np.ndarray,
    y_sign: np.ndarray,
    metric: str,
    verbose: bool,
    **kwargs
) -> dict:
    f = dict_metric[metric]
    if metric == 'fscore':
        y_score = y_sign
    score = f(y_true=y_true, y_score=y_score, **kwargs)
    return {k: s for k, s in zip(f.scores, score)}


def _metric_scores(
    df: pd.DataFrame,
    grpby_i: str | None,
    grp: str | None,
    score: np.ndarray,
    sign: np.ndarray,
    grt: np.ndarray,
    srcs: np.ndarray,
    mthds: list,
    metrics: list,
    runby: str,
    emin: int,
    verbose: bool,
    **kwargs
) -> None:
    assert isinstance(metrics, (str, list)), 'metrics must be str or list'
    if isinstance(metrics, str):
        metrics = [metrics]
    if runby == 'expr':
        m = ('benchmark - evaluating by experiment on:\n' +
        f'n_expr={score.shape[0]}, n_sources={score.shape[1]} across metrics={metrics}')
        _log(m, level='info', verbose=verbose)
        for m in range(len(mthds)):
            mth = mthds[m]
            scr_i = score[:, :, m]
            sgn_i = sign[:, :, m]
            for metric in metrics:
                vals = _metric(
                    y_score=scr_i,
                    y_true=grt,
                    y_sign=sgn_i,
                    metric=metric,
                    verbose=verbose,
                    **kwargs
                )
                for cname, val in vals.items():
                    row = [grpby_i, grp, None, mth, cname, val]
                    df.append(row)
    elif runby == 'source':
        m = f'benchmark - evaluating by source'
        _log(m, level='info', verbose=verbose)
        for m in range(len(mthds)):
            mth = mthds[m]
            # Remove sources with less than emin
            src_msk = np.sum(grt > 0., axis=0) >= emin
            scr_i, sgn_i, grt_i = score[:, src_msk, :], sign[:, src_msk, :], grt[:, src_msk]
            srcs_method = srcs[src_msk]
            for s in range(len(srcs_method)):
                src = srcs_method[s]
                scr_source = scr_i[:, s, m]
                sgn_source = sgn_i[:, s, m]
                grt_source = grt_i[:, s]
                # Check that grt is not all the same
                unq_grt = np.unique(grt_source[~np.isnan(scr_source)])
                # Convert from vector to arr
                scr_source, sgn_source, grt_source = scr_source[np.newaxis], sgn_source[np.newaxis], grt_source[np.newaxis]
                if unq_grt.size > 1:
                    for metric in metrics:
                        vals = _metric(
                            y_score=scr_source,
                            y_true=grt_source,
                            y_sign=sgn_source,
                            metric=metric,
                            verbose=verbose,
                            **kwargs
                        )
                        for cname, val in vals.items():
                            row = [grpby_i, grp, src, mth, cname, val]
                            df.append(row)


def _eval_scores(
    adata: pd.DataFrame,
    groupby: None | list,
    runby: str,
    metrics: str | list,
    thr: float,
    emin: int,
    verbose: bool,
    **kwargs
) -> pd.DataFrame:
    scores, signs, srcs, mthds = _tensor_scores(adata=adata, thr=thr)
    grts = _tensor_truth(obs=adata.obs, srcs=srcs)
    msks, grpbys, grps = _mask_grps(obs=adata.obs, groupby=groupby, verbose=verbose)
    df = []
    if msks is not None:
        n_grpbys = len(msks)
        for i in range(n_grpbys):
            msk_i = msks[i]
            grpby_i = grpbys[i]
            grps_i = grps[i]
            n_grps = len(grps_i)
            for j in range(n_grps):
                msk = msk_i[j]
                grp = grps_i[j]
                n = np.sum(msk)
                if n >= emin:
                    score, sign, grt = scores[msk, :, :], signs[msk, :, :], grts[msk, :]
                    # Special case when groupby == perturb, remove extra grts
                    if grp in srcs:
                        m = grp == srcs
                        grt[:, ~m] = 0.
                    # Compute and append scores to df
                    _metric_scores(
                        df=df,
                        grpby_i=grpby_i,
                        grp=grp,
                        score=score,
                        sign=sign,
                        grt=grt,
                        srcs=srcs,
                        mthds=mthds,
                        metrics=metrics,
                        runby=runby,
                        emin=emin,
                        verbose=verbose,
                        **kwargs
                    )
    else:
        n_exp = scores.shape[0]
        if n_exp >= emin:
            # Compute and append scores to df
            _metric_scores(
                df=df,
                grpby_i=None,
                grp=None,
                score=scores,
                sign=signs,
                grt=grts,
                srcs=srcs,
                mthds=mthds,
                metrics=metrics,
                runby=runby,
                emin=emin,
                verbose=verbose,
                **kwargs
            )
    # Format df
    df = pd.DataFrame(df, columns=['groupby', 'group', 'source', 'method', 'metric', 'score'])
    # Remove repeated columns
    df = df.loc[:, df.nunique(dropna=False) > 1]
    return df


@docs.dedent
def benchmark(
    adata: AnnData,
    net: pd.DataFrame | dict,
    metrics: str | list = ['auc', 'fscore', 'qrank'],
    groupby: str | None = None,
    runby: str = 'expr',
    sfilt: bool = False,
    thr: float = 0.10,
    emin: int = 5,
    verbose: bool = False,
    kws_decouple: dict = dict(),
    **kwargs
):
    """
    Benchmark enrichment methods or networks on a given set of perturbation experiments.

    Parameters
    ----------
    %(adata)s
    %(net)s
    metrics
        Which metrics to use to evaluate classification performance.
    groupby
        Whether to group experiments by a metadata column in ``anndata.AnnData.obs``.
    runby
        Whether to evaluate performances at the experiment (``"expr"``) or at the source (``"source"``) level.
    sfilt
        Whether to fitler out sources in net for which there are not perturbation data.
    thr
        Threshold of significance.
        Methods that generate no p-value will be consider significant if the ``thr`` > :math:`1 - quantile`
    emin
        Minimum number of experiments per group.
    %(verbose)s
    kws_decouple
        Keyword arguments to pass to ``decoupler.mt.decouple``.

    Returns
    -------
    Dataframe containing metric scores.
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be anndata.AnnData'
    _validate_obs(adata.obs)
    groupby = _validate_groupby(obs=adata.obs, groupby=groupby, runby=runby)
    assert isinstance(emin, int) and emin > 0, 'emin must be int and > 0'
    # Init default args
    kws_decouple = kws_decouple.copy()
    kws_decouple.setdefault('tmin', 5)
    kws_decouple.setdefault('args', dict())
    # Clean adata
    for col in list(adata.obsm.keys()):
        if col.startswith('score_') or col.startswith('padj_'):
            del adata.obsm[col]
    # Run benchmark per net
    if isinstance(net, pd.DataFrame):
        m = f'benchmark - running benchmark for one network'
        _log(m, level='info', verbose=verbose)
        # Process
        net = prune(features=adata.var_names, net=net, tmin=kws_decouple['tmin'], verbose=verbose)
        adata, net = _filter(adata=adata, net=net, sfilt=sfilt, verbose=verbose)
        _sign(adata=adata)
        # Run benchmark
        decouple(data=adata, net=net, verbose=verbose, layer='tmp', **kws_decouple)
        df = _eval_scores(
            adata=adata,
            groupby=groupby,
            runby=runby,
            metrics=metrics,
            thr=thr,
            emin=emin,
            verbose=verbose,
            **kwargs
        )
    else:
        m = f'benchmark - running benchmark for multiple networks'
        _log(m, level='info', verbose=verbose)
        df = []
        for n_name in net:
            nnet = net[n_name]
            nnet = prune(features=adata.var_names, net=nnet, tmin=kws_decouple['tmin'], verbose=verbose)
            ndata, nnet = _filter(adata=adata, net=nnet, sfilt=sfilt, verbose=verbose)
            _sign(adata=ndata)
            # Run benchmark
            decouple(data=ndata, net=nnet, verbose=verbose, layer='tmp', **kws_decouple)
            tmp = _eval_scores(
                adata=ndata,
                groupby=groupby,
                runby=runby,
                metrics=metrics,
                thr=thr,
                emin=emin,
                verbose=verbose,
                **kwargs
            )
            tmp.insert(0, 'net', n_name)
            df.append(tmp)
        df = pd.concat(df)
    return df
