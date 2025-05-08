from typing import Tuple

import pandas as pd
import numpy as np
import scipy.sparse as sps
from anndata import AnnData

from decoupler._log import _log
from decoupler.pp.net import prune


def _validate_groupby(
    obs: pd.DataFrame,
    groupby: str | list | None,
    runby: str,
) -> None | list:
    assert isinstance(groupby, (str, list)) or groupby is None, \
    'groupby must be str, list or None'
    assert isinstance(runby, str) and runby in ['expr', 'source'], \
    'runby must be str and either expr or source'
    if groupby is not None:
        if type(groupby) is str:
            groupby = [groupby]
        for grp_i in groupby:
            if type(grp_i) is str:
                grp_i = [grp_i]
            # For each group inside each groupby
            for grp_j in grp_i:
                assert not ('source' == grp_j and runby == 'source'), \
                f'source cannot be in groupby if runby="source"'
                # Assert that columns exist in obs
                assert grp_j in obs.columns, \
                f'Column name "{grp_j}" must be in adata.obs.columns'
                # Assert that column doesn't contain "|"
                assert '|' not in grp_j, \
                'Column names must not contain the \"|\" character'
        return groupby


def _validate_obs(
    obs: pd.DataFrame,
) -> None:
    assert 'source' in obs.columns, \
    'source must be in adata.obs.columns'
    assert 'type_p' in obs.columns, \
    'type_p must be in adata.obs.columns'
    assert pd.api.types.is_numeric_dtype(obs['type_p']), \
    'type_p must contain numeric values'
    assert np.isin(obs['type_p'].sort_values().unique(), np.array([-1, 1])).all(), \
    'type_p must be -1 or +1'


def _filter(
    adata: AnnData,
    net: pd.DataFrame,
    sfilt: bool,
    verbose: bool,
) -> Tuple[AnnData, pd.DataFrame]:
    # Remove experiments without sources in net
    srcs = net['source'].unique()
    prts = set()
    msk_exp = np.zeros(adata.obs_names.size, dtype=np.bool_)
    for i, src in enumerate(adata.obs['source']):
        if isinstance(src, list):
            prts.update(src)
            if np.isin(src, srcs).any():
                msk_exp[i] = True
        elif isinstance(src, str):
            prts.add(src)
            if src in srcs:
                msk_exp[i] = True
    r_exp = int((~msk_exp).sum())
    m = f'benchmark - removing {r_exp} experiments without sources in net'
    _log(m, level='info', verbose=verbose)
    adata = adata[msk_exp, :].copy()
    # Remove sources without experiments in obs
    if sfilt:
        msk_src = np.isin(net['source'], prts)
        net = net.loc[msk, :]
        rsrc = net.loc[~msk, 'source'].values.size
        m = f'benchmark - removing {rsrc} sources without experiments in obs'
        _log(m, level='info', verbose=verbose)
    adata.uns['p_sources'] = prts
    return adata, net


def _sign(
    adata: AnnData,
) -> None:
    v_sign = adata.obs['type_p'].values.reshape(-1, 1)
    if sps.issparse(adata.X):
        adata.layers['tmp'] = adata.X.multiply(v_sign).tocsr()
    else:
        adata.layers['tmp'] = adata.X * v_sign


def _validate_bool(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    assert isinstance(y_true, np.ndarray), 'y_true must be numpy.ndarray'
    assert isinstance(y_score, np.ndarray), 'y_score must be numpy.ndarray'
    unq = np.sort(np.unique(y_true))
    m = 'y_true must contain two binary classes, 0 and 1'
    assert unq.size <= 2, m
    lbl = np.array([0, 1])
    assert np.all(unq == lbl), m
    assert y_true.size == y_score.size, \
    'y_true and y_score must have the same size'
