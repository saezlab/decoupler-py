from typing import Tuple, Callable

import pandas as pd
import numpy as np
from anndata import AnnData
import scipy.sparse as sps
import scipy.stats as sts
from tqdm.auto import tqdm

from decoupler._log import _log
from decoupler._datatype import DataType
from decoupler.pp.net import prune, adjmat, idxmat
from decoupler.pp.data import extract


def _return(
    name: str,
    data: DataType,
    es: pd.DataFrame,
    pv: pd.DataFrame,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame] | AnnData | None:
    if isinstance(data, AnnData):
        if data.obs_names.size != es.index.size:
            m = 'Provided AnnData contains empty observations, returning repaired object'
            _log(m, level='warn', verbose=verbose)
            data = data[es.index, :].copy()
            data.obsm[f'score_{name}'] = es
            if pv is not None:
                data.obsm[f'padj_{name}'] = pv
            return data
        else:
            data.obsm[f'score_{name}'] = es
            if pv is not None:
                data.obsm[f'padj_{name}'] = pv
            return None
    else:
        return es, pv


def _run(
    name: str,
    func: Callable,
    adj: bool,
    test: bool,
    data: DataType,
    net: pd.DataFrame,
    tmin: int | float = 5,
    layer: str | None = None,
    raw: bool = False,
    empty: bool = True,
    bsize: int | float = 250_000,
    verbose: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame] | AnnData | None:
    _log(f'{name} - Running {name}', level='info', verbose=verbose)
    # Process data
    mat, obs, var = extract(data, layer=layer, raw=raw, empty=empty, verbose=verbose)
    sparse = sps.issparse(mat)
    # Process net
    net = prune(features=var, net=net, tmin=tmin, verbose=verbose)
    # Handle stat type
    if adj:
        sources, targets, adjm = adjmat(features=var, net=net, verbose=verbose)
        # Handle sparse
        if sparse:
            nbatch = int(np.ceil(obs.size / bsize))
            es, pv = [], []
            for i in tqdm(range(nbatch), disable=not verbose):
                srt, end = i * bsize, i * bsize + bsize
                bmat = mat[srt:end].toarray()
                bes, bpv = func(bmat, adjm, verbose=verbose, **kwargs)
                es.append(bes)
                pv.append(bpv)
            es = np.vstack(es)
            es = pd.DataFrame(es, index=obs, columns=sources)
        else:
            es, pv = func(mat, adjm, verbose=verbose, **kwargs)
            es = pd.DataFrame(es, index=obs, columns=sources)
    else:
        sources, cnct, starts, offsets = idxmat(features=var, net=net, verbose=verbose)
        es, pv = func(mat, cnct, starts, offsets, verbose=verbose, **kwargs)
        es = pd.DataFrame(es, index=obs, columns=sources)
    # Handle pvals and FDR correction
    if test:
        pv = np.vstack(pv)
        pv = pd.DataFrame(pv, index=obs, columns=sources)
        if name != 'mlm':
            _log(f'{name} - adjusting p-values by FDR', level='info', verbose=verbose)
            pv.loc[:, :] = sts.false_discovery_control(pv.values, axis=1, method='bh')
    else:
        pv = None
    _log(f'{name} - done', level='info', verbose=verbose)
    return _return(name, data, es, pv, verbose=verbose)
