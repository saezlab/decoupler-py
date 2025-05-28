import warnings

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import scipy.stats as sts
import scipy.sparse as sps
from anndata import AnnData
import dcor

from decoupler._odeps import dcor, _check_import
from decoupler._docs import docs
from decoupler.pp.data import extract



@docs.dedent
def rankby_order(
    adata: AnnData,
    order: str,
    stat: str = 'dcor',
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Rank features along a continuous, ordered process such as pseudotime.

    Parameters
    ----------
    %(adata)s
    %(order)s
    stat
        Which statistic to compute.
        Must be one of these:
        
        - ``dcor`` (distance correlation from ``dcor.independence.distance_correlation_t_test``)
        - ``pearsonr`` (Pearson's R from ``scipy.stats.pearsonr``)
        - ``spearmanr`` (Spearman's R from ``scipy.stats.spearmanr``)
        - ``kendalltau`` (Kendall's Tau from ``scipy.stats.kendalltau``)

    %(verbose)s
    kwargs
        Key arguments passed to the selected ``stat`` function.

    Returns
    -------
    DataFrame with features associated with the ordering variable.
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be anndata.AnnData'
    assert isinstance(order, str) and order in adata.obs.columns, 'order must be str and in adata.obs.columns'
    stats = {'dcor', 'pearsonr', 'spearmanr', 'kendalltau'}
    assert (isinstance(stat, str) and stat in stats) or callable(stat), \
    f'stat must be str and one of these {stats}, or a function that returns statistic and pvalue'
    # Get vars and ordinal variable
    X = adata.X
    if sps.issparse(X):
        X = X.toarray()
    X = X.astype(float)
    y = adata.obs[order].values.astype(float)
    # Init
    df = pd.DataFrame()
    df['name'] = adata.var_names
    # Fit
    if stat == 'dcor':
        _check_import(dcor)
        f = dcor.independence.distance_correlation_t_test
    elif stat == 'pearsonr':
        f = sts.pearsonr
    elif stat == 'spearmanr':
        f = sts.spearmanr
    elif stat == 'kendalltau':
        f = sts.kendalltau
    else:
        f = stat
    ss = []
    ps = []
    for i in tqdm(range(X.shape[1]), disable=not verbose):
        x = X[:, i]
        if not np.all(x == x[0]):
            res = f(x, y)
            s = res.statistic
            p = res.pvalue
        else:
            s = 0
            p = 1
        ss.append(s)
        ps.append(p)
    df['stat'] = ss
    df['pval'] = ps
    df['padj'] = sts.false_discovery_control(df['pval'])
    df['abs_stat'] = df['stat'].abs()
    df = df.sort_values(['padj', 'pval', 'abs_stat'], ascending=[True, True, False]).reset_index(drop=True)
    df = df.drop(columns='abs_stat')
    return df
