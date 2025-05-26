import warnings

import pandas as pd
import scipy.stats as sts
import scipy.sparse as sps
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
    from xgboost import XGBRegressor
from anndata import AnnData

from decoupler._docs import docs
from decoupler.pp.data import extract


@docs.dedent
def rankby_order(
    adata: AnnData,
    order: str,
    thr_padj: int | float = 0.05,
    seed: int = 42,
    **kwargs
) -> pd.DataFrame:
    """
    Rank features along a continuous, ordered process such as pseudotime.

    Parameters
    ----------
    %(adata)s
    %(order)s
    thr_padj
        Threshold used to assign significance after FDR correction.
    %(seed)s

    Returns
    -------
    DataFrame with features associated with the ordering variable. For each feature the following statistics are reported:
    - importance of the ``XGBRegressor``
    - Pearson correlation coefficient
    - the sign of the association, 0 if the correltation is non-significant, and +1 or -1 depending on the correlation sign.
    """
    # Get vars and ordinal variable
    X = adata.X
    if sps.issparse(X):
        X = X.toarray()
    y = adata.obs[order].values
    # Fit
    reg = XGBRegressor(random_state=seed, **kwargs).fit(X, y)
    df = pd.DataFrame()
    df['name'] = adata.var_names
    df['impr'] = reg.feature_importances_
    df['corr'], df['pval'] = sts.pearsonr(X, y.reshape(-1, 1), axis=0)
    df['padj'] = sts.false_discovery_control(df['pval'])
    df = df.sort_values('impr', ascending=False).reset_index(drop=True)
    # Find direction of change
    sign = []
    for corr, padj in zip(df['corr'], df['padj']):
        if padj < thr_padj:
            if corr > 0:
                s = 1
            else:
                s = -1
        else:
            s = 0
        sign.append(s)
    df['sign'] = sign
    return df
