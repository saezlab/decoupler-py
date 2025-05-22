from typing import Tuple

import pandas as pd
import scipy.stats as sts
from anndata import AnnData

from decoupler._docs import docs


def _input_rank_obsm(
    adata: AnnData,
    key: str,
) -> Tuple[pd.DataFrame, list, list]:
    # Validate
    assert isinstance(adata, AnnData), 'adata must be anndata.AnnData'
    assert key in adata.obsm, f'key={key} must be in adata.obsm'
    # Process
    name_col = (
        key
        .replace('X_', '')
        .replace('pca', 'PC')
        .replace('mofa', 'Factor')
        .replace('umap', 'UMAP')
    )
    df = adata.obsm[key]
    if isinstance(df, pd.DataFrame):
        y_vars = df.std(ddof=1, axis=0).sort_values(ascending=False).index
        df = df.loc[:, y_vars].values
    else:
        ncol = df.shape[1]
        digits = len(str(ncol))
        y_vars = [f"{name_col}{str(i).zfill(digits)}" for i in range(1, ncol + 1)]
    df = pd.DataFrame(
        data=df,
        index=adata.obs_names,
        columns=y_vars
    )
    x_vars = adata.obs.columns
    # Merge
    df = pd.merge(df, adata.obs, left_index=True, right_index=True)
    return df, x_vars, y_vars


@docs.dedent
def rankby_obsm(
    adata: AnnData,
    key: str,
    uns_key: str | None = 'rank_obsm',
) -> None | pd.DataFrame:
    """
    Ranks features in ``adata.obsm`` by the significance of their association with metadata in ``adata.obs``.

    For categorical variables it uses ANOVA, for continous Spearman's correlation.

    The obtained p-values are corrected by Benjamini-Hochberg.

    Parameters
    ----------
    %(adata)s
    %(key)s
    uns_key
        ``adata.uns`` key to store the results.

    Returns
    -------
    If ``uns_key=False``, a pandas.DataFrame with the resulting statistics.
    """
    assert isinstance(uns_key, str) or uns_key is None, \
    'uns_key must be str or None'
    # Extract
    df, x_vars, y_vars = _input_rank_obsm(adata=adata, key=key)
    # Test
    res = []
    for x_var in x_vars:
        for y_var in y_vars:
            if pd.api.types.is_numeric_dtype(df[x_var]):
                # Correlation
                x = df[x_var].values.ravel()
                y = df[y_var].values.ravel()
                stat, pval = sts.spearmanr(x, y)
            else:
                # ANOVA
                x = [group[y_var].dropna().values for _, group in df.groupby(x_var, observed=True)]
                # At least n=2 per group else skip
                if all(len(g) >= 2 for g in x):
                    stat, pval = sts.f_oneway(*x)
                else:
                    stat, pval = None, 1.
            row = [y_var, x_var, stat, pval]
            res.append(row)
    res = pd.DataFrame(res, columns=['obsm', 'obs', 'stat', 'pval'])
    res['padj'] = sts.false_discovery_control(res['pval'])
    # Rank
    res = res.sort_values('padj').reset_index(drop=True)
    # Add obsm key
    res.key = key
    # Save or return
    if uns_key:
        adata.uns[uns_key] = res
    else:
        return res
