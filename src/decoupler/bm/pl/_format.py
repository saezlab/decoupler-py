import pandas as pd
import numpy as np


def _format(
    df: pd.DataFrame,
    cols: list,
) -> pd.DataFrame:
    # Validate
    assert isinstance(df, pd.DataFrame), 'df must be pandas.DataFrame'
    assert isinstance(cols, list), 'cols must be list'
    assert 'metric' in df.columns, 'metric must be in df.columns'
    assert 'score' in df.columns, 'score must be in df.columns'
    # Extract
    tmp = df[df['metric'].isin(cols)].copy()
    assert tmp.shape[0] > 0, 'cols must be in df["metric"]'
    # Add small variations so not same number
    rng = np.random.default_rng(seed=0)
    tmp.loc[:, 'score'] = tmp.loc[:, 'score'] + rng.normal(loc=0, scale=2.2e-16, size=tmp.shape[0])
    tmp.loc[:, 'score'] = tmp.loc[:, 'score'].clip(lower=0)
    # Transform
    grp_cols = ['net', 'groupby', 'group', 'source', 'method']
    grp_cols = [c for c in grp_cols if c in df.columns]
    tmp = (
        tmp
        .pivot(index=grp_cols, columns='metric', values='score')
        .reset_index()
    ).dropna(axis=1)
    if np.all(np.isin(['groupby', 'group'], tmp.columns)):
        tmp = (
            tmp
            .pivot(index=['source', 'method'] + cols, columns='groupby', values='group')
            .reset_index()
        )
    # Remove names
    tmp.index.name = None
    tmp.columns.name = None
    return tmp
