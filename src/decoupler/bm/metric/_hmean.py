import pandas as pd
import numpy as np

from decoupler._docs import docs
from decoupler.bm.pl._format import _format


def _hmean(
    x: float | int,
    y: float | int,
    beta: float | int = 1,
) -> float:
    assert isinstance(beta, (int, float)) and 0 < beta, \
    'beta must be numeric and > 0'
    h = np.zeros(len(x))
    msk = (x != 0.) & (y != 0.)
    h[msk] = (1 + beta**2) * (x[msk] * y[msk]) / ((x[msk] * beta**2) + y[msk])
    return h


@docs.dedent
def hmean(
    df: pd.DataFrame,
    metrics: str | list = ['auc', 'fscore', 'qrank'],
    beta: int | float = 0.5,
) -> pd.DataFrame:
    """
    Computes the harmonic mean between two metric statistics.

    Parameters
    ----------
    %(df)s
    metrics
        Metrics which to compute the harmonic mean between their own statistics.
    beta
        Controls the balance between statistics, where beta > 1 favors the first one (for example recall),
        beta < 1 the other one (for example precision), and beta = 1 gives equal weight to both.

    Returns
    -------
    Dataframe containing the harmonic mean per metric.
    """
    # Validate
    assert isinstance(df, pd.DataFrame), 'df must be pandas.DataFrame'
    assert isinstance(metrics, (str, list)), 'metrics must be str or list'
    if isinstance(metrics, str):
        metrics = [metrics]
    # Run
    d_metrics = {
        'auc': {
            'name': 'H(auroc, auprc)',
            'cols': ['auprc', 'auroc'],
        },
        'fscore': {
            'name': 'F-score',
            'cols': ['precision', 'recall'],
        },
        'qrank': {
            'name': 'H(1-qrank, -log10(pval))',
            'cols': ['-log10(pval)', '1-qrank'],
        },
    }
    hdf = []
    h_cols = []
    for i, metric in enumerate(metrics):
        # Format
        cols = d_metrics[metric]['cols']
        tmp = _format(df=df, cols=cols)
        # Compute harmonic mean
        name = d_metrics[metric]['name']
        tmp[name] = _hmean(tmp[cols[0]], tmp[cols[1]], beta=beta)
        if i == 0:
            hdf.append(tmp)
        else:
            hdf.append(tmp[cols + [name]])
        h_cols.append(name)
    hdf = pd.concat(hdf, axis=1)
    # Mean qrank (final score)
    hdf['score'] = hdf[h_cols].mean(axis=1, numeric_only=True)
    hdf['score'] = (hdf['score'] - hdf['score'].min()) / (hdf['score'].max() - hdf['score'].min())
    return hdf
        