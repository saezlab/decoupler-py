import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
import marsilea as ma
import marsilea.plotter as mp

from decoupler._docs import docs
from decoupler._Plotter import Plotter


def _mcolor(
    cmap: str,
) -> str:
    assert isinstance(cmap, str), 'cmap must be str'
    cmap = plt.get_cmap(cmap)
    mid_color_rgb = cmap(0.5)
    mid_color_hex = to_hex(mid_color_rgb)
    return mid_color_hex


def _palette(
    labels: list,
    cmap: str = 'tab20'
) -> dict:
    assert isinstance(cmap, str), 'cmap must be str'
    rnks = np.arange(len(labels))
    if cmap in plt.colormaps():
        cmap = plt.get_cmap(cmap)
        sidx = np.argsort(np.argsort(labels))
        colors = np.array([to_hex(cmap(i)) for i in rnks])
        palette = dict(zip(rnks, colors[sidx]))
    else:
        palette = dict(zip(rnks, [cmap for _ in range(rnks.size)]))
    return palette


@docs.dedent
def summary(
    df: pd.DataFrame,
    y: str,
    metrics: list = ['auc', 'fscore', 'qrank'],
    cmap_y: str = 'tab20',
    cmap_auc: str = 'Greens',
    cmap_fscore: str = 'Blues',
    cmap_qrank: str = 'Reds',
    **kwargs
) -> None | Figure:
    """
    Summarizes metrics into a final score by computing the quantile-normalized rank across them.

    Parameters
    ----------
    %(df)s
    %(y)s
    metrics
        Which metrics to include.
    cmap_y
        Color map for the y grouping.
    cmap_auc
        Color map for the auc metric.
    cmap_fscore
        Color map for the fscore metric.
    cmap_qrank
        Color map for the qrank metric.
    %(plot)s
    """
    # Validate
    assert isinstance(df, pd.DataFrame), 'df must be pandas.DataFrame'
    assert isinstance(y, str) and y in df.columns, 'y must be str and in df.columns'
    assert isinstance(metrics, (str, list)), 'metrics must be str or list'
    if isinstance(metrics, str):
        metrics = [metrics]
    assert set(metrics).issubset({'auc', 'fscore', 'qrank'})
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # Fix
    df = df.copy().sort_values('score', ascending=False).set_index(y)
    # Scale
    cols = list(df.select_dtypes(include='number').columns)
    df[cols] = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())
    # Instance
    kwargs['ax'] = None
    bp = Plotter(**kwargs)
    bp.fig.delaxes(bp.ax)
    plt.close(bp.fig)
    # Plot
    lst_h = []
    pad = 0.05
    if 'auc' in metrics:
        cnames = ['auroc', 'auprc', 'H(auroc, auprc)']
        h = ma.Heatmap(df[cnames], cmap=cmap_auc, name="h_auc", width=1, height=3, lw=.1, cbar_kws=dict(orientation="horizontal", title="Scaled auc"))
        h.group_cols(['auc', 'auc', 'auc'], order=['auc'])
        h.add_top(
            ma.plotter.Chunk(['auc'], fill_colors=[_mcolor(cmap_auc)], align='center')
        )
        h.add_bottom(mp.Labels(cnames[:2] + ['h mean'], ), pad=pad)
        lst_h.append(h)
    if 'fscore' in metrics:
        cnames = ['precision', 'recall', 'F-score']
        h = ma.Heatmap(df[cnames], cmap=cmap_fscore, name="h_fsc", width=1, height=2, lw=.1, cbar_kws=dict(orientation="horizontal", title="Scaled fscore"))
        h.group_cols(['fscore', 'fscore', 'fscore'], order=['fscore'])
        h.add_top(
            ma.plotter.Chunk(['fscore'], fill_colors=[_mcolor(cmap_fscore)], align='center')
        )
        h.add_bottom(mp.Labels(cnames[:2] + ['h mean'], ), pad=pad)
        lst_h.append(h)
    if 'qrank' in metrics:
        cnames = ['-log10(pval)', '1-qrank', 'H(1-qrank, -log10(pval))']
        h = ma.Heatmap(df[cnames], cmap=cmap_qrank, name="h_rnk", width=1, height=1, lw=.1, cbar_kws=dict(orientation="horizontal", title="Scaled qrank"))
        h.group_cols(['qrank', 'qrank', 'qrank'], order=['qrank'])
        h.add_top(
            ma.plotter.Chunk(['qrank'], fill_colors=[_mcolor(cmap_qrank)], align='center')
        )
        h.add_bottom(mp.Labels(cnames[:2] + ['h mean'], ), pad=pad)
        lst_h.append(h)
    # Build canvas
    c = lst_h[0]
    # Add bar to the left
    bar = mp.Bar(df['score'].T, label='Score', color='gray', palette=_palette(labels=df.index, cmap=cmap_y))
    c.add_left(bar)
    c.add_left(mp.Labels(df.index, ), pad=pad)
    c.add_title(left=y, pad=pad)
    for h in lst_h[1:]:
        c += h
    c.add_legends(side='right')
    c.render()
    if bp.return_fig or bp.save is not None:
        plt.close()
    i = [i for i in range(len(c.figure.axes)) if c.figure.axes[i].get_xlabel() == 'Score'][0]
    c.figure.axes[i].invert_xaxis()
    bp.fig = c.figure
    bp.fig.set_figwidth(bp.figsize[0])
    bp.fig.set_figheight(bp.figsize[1])
    bp.fig.set_dpi(bp.dpi)
    return bp._return()
