import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import adjustText as at

from decoupler._docs import docs
from decoupler._Plotter import Plotter
from decoupler.pp.net import _validate_net


@docs.dedent
def volcano(
    data: pd.DataFrame,
    x: str,
    y: str,
    net: pd.DataFrame | None = None,
    name: str | None = None,
    top: int = 5,
    thr_stat: float = 0.5,
    thr_sign: float = 0.05,
    max_stat: float | None = None,
    max_sign: float | None = None,
    color_pos: str = '#D62728',
    color_neg: str = '#1F77B4',
    color_null: str = 'gray',
    **kwargs
) -> None | Figure:
    """
    Plot logFC and p-values from a long formated data-frame.

    Parameters
    ----------
    %(data_plot)s
    x
        Column name of ``data`` storing the change statitsics.
    y
        Column name of ``data`` storing the associated p-values.
    %(net)s
    name
        Name of the source to subset ``net``.
    top
        Number of top differentially abundant features to show.
    thr_stat
        Significance threshold for change statitsics.
    thr_sign
        Significance threshold for p-values.
    max_stat
        Limit of change statitsics to plot in absolute value.
    max_sign
        Limit of p-values to plot in ``-log10``.
    color_pos
        Color to plot significant positive features.
    color_neg
        Color to plot significant negative features.
    color_null
        Color to plot rest of the genes.
    %(plot)s
    """
    # Validate inputs
    m = f'data must be a pd.DataFrame containing the columns {x} and {y}'
    assert isinstance(data, pd.DataFrame), m
    assert {x, y}.issubset(data.columns), m
    assert (net is None) == (name is None), \
    'net and name must be both defined or both None'
    assert isinstance(top, int) and top > 0, 'top must be int and > 0'
    assert isinstance(thr_stat, (int, float)) and thr_stat > 0, \
    'thr_stat must be numeric and > 0'
    assert isinstance(thr_sign, (int, float)) and thr_sign > 0, \
    'thr_sign must be numeric and > 0'
    if max_stat is None:
        max_stat = np.inf
    if max_sign is None:
        max_sign = np.inf
    assert isinstance(max_stat, (int, float)) and max_stat > 0, \
    'max_stat must be None, or numeric and > 0'
    assert isinstance(max_sign, (int, float)) and max_sign > 0, \
    'max_sign must be None, or numeric and > 0'
    assert isinstance(color_pos, str), 'color_pos must be str'
    assert isinstance(color_neg, str), 'color_neg must be str'
    assert isinstance(color_null, str), 'color_null must be str'
    # Instance
    bp = Plotter(**kwargs)
    # Transform thr_sign
    thr_sign = -np.log10(thr_sign)
    # Extract df
    df = data.copy()
    df['stat'] = df[x]
    non_zero_min = df[y][df[y] != 0].min()
    df['pval'] = -np.log10(df[y].clip(lower=non_zero_min, upper=1))
    # Filter by net shared targets
    if net is not None:
        vnet = _validate_net(net)
        snet = vnet[vnet['source'] == name]
        assert snet.shape[0] > 0, f'name={name} must be in net["source"]'
        df = df[df.index.isin(snet['target'])]
    # Filter by limits
    msk_stat = np.abs(df['stat']) < np.abs(max_stat)
    msk_sign = df['pval'] < np.abs(max_sign)
    df = df.loc[msk_stat & msk_sign]
    # Define color by up or down regulation and significance
    df['weight'] = color_null
    up_msk = (df['stat'] >= thr_stat) & (df['pval'] >= thr_sign)
    dw_msk = (df['stat'] <= -thr_stat) & (df['pval'] >= thr_sign)
    df.loc[up_msk, 'weight'] = color_pos
    df.loc[dw_msk, 'weight'] = color_neg
    # Plot
    df.plot.scatter(x='stat', y='pval', c='weight', sharex=False, ax=bp.ax)
    bp.ax.set_axisbelow(True)
    # Draw thr lines
    bp.ax.axvline(x=thr_stat, linestyle='--', color="black")
    bp.ax.axvline(x=-thr_stat, linestyle='--', color="black")
    bp.ax.axhline(y=thr_sign, linestyle='--', color="black")
    # Add labels
    bp.ax.set_title(name)
    bp.ax.set_xlabel(x)
    bp.ax.set_ylabel(rf'$-\log_{{10}}({y})$')
    # Show top sign features
    signs = df[up_msk | dw_msk].sort_values('pval', ascending=False)
    signs = signs.iloc[:top]
    texts = []
    for x, y, s in zip(signs['stat'], signs['pval'], signs.index):
        texts.append(bp.ax.text(x, y, s))
    if len(texts) > 0:
        at.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ax=bp.ax)
    return bp._return()
