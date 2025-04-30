import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import adjustText as at

from decoupler._Plotter import Plotter
from decoupler.pp.net import _validate_net


def source_targets(
    data: pd.DataFrame,
    x: str,
    y: str,
    net: pd.DataFrame,
    name: str,
    top: int = 5,
    thr_x: float = 0.,
    thr_y: float = 0.,
    max_x: float | None = None,
    max_y: float | None = None,
    color_pos: str = '#D62728',
    color_neg: str = '#1F77B4',
    **kwargs,
) -> None | Figure:
    """
    """
    # Validate inputs
    m = f'data must be a pd.DataFrame containing the columns {x} and {y}'
    assert isinstance(data, pd.DataFrame), m
    assert {x, y}.issubset(data.columns.union(net.columns)), m
    assert isinstance(net, pd.DataFrame), \
    f'net must be a pd.DataFrame containing the columns {x} and {y}'
    assert isinstance(name, str), 'name must be a str'
    assert isinstance(top, int) and top > 0, 'top must be int and > 0'
    assert isinstance(thr_x, (int, float)), 'thr_x must be numeric'
    assert isinstance(thr_y, (int, float)), 'thr_y must be numeric'
    if max_x is None:
        max_x = np.inf
    if max_y is None:
        max_y = np.inf
    assert isinstance(max_x, (int, float)) and max_x > 0, \
    'max_x must be None, or numeric and > 0'
    assert isinstance(max_y, (int, float)) and max_y > 0, \
    'max_y must be None, or numeric and > 0'
    assert isinstance(color_pos, str), 'color_pos must be str'
    assert isinstance(color_neg, str), 'color_neg must be str'
    # Instance
    bp = Plotter(**kwargs)
    # Extract df
    df = data.copy().reset_index(names='target')
    # Filter by net shared targets
    vnet = _validate_net(net)
    snet = vnet[vnet['source'] == name]
    df = pd.merge(df, snet, on=['target'], how='inner').set_index('target')
    # Filter by limits
    msk_x = np.abs(df[x]) < np.abs(max_x)
    msk_y = np.abs(df[y]) < np.abs(max_y)
    df = df.loc[msk_x & msk_y]
    # Define +/- color
    pos = ((df[x] >= 0) & (df[y] >= 0)) | ((df[x] < 0) & (df[y] < 0))
    df['color'] = color_neg
    df.loc[pos, 'color'] = color_pos
    # Plot
    df.plot.scatter(x=x, y=y, c='color', ax=bp.ax)
    # Draw thr lines
    bp.ax.axvline(x=thr_x, linestyle='--', color="black")
    bp.ax.axhline(y=thr_y, linestyle='--', color="black")
    # Add labels
    bp.ax.set_title(name)
    bp.ax.set_xlabel(x)
    bp.ax.set_ylabel(y)
    # Show top features
    df['order'] = df[x].abs() * df[y].abs()
    signs = df.sort_values('order', ascending=False)
    signs = signs.iloc[:top]
    texts = []
    for x, y, s in zip(signs[x], signs[y], signs.index):
        texts.append(bp.ax.text(x, y, s))
    if len(texts) > 0:
        at.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ax=bp.ax)
    return bp._return()
