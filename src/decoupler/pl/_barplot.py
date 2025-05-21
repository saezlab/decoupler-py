from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from decoupler._docs import docs
from decoupler._Plotter import Plotter


def _set_limits(
    vmin: int | float,
    vcenter: int | float,
    vmax: int | float,
    values: np.ndarray
) -> Tuple[float, float, float]:
    assert np.isfinite(values).all(), 'values in data mut be finite'
    assert isinstance(vmin, (int, float)) or vmin is None, 'vmin must be numerical or None'
    assert isinstance(vcenter, (int, float)) or vcenter is None, 'vcenter must be numerical or None'
    assert isinstance(vmax, (int, float)) or vmax is None, 'vmax must be numerical or None'
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    if vcenter is None:
        vcenter = values.mean()
    if vmin >= vcenter:
        vmin = -vmax
    if vcenter >= vmax:
        vmax = -vmin
    return vmin, vcenter, vmax


@docs.dedent
def barplot(
    data: pd.DataFrame,
    name: str,
    top: int = 25,
    vertical: bool = False,
    cmap: str = 'RdBu_r',
    vmin: float | None = None,
    vcenter: float | None = 0,
    vmax: float | None = None,
    **kwargs,
) -> None | Figure:
    """
    Plot barplots showing top scores.

    Parameters
    ----------
    data
        DataFrame in wide format containing enrichment scores (contrasts, sources).
    name
        Name of the contrast (row) to plot.
    %(top)s
    vertical
        Whether to plot the bars verticaly or horizontaly.
    %(cmap)s
    %(vmin)s
    %(vcenter)s
    %(vmax)s
    %(plot)s
    """
    # Validate
    assert isinstance(data, pd.DataFrame), 'data must be pandas.DataFrame'
    assert isinstance(name, str) and name in data.index, \
    'name must be str and in data.index'
    assert isinstance(top, int) and top > 0, 'top must be int and > 0'
    assert isinstance(vertical, bool), 'vertical must be bool'
    # Process df
    df = data.loc[[name]]
    df.index.name = None
    df.columns.name = None
    df = df.melt(var_name='source', value_name='score')
    df['abs_score'] = df['score'].abs()
    df = df.sort_values('abs_score', ascending=False)
    df = df.head(top).sort_values('score', ascending=False)
    if not vertical:
        x, y = 'score', 'source'
    else:
        x, y = 'source', 'score'
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    sns.barplot(data=df, x=x, y=y, ax=bp.ax)
    if not vertical:
        sizes = np.array([bar.get_width() for bar in bp.ax.containers[0]])
        bp.ax.set_xlabel('Score')
        bp.ax.set_ylabel('')
    else:
        sizes = np.array([bar.get_height() for bar in bp.ax.containers[0]])
        bp.ax.tick_params(axis='x', rotation=90)
        bp.ax.set_ylabel('Score')
        bp.ax.set_xlabel('')
        bp.ax.invert_xaxis()
    # Compute color limits
    vmin, vcenter, vmax = _set_limits(vmin, vcenter, vmax, df['score'])
    # Rescale cmap
    divnorm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap_f = plt.get_cmap(cmap)
    div_colors = cmap_f(divnorm(sizes))
    for bar, color in zip(bp.ax.containers[0], div_colors):
        bar.set_facecolor(color)
    # Add legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=divnorm)
    sm.set_array([])
    bp.fig.colorbar(sm, ax=bp.ax, shrink=0.5)
    return bp._return()
