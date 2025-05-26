import pandas as pd
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
import seaborn as sns

from decoupler._docs import docs
from decoupler._Plotter import Plotter


@docs.dedent
def order(
    df: pd.DataFrame,
    mode: str = 'line',
    kw_order = dict(),
    **kwargs
) -> None | Figure:
    """
    Plot features along a continuous, ordered process such as pseudotime.

    Parameters
    ----------
    df
        Results of ``decoupler.pp.bin_order``.
    mode
        The type of plot to use, either "line" or "mat".
    kw_order
        Other keyword arguments are passed down to ``seaborn.lineplot`` or ``matplotlib.pyplot.imshow``,
        depending on ``mode`` used.
    %(plot)s
    """
    # Validate
    assert isinstance(df, pd.DataFrame), 'df must be pandas.DataFrame'
    assert isinstance(mode, str) and mode in ['line', 'mat'], \
    'mode must be str and either "line" or "mat"'
    assert isinstance(kw_order, dict), \
    'kw_order must be dict'
    # Process
    ymax = df['value'].max()
    xmin, xmax = df['order'].min(), df['order'].max()
    n_names = df['name'].unique().size
    # Add cbar if added
    has_cbar = False
    if np.isin(['label', 'color'], df.columns).all():
        colors = df[df['name'] == df.loc[0, 'name']]['color']
        colors = [[to_rgb(c) for c in colors]]
        has_cbar = True
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    if mode == 'line':
        if has_cbar:
            bp.ax.imshow(
                colors,
                aspect='auto',
                extent=[xmin, xmax, 1.05 * ymax, 1.2 * ymax],
                transform=bp.ax.transData,
                zorder=2
            )
            bp.ax.axhline(y=1.05 * ymax, c='black', lw=1)
        kw_order = kw_order.copy()
        kw_order.setdefault('palette', 'tab20')
        sns.lineplot(
            data=df,
            x='order',
            y='value',
            hue='name',
            ax=bp.ax,
            **kw_order
        )
        bp.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    elif mode == 'mat':
        mat = (
            df
            .groupby(['name', 'order'], as_index=False)['value'].mean()
            .pivot(index='name', columns='order', values='value')
        )
        img = bp.ax.imshow(mat, extent=[xmin, xmax, 0, n_names], aspect='auto', **kw_order)
        if has_cbar:
            bp.ax.imshow(colors, aspect='auto', extent=[xmin, xmax, n_names, 1.1 * n_names], zorder=2)
            bp.ax.axhline(y=n_names, c='black', lw=1)
            bp.ax.set_ylim(0, 1.1 * n_names)
        bp.fig.colorbar(img, ax=bp.ax, shrink=0.5, label='Mean value', location='top')
        bp.ax.set_yticks(np.arange(n_names) + 0.5)
        bp.ax.set_yticklabels(np.flip(mat.index))
        bp.ax.grid(axis='y', visible=False)
        bp.ax.set_xlabel('order')
    bp.ax.set_xlim(xmin, xmax)
    return bp._return()
