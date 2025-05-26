import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
import seaborn as sns
from anndata import AnnData

from decoupler._docs import docs
from decoupler._Plotter import Plotter
from decoupler.pp.anndata import get_obsm, bin_order
from decoupler.pp.net import prune


@docs.dedent
def order_targets(
    adata: AnnData,
    net: pd.DataFrame,
    source: str,
    order: str,
    score: str = 'score_ulm',
    label: str | None = None,
    nbins: int = 100,
    top: int = 10,
    pos_cmap: str = 'Reds',
    neg_cmap: str = 'Blues',
    color_score: str = '#88c544',
    vmin: int | float = None,
    vmax: int | float = None,
    **kwargs
) -> None | Figure:
    """
    Plot a source score, together with its targets readouts, along a continuous ordered process such as pseudotime.

    Parameters
    ----------
    %(adata)s
    %(net)s
    source
        Which source from ``net`` to show.
    %(order)s
    score
        ``adata.obsm`` key where enrichment scores are stored.
    %(label)s
    %(nbins)s
    top
        How many targets to show ranked by their standard deviation along the ordered process.
    pos_cmap
        Colormap for targets with positive weights.
    net_cmap
        Colormap for targets with negative weights.
    color_score
        Color used to plot the enrichment score.
    vmin
        Minimum value to color.
    vmax
        Minimum value to color.
    %(plot)s
    """
    # Validate
    assert isinstance(source, str), 'source must be str'
    assert isinstance(score, str), 'score must be str'
    assert isinstance(top, int) and top > 0, 'top must be int and > 0'
    # Filter net to adata
    snet = prune(features=adata.var_names, net=net, tmin=0)
    snet = snet[snet['source'] == source]
    assert snet.shape[0] > 0, f'{source} must be in net["source"]'
    # Get scoring method
    score = get_obsm(adata=adata, key=score)
    # Split by sign
    pos_names = snet[snet['weight'] > 0]['target'].values.astype('U')
    neg_names = snet[snet['weight'] < 0]['target'].values.astype('U')
    # Bin by order
    df_ftr = bin_order(
        adata=adata,
        order=order,
        names=snet['target'].to_list(),
        label=label,
        nbins=nbins,
    )
    df_scr = bin_order(
        adata=score,
        order=order,
        names=source,
        label=label,
        nbins=nbins,
    )
    # Check if labels are included
    has_cbar = False
    if np.isin(['label', 'color'], df_ftr.columns).all():
        colors = df_ftr[df_ftr['name'] == df_ftr.loc[0, 'name']]['color']
        colors = [[to_rgb(c) for c in colors]]
        has_cbar = True
    # Get mat of target values
    mat = (
        df_ftr
        .groupby(['name', 'order'], as_index=False)['value'].mean()
        .pivot(index='name', columns='order', values='value')
    )
    if vmax is None:
        vmax = mat.values.max()
    if vmin is None:
        vmin = mat.values.min()
    # Sort by magnitude
    sorted_names = mat.std(1, ddof=1).sort_values().tail(top).index
    pos_names = sorted_names.intersection(pos_names)[::-1]
    neg_names = sorted_names.intersection(neg_names)
    n_names = pos_names.size + neg_names.size
    # Instance
    kwargs = kwargs.copy()
    kwargs.setdefault('figsize', (6, np.max([sorted_names.size / 3, 3])))
    kwargs['ax'] = None
    bp = Plotter(**kwargs)
    bp.fig.delaxes(bp.ax)
    plt.close(bp.fig)
    # Plot
    bp.fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[0.2, 0.8],
        figsize=bp.figsize,
        dpi=bp.dpi,
    )
    # Source score
    ax = axes[0]
    sns.lineplot(
        data=df_scr,
        x='order',
        y='value',
        ax=ax,
        color=color_score,
        lw=2
    )
    ax.set_ylabel(f'{source}\nscore')
    # Target values
    ax = axes[1]
    ax.set_xlabel('order')
    ax.grid(axis='y', visible=False)
    ax.set_ylabel('Targets')
    yticklabels = []
    # Find minmax
    omin, omax = df_ftr['order'].min(), df_ftr['order'].max()
    # Add neg targets
    if neg_names.size > 0:
        img = ax.imshow(mat.loc[neg_names], extent=[omin, omax, 0, neg_names.size], aspect='auto', cmap=neg_cmap, vmin=vmin, vmax=vmax)
        yticklabels.extend(list(neg_names)[::-1])
        cbar_mappable = ScalarMappable(cmap=neg_cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        pos = ax.get_position().bounds
        cax = bp.fig.add_axes([0.97, pos[1], 0.05, (pos[3] / 2) - .025])
        cax.grid(axis='y', visible=False)
        bp.fig.colorbar(cbar_mappable, cax=cax, aspect=5, shrink=0.5, label='- target\nvalues', location='right')
    ax.axhline(y=neg_names.size, c='black', lw=1)
    # Add pos targets
    if pos_names.size > 0:
        img = ax.imshow(mat.loc[pos_names], extent=[omin, omax, neg_names.size, neg_names.size + pos_names.size], aspect='auto', cmap='Reds', vmin=vmin, vmax=vmax)
        yticklabels.extend(list(pos_names)[::-1])
        cbar_mappable = ScalarMappable(cmap=pos_cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        pos = ax.get_position().bounds
        cax = bp.fig.add_axes([0.97, pos[1] + (pos[3] / 2) + .025, 0.05, (pos[3] / 2) - .025])
        cax.grid(axis='y', visible=False)
        bp.fig.colorbar(cbar_mappable, cax=cax, aspect=5, shrink=0.5, label='+ target\nvalues', location='right')
    # Plot labels
    ax.set_ylim(0, n_names)
    if has_cbar:
        ax.imshow(colors, aspect='auto', extent=[omin, omax, n_names, 1.1 * n_names], zorder=2)
        ax.axhline(y=n_names, c='black', lw=1)
        ax.set_ylim(0, 1.1 * n_names)
    # Format plot
    ax.set_yticks(np.arange(n_names) + 0.5)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim(omin, omax)
    bp.fig.subplots_adjust(hspace=0)
    return bp._return()
