from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from anndata import AnnData
import marsilea as ma
import marsilea.plotter as mp

from decoupler._docs import docs
from decoupler._Plotter import Plotter


def _input(
    adata: AnnData,
    uns_key: str,
    names: str | list | None = None,
    nvar: int | float | list | None = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert isinstance(adata, AnnData), 'adata must be adata.AnnData'
    assert isinstance(uns_key, str) and uns_key in adata.uns, \
    'uns_key must be str and in adata.uns'
    assert isinstance(names, (str, list)) or names is None, \
    'names must be str, list or None'
    assert isinstance(nvar, (int, float, list)) or nvar is None, \
    'nvar must be numeric, list or None'
    # Filter stats by obs names
    stats = adata.uns[uns_key]
    obsm_key = stats.key
    stats = stats.sort_values('obsm')
    var_names = stats['obsm'].unique()
    if isinstance(names, str):
        names = [names]
    if names:
        stats = stats[stats['obs'].isin(names)]
    # Filter stats by obsm nvar
    obsm = adata.obsm[obsm_key]
    if isinstance(nvar, (int, float)):
        stats = stats.groupby('obs', as_index=False).head(int(nvar))
        var_names = var_names[:nvar]
        obsm = obsm[:, :nvar]
    elif isinstance(nvar, list):
        nvar = sorted(nvar)
        stats = stats[stats['obsm'].isin(nvar)]
        idx = np.searchsorted(var_names, nvar)
        obsm = obsm[:, idx]
        var_names = nvar
    # Extract obsm
    obsm = pd.DataFrame(obsm, columns=var_names)
    # Transform stats
    min_p = stats[stats['padj'] > 0]['padj'].min()
    stats.loc[stats['padj'] == 0, 'padj'] = min_p
    stats['padj'] = -np.log10(stats['padj'])
    stats = stats.pivot(index='obs', columns='obsm', values='padj')
    stats.index.name = None
    stats.columns.name = None
    return obsm, stats


@docs.dedent
def rank_obsm(
    adata: AnnData,
    key: str = 'rank_obsm',
    names: str | list | None = None,
    nvar: int | float | list | None = 10,
    dendrogram: bool = True,
    thr_sign: float = 0.05,
    titles: list = ['Scores', 'Stats'],
    cmap_stat: str = 'Purples',
    cmap_obsm: str = 'BrBG',
    cmap_obs: dict | None = None,
    **kwargs
) -> None | Figure:
    """
    %(plot)s
    """
    assert isinstance(dendrogram, bool), 'dendrogram must be bool'
    assert isinstance(thr_sign, float) and 1 >= thr_sign >= 0, \
    'thr_sign must be float and between 0 and 1'
    assert isinstance(titles, list) and len(titles) == 2, \
    'titles must be list and with 2 elements'
    assert isinstance(cmap_obs, dict) or cmap_obs is None, \
    'cmap_obs must be dict or None'
    # Extract
    obsm, stats = _input(adata=adata, uns_key=key, names=names, nvar=nvar)
    # Instance
    kwargs['ax'] = None
    bp = Plotter(**kwargs)
    bp.fig.delaxes(bp.ax)
    plt.close(bp.fig)
    # Plot stats
    h1 = ma.Heatmap(stats, cmap=cmap_stat, name="h1", width=4, height=1, label=r'$-\log_{10}(padj)$')
    h1.add_title(top=titles[1], align="center")
    if dendrogram:
        h1.add_dendrogram("left")
    h1.add_right(mp.Labels(stats.index, ))
    sign_msk = stats.values > -np.log10(thr_sign)
    layer = mp.MarkerMesh(sign_msk, marker='*', label=f"padj < {thr_sign}", color='red')
    h1.add_layer(layer, name='sign')
    # Plot obsm
    h2 = ma.Heatmap(obsm, cmap=cmap_obsm, name="h2", width=0.4, height=4, label='Score')
    h2.add_title(top=titles[0], align="center")
    if dendrogram:
        h2.add_dendrogram("left")
    h2.add_bottom(mp.Labels(obsm.columns))
    # Add obs legends
    if names is None:
        names = stats.index
    for name in names:
        is_numeric = pd.api.types.is_numeric_dtype(adata.obs[name])
        if is_numeric:
            if cmap_obs is None:
                cmap = 'viridis'
            else:
                cmap = {name: cmap_obs[name]}
            colors = mp.ColorMesh(adata.obs[name], cmap=cmap, label=name)
        else:
            cats = adata.obs[name].sort_values().unique()
            if cmap_obs is None:
                tab10 = plt.get_cmap('tab10')
                palette = {k: tab10(i) for i, k in enumerate(cats)}
            else:
                palette = {name: cmap_obs[name]}
            colors = mp.Colors(adata.obs[name], palette=palette, label=name)
        h2.add_right(colors, pad=0.1, size=0.1)
    # Build plot
    c = (h1 / .05 / h2)
    c.add_legends(side='right', stack_by='row', stack_size=3, align_legends='top')
    c.render()
    if bp.return_fig or save is not None:
        plt.close()
    # Add borders
    hax = c.get_ax(board_name='h1', ax_name='h1')
    border = matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False, ec=".1", lw=2, transform=hax.transAxes)
    hax.add_artist(border)
    hax = c.get_ax(board_name='h2', ax_name='h2')
    border = matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False, ec=".1", lw=2, transform=hax.transAxes)
    hax.add_artist(border)
    bp.fig = c.figure
    bp.fig.set_figwidth(bp.figsize[0])
    bp.fig.set_figheight(bp.figsize[1])
    bp.fig.set_dpi(bp.dpi)
    return bp._return()
