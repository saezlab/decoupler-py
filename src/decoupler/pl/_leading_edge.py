from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm

from decoupler._docs import docs
from decoupler._Plotter import Plotter
from decoupler.pp.net import prune
from decoupler.mt._gsea import _esrank


@docs.dedent
def leading_edge(
    df: pd.DataFrame,
    net: pd.DataFrame,
    stat: str,
    name: str,
    cmap='RdBu_r',
    color='#88c544',
    **kwargs
) -> Tuple[None | Figure, np.ndarray]:
    """
    Plot the running score of GSEA.

    Parameters
    ----------
    %(data_plot)s
    %(net)s
    stat
        Column with the ranking statistic, for example t-values or :math:`log_{2}FCs`.
    name
        Which source to plot.
    %(cmap)s
    color
        Color to plot the running-sum statistic.
    %(plot)s
    """
    class MidpointNormalize(matplotlib.colors.Normalize):
        def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
            self.vcenter = vcenter
            super().__init__(vmin, vmax, clip)
        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))
    # Validate
    assert isinstance(df, pd.DataFrame), 'df must be pd.DataFrame'
    assert isinstance(stat, str) and stat in df.columns, 'stat must be str and in pd.DataFrame'
    assert isinstance(net, pd.DataFrame), 'net must be pd.DataFrame'
    assert isinstance(name, str), 'name must be str'
    # Extract feature level stats and names from df
    c = df.index.values.astype('U')
    m = df[stat].values
    # Remove empty values
    msk = np.isfinite(m)
    c = c[msk]
    m = m[msk]
    # Transform net
    snet = prune(features=c, net=net, tmin=0, verbose=False)
    snet = snet[snet['source'] == name]
    assert snet.shape[0] > 0, f'name={name} must be in net["source"]'
    # Sort features
    idx = np.argsort(-m)
    m = m[idx]
    c = c[idx]
    # Get ranks
    rnks = np.arange(c.size)
    # Get msk
    set_msk = np.isin(c, snet['target'])
    # Get decending penalty
    n_features = set_msk.size
    nf_in_set = set_msk.sum()
    dec = 1.0 / (n_features - nf_in_set)
    # Compute es
    mx_value, j, es = _esrank(row=m, rnks=rnks, set_msk=set_msk, dec=dec)
    # Get leading edge features
    sign = np.sign(mx_value)
    set_rnks = rnks[set_msk]
    if sign > 0:
        le_c = c[set_rnks[set_rnks <= j]]
    else:
        le_c = np.flip(c[set_rnks[set_rnks >= j]])
    # Instance
    kwargs['ax'] = None
    bp = Plotter(**kwargs)
    bp.fig.delaxes(bp.ax)
    plt.close(bp.fig)
    # Plot
    gridspec_kw = {'height_ratios': [4, 0.5, 0.5, 2]}
    bp.fig, axes = plt.subplots(4, 1, gridspec_kw=gridspec_kw, figsize=bp.figsize, sharex=True, dpi=bp.dpi)
    axes = axes.ravel()
    # Plot random walk
    ax = axes[0]
    ax.margins(0.)
    ax.plot(rnks, es, color=color, linewidth=2)
    ax.axvline(rnks[j], linestyle='--', color=color)
    ax.axhline(0, linestyle='--', color=color)
    ax.set_ylabel('Enrichment\nScore')
    ax.set_title(name)
    # Plot gset mask
    ax = axes[1]
    ax.margins(0.)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.vlines(rnks[set_msk], 0, 1, linewidth=0.5, color=color)
    # Plot color bar
    ax = axes[2]
    ax.margins(0.)
    ax.set_yticklabels([])
    ax.set_yticks([])
    vmin = np.percentile(np.min(m), 2)
    vmax = np.percentile(np.max(m), 98)
    norm = MidpointNormalize(vmin=vmin, vcenter=0, vmax=vmax)
    ax.pcolormesh(
        m[np.newaxis, :],
        rasterized=True,
        norm=norm,
        cmap=cmap,
    )
    ax.set_xlim(0, rnks.size-1)  # Remove extreme to the right
    # Plot ranks
    ax = axes[3]
    ax.margins(0.)
    ax.fill_between(rnks, y1=m, y2=0, color="#C9D3DB")
    non_zero_rnks = rnks[m > 0]
    if non_zero_rnks.size == 0:
        zero_rnk = rnks[-1]
    else:
        zero_rnk = non_zero_rnks[-1] + 1
    ax.axvline(zero_rnk, linestyle='--', color="#C9D3DB")
    ax.set_xlabel('Rank')
    ax.set_ylabel('Ranked\nmetric')
    # Remove spaces
    bp.fig.subplots_adjust(wspace=0, hspace=0)
    return bp._return(), le_c
