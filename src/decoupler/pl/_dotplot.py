import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm

from decoupler._Plotter import Plotter


def dotplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    c: str,
    s: str,
    scale: int | float = 0.15,
    cmap: str = 'RdBu_r',
    vcenter: int | float | None = None,
    **kwargs
) -> None | Figure:
    """
    Plot results of enrichment analysis as dots.

    """
    # Validate
    assert isinstance(df, pd.DataFrame), 'df must be a pd.DataFrame'
    assert isinstance(x, str) and x in df.columns, 'x must be str and in df.columns'
    assert isinstance(y, str) and y in df.columns, 'y must be str and in df.columns'
    assert isinstance(c, str) and c in df.columns, 'c must be str and in df.columns'
    assert isinstance(s, str) and s in df.columns, 's must be str and in df.columns'
    assert isinstance(scale, (int, float)), 'scale must be numerical'
    assert isinstance(vcenter, (int, float)) or vcenter is None, 'vcenter must be numeric or None'
    # Extract from df
    x_vals = df[x].values
    y_vals = df[y].values
    c_vals = df[c].values
    s_vals = df[s].values
    # Sort by x
    idxs = np.argsort(x_vals)
    x_vals = x_vals[idxs]
    y_vals = y_vals[idxs]
    c_vals = c_vals[idxs]
    s_vals = s_vals[idxs]
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    ns = (s_vals * scale * plt.rcParams["lines.markersize"]) ** 2
    bp.ax.grid(axis='x')
    if vcenter:
        norm = TwoSlopeNorm(vmin=None, vcenter=vcenter, vmax=None)
    else:
        norm = None
    scatter = bp.ax.scatter(
        x=x_vals,
        y=y_vals,
        c=c_vals,
        s=ns,
        cmap=cmap,
        norm=norm,
    )
    bp.ax.set_axisbelow(True)
    bp.ax.set_xlabel(x)
    # Add legend
    handles, labels = scatter.legend_elements(
        prop="sizes",
        num=3,
        fmt="{x:.2f}",
        func=lambda s: np.sqrt(s) / plt.rcParams["lines.markersize"] / scale
    )
    bp.ax.legend(
        handles,
        labels,
        title=s,
        frameon=False,
        loc='lower left',
        bbox_to_anchor=(1.05, 0.5),
        alignment='left',
        labelspacing=1.
    )
    # Add colorbar
    clb = bp.fig.colorbar(
        scatter,
        ax=bp.ax,
        shrink=0.25,
        aspect=5,
        orientation='vertical',
        anchor=(0., 0.),
    )
    clb.ax.set_title(c, loc="left",)
    bp.ax.margins(x=0.25, y=0.1)
    return bp._return()
