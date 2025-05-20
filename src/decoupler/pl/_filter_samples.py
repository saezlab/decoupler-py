import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from anndata import AnnData

from decoupler._docs import docs
from decoupler._Plotter import Plotter


@docs.dedent
def filter_samples(
    adata: AnnData,
    groupby: str | list,
    log: bool = True,
    min_cells: int | float = 10,
    min_counts: int | float = 1000,
    **kwargs
) -> None | Figure:
    """
    Plot to assess the quality of the obtained pseudobulk samples from ``decoupler.pp.pseudobulk``.

    Parameters
    ----------
    %(adata)s
    groupby
        Name or nomes of the ``adata.obs`` column/s to group by.
    log
        If set, log10 transform the ``psbulk_n_cells`` and ``psbulk_counts`` columns during visualization.
    %(min_cells)s
    %(min_counts)s
    %(plot)s
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be AnnData'
    assert isinstance(adata.obs, pd.DataFrame) and adata.obs is not None, \
    f'adata.obs must be a pd.DataFrame not {type(adata.obs)}'
    assert all(col in adata.obs.columns for col in ['psbulk_cells', 'psbulk_counts']), \
    'psbulk_* columns not present in adata.obs, this function should be used after running decoupler.pp.pseudobulk'
    assert isinstance(groupby, (str, list)), 'groupby must be str or list'
    if isinstance(groupby, str):
        groupby = [groupby]
    assert all(col in adata.obs for col in groupby), 'columns in groupby must be in adata.obs'
    # Extract obs
    df = adata.obs.copy()
    # Transform to log10
    label_x, label_y = 'cells', 'counts'
    if log:
        df['psbulk_cells'] = np.log10(df['psbulk_cells'] + 1)
        df['psbulk_counts'] = np.log10(df['psbulk_counts'] + 1)
        label_x, label_y = r'$\log_{10}$ ' + label_x, r'$\log_{10}$ ' + label_y
        min_cells, min_counts = np.log10(min_cells), np.log10(min_counts)
    # Plot
    if len(groupby) > 1:
        # Instance
        assert kwargs.get('ax') is None, 'when groupby is list, ax must be None'
        kwargs['ax'] = None
        bp = Plotter(**kwargs)
        bp.fig.delaxes(bp.ax)
        plt.close(bp.fig)
        bp.fig, axes = plt.subplots(len(groupby), 1, figsize=bp.figsize, dpi=bp.dpi, tight_layout=True)
        axes = axes.ravel()
        for ax, grp in zip(axes, groupby):
            ax.grid(zorder=0)
            ax.set_axisbelow(True)
            sns.scatterplot(x='psbulk_cells', y='psbulk_counts', hue=grp, ax=ax, data=df, zorder=1)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=grp)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            ax.axvline(x=min_cells, linestyle='--', color="black")
            ax.axhline(y=min_counts, linestyle='--', color="black")
    else:
        # Instance
        groupby = groupby[0]
        bp = Plotter(**kwargs)
        bp.ax.grid(zorder=0)
        bp.ax.set_axisbelow(True)
        sns.scatterplot(x='psbulk_cells', y='psbulk_counts', hue=groupby, ax=bp.ax, data=df, zorder=1)
        bp.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=groupby)
        bp.ax.set_xlabel(label_x)
        bp.ax.set_ylabel(label_y)
        bp.ax.axvline(x=min_cells, linestyle='--', color="black")
        bp.ax.axhline(y=min_counts, linestyle='--', color="black")
    return bp._return()
