import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from decoupler._Plotter import Plotter


def psbulk_samples(
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
    adata
        AnnData obtained after running ``decoupler.get_pseudobulk``.
    groupby
        Name or nomes of the ``adata.obs`` column/s to group by.
    log
        If set, log10 transform the ``psbulk_n_cells`` and ``psbulk_counts`` columns during visualization.
    min_cells
        Threshold used to filter samples with few number of cells. 
    min_counts
        Threshold used to filter samples with few total counts.
    .. inheritdoc:: Plotter.__init__
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be AnnData'
    assert isinstance(adata.obs, pd.DataFrame) and adata.obs is not None, \
    f'adata.obs must be a pd.DataFrame not {type(adata.obs)}'
    assert all(col in df.columns for col in ['psbulk_n_cells', 'psbulk_counts']), \
    'psbulk_* columns not present in adata.obs, this function should be used after running decoupler.pp.pseudobulk'
    assert isinstance(groupby, (str, list)), 'groupby must be str or list'
    if isinstance(groupby, list):
        assert all(col in adata.obs for col in groupby), 'columns in groupby must be in adata.obs'
        assert kwargs.get('ax') is not None, 'when groupby is list, ax must be None'
    # Extract obs
    df = adata.obs.copy()
    # Transform to log10
    if log:
        df['psbulk_n_cells'] = np.log10(df['psbulk_n_cells'])
        df['psbulk_counts'] = np.log10(df['psbulk_counts'])
        label_x, label_y = r'$\log_{10}$ number of cells', r'$\log_{10}$ total sum of counts'
        min_cells, min_counts = np.log10(min_cells), np.log10(min_counts)
    else:
        label_x, label_y = 'number of cells', 'total sum of counts'
    # Plot
    if type(groupby) is list:
        # Instance
        kwargs['ax'] = None
        bp = Plotter(**kwargs)
        bp.fig.delaxes(bp.ax)
        bp.fig, axes = plt.subplots(1, len(groupby), figsize=bp.figsize, dpi=bp.dpi, tight_layout=True)
        axes = axes.ravel()
        for ax, grp in zip(axes, groupby):
            ax.grid(zorder=0)
            ax.set_axisbelow(True)
            sns.scatterplot(x='psbulk_n_cells', y='psbulk_counts', hue=grp, ax=ax, data=df, zorder=1)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=grp)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            ax.axvline(x=min_cells, linestyle='--', color="black")
            ax.axhline(y=min_counts, linestyle='--', color="black")
    else:
        # Instance
        bp = Plotter(**kwargs)
        bp.ax.grid(zorder=0)
        bp.ax.set_axisbelow(True)
        sns.scatterplot(x='psbulk_n_cells', y='psbulk_counts', hue=groupby, ax=bp.ax, data=df, zorder=1)
        bp.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=groupby)
        bp.ax.set_xlabel(label_x)
        bp.ax.set_ylabel(label_y)
        bp.ax.axvline(x=min_cells, linestyle='--', color="black")
        bp.ax.axhline(y=min_counts, linestyle='--', color="black")
    return bp._return()
