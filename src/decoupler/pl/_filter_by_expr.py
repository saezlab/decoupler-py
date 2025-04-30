import numpy as np
from matplotlib.figure import Figure
import seaborn as sns
from anndata import AnnData

from decoupler._Plotter import Plotter
from decoupler.pp.data import extract
from decoupler.pp.anndata import _min_sample_size, _ssize_tcount


def filter_by_expr(
    adata: AnnData,
    group: str | None = None,
    lib_size: float | None = None,
    min_count: int = 10,
    min_total_count: int = 15,
    large_n: int = 10,
    min_prop: float = 0.7,
    cmap: str = 'viridis',
    **kwargs,
) -> None | Figure:
    """
    Plot to help determining the thresholds of the ``decoupler.filter_by_expr`` function.

    Parameters
    ----------
    adata
        AnnData obtained after running ``decoupler.get_pseudobulk``.
    cmap
        Colormap to use.
    .. inheritdoc:: decoupler.pl._filter_by_expr
    .. inheritdoc:: Plotter.__init__
    """
    assert isinstance(adata, AnnData), 'adata must be AnnData'
    # Extract inputs
    X, _, _ = extract(adata, empty=False)
    obs = adata.obs
    # Minimum sample size cutoff
    min_sample_size = _min_sample_size(
        obs=obs,
        group=group,
        large_n=large_n,
        min_prop=min_prop,
    )
    # Compute sample size and total count
    sample_size, total_count = _ssize_tcount(
        X=X,
        lib_size=lib_size,
        min_count=min_count,
    )
    # Total counts
    total_count[total_count < 1.] = np.nan  # Handle 0s
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    sns.histplot(
        x=np.log10(total_count),
        y=sample_size,
        cmap=cmap,
        cbar=True,
        cbar_kws=dict(shrink=.75, label='Number of features'),
        discrete=(False, True),
        ax=bp.ax,
    )
    bp.ax.axhline(y=min_sample_size - 0.5, c='gray', ls='--')
    bp.ax.axvline(x=np.log10(min_total_count), c='gray', ls='--')
    bp.ax.set_xlabel(r'$\log_{10}$ total sum of counts')
    bp.ax.set_ylabel('Number of samples')
    return bp._return()
