import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from anndata import AnnData

from decoupler._docs import docs
from decoupler._Plotter import Plotter


@docs.dedent
def obsbar(
    adata: AnnData,
    y: str,
    hue: str | None = None,
    kw_barplot: dict = dict(),
    **kwargs
) -> None | Figure:
    """
    Plot ``adata.obs`` metadata as a grouped barplot.

    Parameters
    ----------
    %(adata)s
    y
        Column name in ``adata.obs`` to plot in y axis.
    hue
        Column name in ``adata.obs`` to color bars.
    kw_barplot
        Keyword arguments passed to ``seaborn.barplot``.
    %(plot)s
    """
    # Validate
    assert isinstance(adata, AnnData), 'adata must be an AnnData instance'
    assert isinstance(y, str), 'y must be str'
    assert isinstance(hue, str) or hue is None, 'hue must be str or None'
    cols = {y, hue}
    if hue is None:
        cols.remove(None)
    assert cols.issubset(adata.obs.columns), \
    f'y={y} and hue={hue} must be in adata.obs.columns={adata.obs.columns}'
    cols = list(cols)
    # Process
    data = (
        adata.obs
        .groupby(cols, observed=True, as_index=False)
        .size()
    )
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    sns.barplot(
        data=data,
        y=y,
        x='size',
        hue=hue,
        ax=bp.ax,
        **kw_barplot
    )
    if hue is not None and y != hue:
        bp.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=hue)
    return bp._return()
