import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib.figure import Figure

from decoupler._docs import docs
from decoupler._Plotter import Plotter


@docs.dedent
def filter_by_prop(
    adata: AnnData,
    min_prop: float = 0.1,
    min_smpls: int = 2,
    log: bool = True,
    color = 'gray',
    **kwargs
) -> None | Figure:
    """
    Plot to help determining the thresholds of the ``decoupler.pp.filter_by_prop`` function.

    Parameters
    ----------
    %(adata)s
    %(min_prop_prop)s
    %(min_smpls)s
    log
        Whether to log-scale the y axis.
    color
        Color to use in ``matplotlib.pyplot.hist``.
    %(plot)s
    """
    assert isinstance(adata, AnnData), 'adata must be AnnData'
    assert 'psbulk_props' in adata.layers.keys(), \
    'psbulk_props must be in adata.layers, use this function afer running decoupler.pp.pseudobulk'
    props = adata.layers['psbulk_props']
    if isinstance(props, pd.DataFrame):
        props = props.values
    nsmpls = np.sum(props >= min_prop, axis=0)
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    _ = bp.ax.hist(
        nsmpls,
        bins=range(min(nsmpls), max(nsmpls) + 2),
        log=log,
        color=color,
        align='left',
        rwidth=0.95,
    )
    bp.ax.axvline(x=min_smpls - 0.5, c='black', ls='--')
    bp.ax.set_xlabel('Samples (≥ min_prop)')
    bp.ax.set_ylabel('Number of genes')
    return bp._return()
