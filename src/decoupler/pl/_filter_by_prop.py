from anndata import AnnData
from matplotlib.figure import Figure

from decoupler._Plotter import Plotter

def filter_by_prop(
    adata: AnnData,
    min_prop: float = 0.2,
    min_smpls: int = 2,
    cmap: str = 'viridis',
) -> None | Figure:
    """
    Plot to help determining the thresholds of the ``decoupler.pp.filter_by_prop`` function.

    Parameters
    ----------
    adata
        AnnData obtained after running ``decoupler.pp.pseudobulk``. It requieres ``.layer['psbulk_props']``.
    cmap
        Colormap to use.
    .. inheritdoc:: decoupler.pp.filter_by_expr
    .. inheritdoc:: Plotter.__init__
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
    _ = bg.ax.hist(nsmpls, log=True, color='gray')
    bp.ax.axvline(x=min_smpls, c='black', ls='--')
    bp.ax.set_xlabel('Number of samples where >= min_prop')
    bp.ax.set_ylabel('Number of genes')
    return bp._return()
