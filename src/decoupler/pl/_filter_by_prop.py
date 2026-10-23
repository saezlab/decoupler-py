import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.figure import Figure

from decoupler._docs import docs
from decoupler._Plotter import Plotter


@docs.dedent
def filter_by_prop(
    adata: AnnData, min_prop: float = 0.1, min_smpls: int = 2, kw_hist: dict | None = None, **kwargs
) -> None | Figure:
    """
    Plot to help determining the thresholds of the ``decoupler.pp.filter_by_prop`` function.

    Parameters
    ----------
    %(adata)s
    %(min_prop_prop)s
    %(min_smpls)s
    kw_hist
        Keyword arguments passed to ``matplotlib.pyplot.hist``.
    %(plot)s

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        adata = dc.ds.covid5k()
        pdata = dc.pp.pseudobulk(adata, sample_col="individual", groups_col="celltype")
        tcells = pdata[pdata.obs["celltype"] == "T cell"].copy()
        dc.pl.filter_by_prop(tcells)
    """
    assert isinstance(adata, AnnData), "adata must be AnnData"
    assert "psbulk_props" in adata.layers.keys(), (
        "psbulk_props must be in adata.layers, use this function afer running decoupler.pp.pseudobulk"
    )
    if kw_hist is None:
        kw_hist = {}
    kw_hist.setdefault("color", "gray")
    kw_hist.setdefault("align", "left")
    kw_hist.setdefault("rwidth", 0.95)
    kw_hist.setdefault("log", True)
    props = adata.layers["psbulk_props"]
    if isinstance(props, pd.DataFrame):
        props = props.values
    nsmpls = np.sum(props >= min_prop, axis=0)
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    _ = bp.ax.hist(nsmpls, bins=range(min(nsmpls), max(nsmpls) + 2), **kw_hist)
    bp.ax.axvline(x=min_smpls - 0.5, c="black", ls="--")
    bp.ax.set_xlabel("Samples (≥ min_prop)")
    bp.ax.set_ylabel("Number of genes")
    return bp._return()
