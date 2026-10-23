import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import decoupler as dc


def test_filter_by_prop(
    pdata,
):
    fig = dc.pl.filter_by_prop(adata=pdata, return_fig=True, kw_hist={"alpha": 0.5})
    assert isinstance(fig, Figure)
    plt.close(fig)
