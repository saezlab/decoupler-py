import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import decoupler as dc


def test_filter_by_prop(
    pdata,
):
    fig = dc.pl.filter_by_prop(adata=pdata, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
