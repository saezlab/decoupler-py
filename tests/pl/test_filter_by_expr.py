import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import decoupler as dc


def test_filter_by_expr(
    pdata,
):
    fig = dc.pl.filter_by_expr(adata=pdata, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
