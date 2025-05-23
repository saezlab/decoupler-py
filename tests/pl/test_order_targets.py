import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'source,label,vmin,vmax',
    [
        ['T1', None, 0, 10],
        ['T2', 'group', -3, 10],
        ['T3', 'group', -20, 15],
        ['T4', 'group', -1, 20],
        ['T5', 'group', -2, 14],
        ['T5', 'group', None, None],
    ]
)
def test_order_targets(
    tdata,
    net,
    source,
    label,
    vmin,
    vmax,
):
    dc.mt.ulm(tdata, net, tmin=0)
    fig = dc.pl.order_targets(
        adata=tdata,
        net=net,
        order='pstime',
        source=source,
        label=label,
        vmin=vmin,
        vmax=vmax,
        return_fig=True,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)
