import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

import decoupler as dc


@pytest.mark.parametrize(
    'y,hue,kw',
    [
        ['group', None, {}],
        ['group', 'group', {'width': 0.5}],
        ['group', 'sample', {'palette': 'tab10'}],
        ['sample', 'group', {'palette': 'tab20'}],
    ]
)
def test_obsbar(
    adata,
    y,
    hue,
    kw,
):
    fig = dc.pl.obsbar(adata=adata, y=y, hue=hue, kw_barplot=kw, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
