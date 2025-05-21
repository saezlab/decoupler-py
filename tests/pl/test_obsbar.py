import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'y,hue,kw',
    [
        ['group', None, dict()],
        ['group', 'group', dict(width=0.5)],
        ['group', 'sample', dict(palette='tab10')],
        ['sample', 'group', dict(palette='tab20')],
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
