import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'use_net,name',
    [
        [False, None],
        [True, 'T1'],
        [True, 'T2'],
        [True, 'T3'],
    ]
)
def test_volcano(
    deg,
    net,
    use_net,
    name,
):
    if not use_net:
        net = None
        name = None
    fig = dc.pl.volcano(data=deg, x='stat', y='padj', net=net, name=name, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
