import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'use_net,name,a_err',
    [
        [False, None, False],
        [True, 'T1', False],
        [True, 'T2', False],
        [True, 'T3', False],
        [True, 'T10', True],
    ]
)
def test_volcano(
    deg,
    net,
    use_net,
    name,
    a_err,
):
    if not use_net:
        net = None
        name = None
    if not a_err:
        fig = dc.pl.volcano(data=deg, x='stat', y='padj', net=net, name=name, return_fig=True)
        assert isinstance(fig, Figure)
        plt.close(fig)
    else:
        with pytest.raises(AssertionError):
            dc.pl.volcano(data=deg, x='stat', y='padj', net=net, name=name, return_fig=True)
