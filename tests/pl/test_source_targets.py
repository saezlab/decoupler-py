import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'name,a_err', [
        ['T1', False],
        ['T10', True],
    ]
)
def test_source_targets(
    deg,
    net,
    name,
    a_err,
):
    if not a_err:
        fig = dc.pl.source_targets(data=deg, net=net, name=name, x='weight', y='stat', return_fig=True)
        assert isinstance(fig, Figure)
        plt.close(fig)
    else:
        with pytest.raises(AssertionError):
            dc.pl.source_targets(data=deg, net=net, name=name, x='weight', y='stat', return_fig=True)
