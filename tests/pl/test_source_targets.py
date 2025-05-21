import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import decoupler as dc


def test_source_targets(
    deg,
    net,
):
    fig = dc.pl.source_targets(data=deg, net=net, name='T1', x='weight', y='stat', return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
