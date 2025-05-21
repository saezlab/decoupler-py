import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import decoupler as dc


def test_source_targets(
    net,
):
    deg = pd.DataFrame(
        data = [
            ['G01', 1, 1, 5],
            ['G02', 3, 1, 10],
            ['G03', 4, 10, 10],
            ['G04', 5, 15, 11],
        ],
        columns=['y', 'x', 'c', 's'],
    ).set_index('y')
    fig = dc.pl.source_targets(data=deg, net=net, name='T1', x='weight', y='x', return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
