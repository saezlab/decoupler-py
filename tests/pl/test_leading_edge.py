import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'stat,name',
    [
        ['stat', 'T1'],
        ['stat', 'T2'],
        ['pval', 'T3'],
        ['pval', 'T4'],
    ]
)
def test_leading_edge(
    net,
    stat,
    name,
):
    df = pd.DataFrame(
        data=[[i, i ** 2] for i in range(9)],
        columns=['stat', 'pval'],
        index=[f'G0{i}' for i in range(9)],
    )
    fig, le = dc.pl.leading_edge(df=df, net=net, stat=stat, name=name, return_fig=True)
    assert isinstance(le, np.ndarray)
    assert isinstance(fig, Figure)
