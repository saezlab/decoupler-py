import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'stat,name,a_err',
    [
        ['stat', 'T1', False],
        ['stat', 'T2', False],
        ['pval', 'T3', False],
        ['pval', 'T4', False],
    ]
)
def test_leading_edge(
    net,
    stat,
    name,
    a_err
):
    df = pd.DataFrame(
        data=[[i, i ** 2] for i in range(9)],
        columns=['stat', 'pval'],
        index=[f'G0{i}' for i in range(9)],
    )
    if not a_err:
        fig, le = dc.pl.leading_edge(df=df, net=net, stat=stat, name=name, return_fig=True)
        assert isinstance(le, np.ndarray)
        assert isinstance(fig, Figure)
        plt.close(fig)
    else:
        with pytest.raises(AssertionError):
            dc.pl.leading_edge(df=df, net=net, stat=stat, name=name, return_fig=True)
