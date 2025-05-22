import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.fixture
def data():
    data = pd.DataFrame(
        data=[
            [5, 6, 7, 1, 1, 2.],
        ],
        index=['C1'],
        columns=['G01', 'G02', 'G03', 'G06', 'G07', 'G08']
    )
    return data


@pytest.fixture
def score():
    score = pd.DataFrame(
        data=[
            [4, 3, -3, -2.],
        ],
        index=['C1'],
        columns=[f'T{i + 1}' for i in range(4)]
    )
    return score


@pytest.mark.parametrize(
    'd_none,unw,sources,targets,by_abs,vcenter',
    [
        [False, False, 5, 5, False, False],
        [False, True, 'T1', 5, True, True],
        [True, False, ['T1'], 5, True, True],
        [True, False, ['T1', 'T3'], 5, True, True],
        [False, False, 5, 'G01', True, True],
        [False, False, 5, ['G01', 'G02', 'G03'], True, True],
    ]
)
def test_network(
    net,
    data,
    score,
    d_none,
    unw,
    sources,
    targets,
    by_abs,
    vcenter,
):
    if d_none:
        s_cmap = 'white'
        data = None
        score = None
    else:
        s_cmap = 'coolwarm'
    if unw:
        net = net.drop(columns=['weight'])
    fig = dc.pl.network(
        data=data,
        score=score,
        net=net,
        sources=sources,
        targets=targets,
        by_abs=by_abs,
        vcenter=vcenter,
        s_cmap = s_cmap,
        figsize=(5, 5),
        return_fig=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)