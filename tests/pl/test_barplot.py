import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.fixture
def df():
    df = pd.DataFrame(
        data=[
            [1, -2, 3, -4],
            [5, -6, 7, -8],
        ],
        index=['C1', 'C2'],
        columns=[f'TF{i}' for i in range(4)]
    )
    return df


@pytest.mark.parametrize(
    'name,top,vertical,vcenter',
    [
        ['C1', 2, True, None],
        ['C2', 10, False, -3],
        ['C2', 10, False, 10],
    ]
)
def test_barplot(
    df,
    name,
    top,
    vertical,
    vcenter,
):
    fig = dc.pl.barplot(data=df, name=name, top=top, vertical=vertical, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
