import pandas as pd
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
    'name,top,vertical',
    [
        ['C1', 2, True],
        ['C2', 10, False],
    ]
)
def test_barplot(
    df,
    name,
    top,
    vertical,
):
    fig = dc.pl.barplot(data=df, name=name, top=top, vertical=vertical, return_fig=True)
    assert isinstance(fig, Figure)
