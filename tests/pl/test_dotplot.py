import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

import decoupler as dc


@pytest.fixture
def df():
    df = pd.DataFrame(
        data=[
            ["TF1", 1, 1, 5],
            ["TF2", 3, 1, 10],
            ["TF3", 4, 10, 10],
            ["TF4", 5, 15, 11],
        ],
        columns=["y", "x", "c", "s"],
    )
    return df


@pytest.mark.parametrize("vcenter", [None, 3])
def test_dotplot(
    df,
    vcenter,
):
    fig = dc.pl.dotplot(df=df, x="x", y="y", c="c", s="s", vcenter=vcenter, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
