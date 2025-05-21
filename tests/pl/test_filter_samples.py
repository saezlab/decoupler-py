import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'groupby,log',
    [
        ['group', True],
        [['group'], True],
        [['sample', 'group'], True],
    ]
)
def test_filter_samples(
    pdata,
    groupby,
    log,
):
    fig = dc.pl.filter_samples(adata=pdata, groupby=groupby, log=log, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
