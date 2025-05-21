import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'names,label,mode',
    [
        [['G01', 'G02', 'G07', 'G08', 'G12'], None, 'line'],
        [['G01', 'G02', 'G07', 'G08'], None, 'mat'],
        [None, 'group', 'line'],
        [None, 'group', 'mat'],
    ]
)
def test_order(
    tdata,
    names,
    label,
    mode,
):
    df = dc.pp.bin_order(adata=tdata, names=['G12', 'G01', 'G07', 'G04'], order='pstime', label=label)
    fig = dc.pl.order(df=df, mode=mode, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
