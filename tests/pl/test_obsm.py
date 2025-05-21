import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.fixture
def tdata_obsm_ranked(
    tdata_obsm,
):
    dc.tl.rankby_obsm(tdata_obsm, key='X_pca')
    return tdata_obsm


@pytest.mark.parametrize(
    'names,nvar,dendrogram,titles,cmap_obs',
    [
        [None, 10, True, ['Scores', 'Stats'], dict()],
        ['group', 5, False, ['asd', 'fgh'], dict()],
        [['group', 'pstime'], 10, True, ['Scores', 'Stats'], dict()],
        [None, 10, True, ['Scores', 'Stats'], dict(group='tab10', pstime='magma', sample='Pastel1')],
        [None, 2, True, ['Scores', 'Stats'], dict(pstime='magma')],
    ]
)
def test_obsm(
    tdata_obsm_ranked,
    names,
    nvar,
    dendrogram,
    titles,
    cmap_obs,
):
    fig = dc.pl.obsm(
        tdata_obsm_ranked,
        names=names,
        nvar=nvar,
        dendrogram=dendrogram,
        titles=titles,
        cmap_obs=cmap_obs,
        return_fig=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)
