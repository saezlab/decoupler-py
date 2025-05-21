import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc


@pytest.fixture
def tdata_obsm_pca(
    tdata_obsm,
):
    dc.tl.rankby_obsm(tdata_obsm, key='X_pca')
    return tdata_obsm


@pytest.fixture
def tdata_obsm_ulm(
    tdata_obsm,
):
    tdata_obsm = tdata_obsm.copy()
    dc.tl.rankby_obsm(tdata_obsm, key='score_ulm')
    return tdata_obsm


@pytest.mark.parametrize(
    'pca,names,nvar,dendrogram,titles,cmap_obs',
    [
        [True, None, 10, True, ['Scores', 'Stats'], dict()],
        [True, 'group', 5, False, ['asd', 'fgh'], dict()],
        [True, ['group', 'pstime'], 10, True, ['Scores', 'Stats'], dict()],
        [True, None, 10, True, ['Scores', 'Stats'], dict(group='tab10', pstime='magma', sample='Pastel1')],
        [True, None, 2, True, ['Scores', 'Stats'], dict(pstime='magma')],
        [True, None, ['PC01', 'PC02'], True, ['Scores', 'Stats'], dict(pstime='magma')],
        [False, None, None, True, ['Scores', 'Stats'], dict()],
        [False, None, 10, True, ['Scores', 'Stats'], dict()],
        [False, None, 'T3', True, ['Scores', 'Stats'], dict()],
        [False, None, ['T5', 'T3'], True, ['Scores', 'Stats'], dict()],
    ]
)
def test_obsm(
    tdata_obsm_pca,
    tdata_obsm_ulm,
    pca,
    names,
    nvar,
    dendrogram,
    titles,
    cmap_obs,
):
    if pca:
        tdata_obsm_ranked = tdata_obsm_pca
    else:
        tdata_obsm_ranked = tdata_obsm_ulm
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
