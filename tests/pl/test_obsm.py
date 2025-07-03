import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

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
        [True, None, 10, True, ['Scores', 'Stats'], {}],
        [True, 'group', 5, False, ['asd', 'fgh'], {}],
        [True, ['group', 'pstime'], 10, True, ['Scores', 'Stats'], {}],
        [True, None, 10, True, ['Scores', 'Stats'], {'group': 'tab10', 'pstime': 'magma', 'sample': 'Pastel1'}],
        [True, None, 2, True, ['Scores', 'Stats'], {'pstime': 'magma'}],
        [True, None, ['PC01', 'PC02'], True, ['Scores', 'Stats'], {'pstime': 'magma'}],
        [False, None, None, True, ['Scores', 'Stats'], {}],
        [False, None, 10, True, ['Scores', 'Stats'], {}],
        [False, None, 'T3', True, ['Scores', 'Stats'], {}],
        [False, None, ['T5', 'T3'], True, ['Scores', 'Stats'], {}],
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
