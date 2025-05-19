import warnings

import pandas as pd
import pytest
import anndata as ad

import decoupler as dc


@pytest.mark.parametrize(
    'url', [
        'https://datasets.cellxgene.cziscience.com/' +
        'f665effe-d95a-4211-ab03-9d1777ca0806.h5ad',
        'https://datasets.cellxgene.cziscience.com/' +
        '1338d08a-481a-426c-ad60-9f4ac08afe16.h5ad'
    ]
)
def test_download_anndata(
    url
):
    warnings.filterwarnings("ignore", module="anndata")
    adata = dc.ds._scell._download_anndata(url=url)
    assert isinstance(adata, ad.AnnData)


def test_pbmc3k():
    warnings.filterwarnings("ignore", module="anndata")
    adata = dc.ds.pbmc3k()
    assert isinstance(adata, ad.AnnData)
    assert adata.raw is None
    assert isinstance(adata.obs, pd.DataFrame)
    cols = {'celltype', 'leiden'}
    assert cols.issubset(adata.obs.columns)
    assert 'louvain' not in adata.obs.columns
    for col in cols:
        assert isinstance(adata.obs[col].dtype, pd.CategoricalDtype)


def test_covid5k():
    adata = dc.ds.covid5k()
    assert isinstance(adata, ad.AnnData)
    assert adata.raw is None
    assert isinstance(adata.obs, pd.DataFrame)
    cols = {'individual', 'sex', 'disease', 'celltype'}
    assert cols.issubset(adata.obs.columns)
    for col in cols:
        assert isinstance(adata.obs[col].dtype, pd.CategoricalDtype)


def test_erygast1k():
    adata = dc.ds.erygast1k()
    assert isinstance(adata, ad.AnnData)
    assert adata.raw is None
    assert isinstance(adata.obs, pd.DataFrame)
    cols = {'sample', 'stage', 'sequencing.batch', 'theiler', 'celltype'}
    assert cols.issubset(adata.obs.columns)
    for col in cols:
        assert isinstance(adata.obs[col].dtype, pd.CategoricalDtype)
    keys = {'X_pca', 'X_umap'}
    assert keys.issubset(adata.obsm.keys())
    