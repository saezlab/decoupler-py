import requests
import io
import warnings

import anndata as ad


def _download_anndata(url):
    warnings.filterwarnings("ignore", category=FutureWarning)
    response = requests.get(url)
    adata = ad.read_h5ad(io.BytesIO(response.content))
    return adata


def pbmc3k():
    url = ('https://raw.githubusercontent.com/chanzuckerberg/' \
    'cellxgene/main/example-dataset/pbmc3k.h5ad')
    adata = _download_anndata(url)
    adata = adata.raw.to_adata()
    adata.obs['celltype'] = adata.obs['louvain']
    adata.obs['leiden'] = adata.obs['louvain'].cat.codes
    adata.obs['leiden'] = adata.obs['leiden'].apply(lambda x: f'Clust:{x}')
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')
    del adata.obs['louvain']
    return adata
