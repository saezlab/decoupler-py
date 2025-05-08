import os
import io
import gzip
import warnings
import requests

import pandas as pd
from tqdm.auto import tqdm
import scipy.io as sio
import anndata as ad

from decoupler._docs import docs
from decoupler._log import _log
from decoupler.ds._utils import ensmbl_to_symbol


def _download_anndata(
    url: str,
    verbose: bool = False,
) -> ad.AnnData:
    warnings.filterwarnings("ignore", category=FutureWarning)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192
        buffer = io.BytesIO()
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc="Downloading .h5ad", disable=not verbose) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                buffer.write(chunk)
                pbar.update(len(chunk))
    buffer.seek(0)
    adata = ad.read_h5ad(buffer)
    return adata


@docs.dedent
def pbmc3k(
    verbose: bool = False,
) -> ad.AnnData:
    """
    Downloads single-cell RNA-seq data of peripheral blood mononuclear
    cells (PBMCs) from a healthy donor.

    Parameters
    ----------
    %(verbose)s

    Returns
    -------
    AnnData object.
    """
    url = ('https://raw.githubusercontent.com/chanzuckerberg/' \
    'cellxgene/main/example-dataset/pbmc3k.h5ad')
    adata = _download_anndata(url, verbose=verbose)
    adata = adata.raw.to_adata()
    adata.obs['celltype'] = adata.obs['louvain']
    adata.obs['leiden'] = adata.obs['louvain'].cat.codes
    adata.obs['leiden'] = adata.obs['leiden'].apply(lambda x: f'Clust:{x}')
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')
    del adata.obs['louvain']
    m = f'generated AnnData with shape={adata.shape}'
    _log(m, level='info', verbose=verbose)
    return adata


@docs.dedent
def covid5k(
    verbose: bool = False,
) -> ad.AnnData:
    """
    Downloads single-cell RNA-seq data of peripheral blood mononuclear
    cells (PBMCs) from a cohort of patients with and without
    COVID-19 :cite:`covid5k`.

    Parameters
    ----------
    %(verbose)s

    Returns
    -------
    AnnData object.
    """
    url = (
        'https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/'
        'sc_experiments/E-MTAB-9221/E-MTAB-9221.aggregated_counts.mtx.gz'
    )
    url_base = os.path.dirname(url)
    id_ebi, _, _, _ = os.path.basename(url).split('.')
    # Download X
    response = requests.get(url)
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        X = sio.mmread(f).T.tocsr().rint()
        X.eliminate_zeros()
    # Download var
    url_var = os.path.join(url_base, f'{id_ebi}.aggregated_counts.mtx_rows.gz')
    var = pd.read_csv(url_var, sep='\t', header=None, usecols=[0], index_col=0)
    var['name'] = ensmbl_to_symbol(var.index)
    msk_var = ~(var['name'].isna() | var['name'].duplicated(keep='first'))
    var = var.loc[msk_var].reset_index(drop=True).set_index('name')
    var.index.name = None
    X = X[:, msk_var]
    # Download obs
    url_obs = os.path.join(url_base, f'{id_ebi}.cell_metadata.tsv')
    cols = ['id', 'individual', 'sex', 'disease', 'inferred_cell_type_-_ontology_labels']
    obs = pd.read_csv(url_obs, sep='\t')[cols].set_index('id')
    obs = obs.rename(columns={'inferred_cell_type_-_ontology_labels': 'celltype'})
    obs.index.name = None
    msk_obs = (~obs['celltype'].isna())
    obs = obs.loc[msk_obs]
    X = X[msk_obs, :]
    # Make AnnData
    adata = ad.AnnData(X, obs=obs, var=var)
    # Download umap
    url_umap = os.path.join(url_base, f'{id_ebi}.umap_neigh_15.tsv')
    umap = pd.read_csv(url_umap, sep='\t', header=None, index_col=0)
    inter = umap.index.intersection(adata.obs_names)
    adata = adata[inter, :].copy()
    adata.obsm['X_umap'] = umap.loc[inter].values
    # Basic filtering
    msk_var = adata.X.getnnz(axis=0) > 3
    msk_obs = adata.X.getnnz(axis=1) > 3
    adata = adata[msk_obs, msk_var].copy()
    # Make categorical
    for col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype('category')
    m = f'generated AnnData with shape={adata.shape}'
    _log(m, level='info', verbose=verbose)
    return adata
