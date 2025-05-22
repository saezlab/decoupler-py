import requests
import io
import gzip
import json

import pandas as pd
import scipy.io as sio
from matplotlib.image import imread
from anndata import AnnData

from decoupler._docs import docs
from decoupler._log import _log


@docs.dedent
def msvisium(
    verbose: bool = False,
) -> AnnData:
    """
    Downloads a spatial RNA-seq (Visium) human sample with multiple sclerosis
    displaying a chronic active lesion in the white matter of the brain :cite:`msvisium`.

    Parameters
    ----------
    %(verbose)s

    Returns
    -------
    AnnData object.
    """
    url = (
        'https://www.ncbi.nlm.nih.gov/geo/download/'
        '?acc=GSM8563708&format=file&file=GSM8563708%5FMS377T%5F'
    )
    # Download mat
    response = requests.get(url + 'matrix%2Emtx%2Egz')
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        X = sio.mmread(f).T.tocsr().rint()
        X.eliminate_zeros()
    var = pd.read_csv(
        url + 'features%2Etsv%2Egz',
        compression='gzip',
        sep='\t',
        header=None,
        usecols=[1],
        index_col=0,
    )
    var.index.name = None
    # Remove repeated genes
    msk_var = ~(var.index.duplicated(keep='first'))
    var = var.loc[msk_var]
    X = X[:, msk_var]
    obs = pd.read_csv(
        url + 'barcodes%2Etsv%2Egz',
        compression='gzip',
        sep='\t',
        header=None,
        usecols=[0],
        index_col=0,
    )
    obs.index.name = None
    # Create anndata
    adata = AnnData(X=X, obs=obs, var=var)
    # Add images
    adata.uns['spatial'] = dict()
    adata.uns['spatial']['MS377T'] = dict()
    adata.uns['spatial']['MS377T']['images'] = dict()
    response = requests.get(url + 'scalefactors%5Fjson%2Ejson%2Egz')
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        adata.uns['spatial']['MS377T']['scalefactors'] = json.load(f)
    response = requests.get(url + 'tissue%5Fhires%5Fimage%2Epng%2Egz')
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        adata.uns['spatial']['MS377T']['images']['hires'] = imread(f)
    response = requests.get(url + 'tissue%5Flowres%5Fimage%2Epng%2Egz')
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        adata.uns['spatial']['MS377T']['images']['lowres'] = imread(f)
    # Add coordinates
    coords = pd.read_csv(
        url + 'tissue%5Fpositions%5Flist%2Ecsv%2Egz',
        compression='gzip',
        index_col=0,
    )
    adata.obs = adata.obs.join(coords, how='left')
    adata.obsm['spatial'] = adata.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
    adata.obs.drop(
        columns=['in_tissue', 'pxl_row_in_fullres', 'pxl_col_in_fullres'],
        inplace=True,
    )
    # Add metadata
    url_meta = (
        'https://cells-test.gi.ucsc.edu/ms-subcortical-lesions/'
        'visium-ms377T/meta.tsv'
    )
    meta = pd.read_csv(url_meta, sep='\t', usecols=[0, 4], index_col=0)
    adata = adata[meta.index, :].copy()
    adata.obs = adata.obs.join(meta, how='right')
    adata.obs['niches'] = adata.obs['niches'].astype('category')
    adata.obs.index.name = None
    # Filter vars
    msk_var = adata.X.getnnz(axis=0) > 9
    adata = adata[:, msk_var].copy()
    m = f'generated AnnData with shape={adata.shape}'
    _log(m, level='info', verbose=verbose)
    return adata
