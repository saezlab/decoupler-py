from anndata import AnnData

from decoupler._log import _log
from decoupler._download import _download


def hsc_tgfb(
    verbose: bool = False
):
    """
    Downloads RNA-seq bulk data consisting of 6 samples of hepatic stellate cells (HSC) where three of them were activated by
    the cytokine Transforming growth factor (TGF-β).
    
    Reference: GSE151251
    """
    # Download
    url = (
        'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE151251&format=file&'
        'file=GSE151251%5FHSCs%5FCtrl%2Evs%2EHSCs%5FTGFb%2Ecounts%2Etsv%2Egz'
    )
    adata = _download(url, compression='gzip', verbose=verbose)
    # Transform to AnnData
    adata = adata.drop_duplicates('GeneName').set_index('GeneName').iloc[:, 5:].T
    adata.columns.name = None
    adata = AnnData(adata)
    adata.X = adata.X.astype(float)
    # Format obs
    adata.obs['condition'] = ['control' if '-Ctrl' in sample_id else 'treatment' for sample_id in adata.obs.index]
    adata.obs['sample_id'] = [sample_id.split('_')[0] for sample_id in adata.obs.index]
    m = f'generated AnnData with shape={adata.shape}'
    _log(m, level='info', verbose=verbose)
    return adata
