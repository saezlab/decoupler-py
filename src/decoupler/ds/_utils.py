import requests
import os
import io

import pandas as pd


def ensmbl_to_symbol(
    genes: list,
    organism: str,
) -> list:
    """
    Transforms ensembl gene ids to gene symbols.

    Parameters
    ----------
    genes
        List of ensembl gene ids to transform.

    Returns
    -------
    List of gene symbols
    """
    url = (
        'http://www.ensembl.org/biomart/martservice?query=<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE Query><Query  virtualSchemaName = "default" formatter = "TSV" header = "0" un'
        'iqueRows = "0" count = "" ><Dataset name = "{organism}" '
        'interface = "default" ><Attribute name = "ensembl_gene_id" /><Attribute name ='
        '"external_gene_name" /></Dataset></Query>'
    )
    # Organisms
    # hsapiens_gene_ensembl
    # mmusculus_gene_ensembl
    # dmelanogaster_gene_ensembl
    # rnorvegicus_gene_ensembl
    # drerio_gene_ensembl
    # celegans_gene_ensembl
    # scerevisiae_gene_ensembl
    # Validate
    assert isinstance(genes, list), 'genes must be list'
    assert isinstance(organism, str), f'organism must be str'
    # Try different mirrors
    response = requests.get(url.format(miror='www', organism=organism))
    if any(msg in response.text for msg in ['Service unavailable', 'Gateway Time-out']):
        response = requests.get(url.format(miror='useast', organism=organism))
    if any(msg in response.text for msg in ['Service unavailable', 'Gateway Time-out']):
        response = requests.get(url.format(miror='asia', organism=organism))
    if not any(msg in response.text for msg in ['Service unavailable', 'Gateway Time-out']):
        eids = pd.read_csv(io.StringIO(response.text), sep='\t', header=None, index_col=0)[1].to_dict()
    elif organism in ['hsapiens_gene_ensembl', 'mmusculus_gene_ensembl']:
        url = f'https://zenodo.org/records/15551885/files/{organism}.csv.gz?download=1'
        eids = pd.read_csv(url, index_col=0, compression='gzip')['symbol'].to_dict()
    else:
        assert False, 'ensembl servers are down, try again later'
    return [eids[g] if g in eids else None for g in genes]
