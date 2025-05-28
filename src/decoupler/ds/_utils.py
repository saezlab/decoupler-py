import requests
import os
import io

import pandas as pd


def ensmbl_to_symbol(
    genes: list,
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
        'iqueRows = "0" count = "" datasetConfigVersion = "0.6" ><Dataset name = "hsapiens_gene_'
        'ensembl" interface = "default" ><Attribute name = "ensembl_gene_id" /><Attribute name ='
        '"external_gene_name" /></Dataset></Query>'
    )
    # Try different mirrors
    response = requests.get(url.format(miror='www'))
    if 'Service unavailable' in response.text:
        response = requests.get(url.format(miror='useast'))
    if 'Service unavailable' in response.text:
        response = requests.get(url.format(miror='asia'))
    assert not 'Service unavailable' in response.text, 'ensembl servers are down, try again later'
    eids = pd.read_csv(io.StringIO(response.text), sep='\t', header=None, index_col=0)[1].to_dict()
    return [eids[g] if g in eids else None for g in genes]
