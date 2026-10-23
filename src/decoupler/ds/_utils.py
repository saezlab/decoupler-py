import io

import pandas as pd

from decoupler._download import _download


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

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        dc.ds.ensmbl_to_symbol(genes=["ENSG00000196092", "ENSG00000115415"], organism="hsapiens_gene_ensembl")
    """
    url = (
        'http://{mirror}.ensembl.org/biomart/martservice?query=<?xml version="1.0" encoding="UTF-8"?>'
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
    assert isinstance(genes, list), "genes must be list"
    assert isinstance(organism, str), "organism must be str"
    # Try different mirrors
    error_msgs = ["Service unavailable", "Gateway Time-out"]
    for mirror in ["www", "useast", "uswest", "asia"]:
        try:
            data = _download(url.format(mirror=mirror, organism=organism))
            text = data.read().decode()
            if any(msg in text for msg in error_msgs) or not text.strip():
                continue
            df = pd.read_csv(io.StringIO(text), sep="\t", header=None, index_col=0)
            if df.empty or 1 not in df.columns:
                continue
            eids = df[1].to_dict()
            return [eids[g] if g in eids else None for g in genes]
        except Exception:  # noqa: BLE001
            continue
    # Zenodo fallback for human and mouse
    if organism in ["hsapiens_gene_ensembl", "mmusculus_gene_ensembl"]:
        url = f"https://zenodo.org/records/15551885/files/{organism}.csv.gz?download=1"
        data = _download(url)
        eids = pd.read_csv(data, index_col=0, compression="gzip")["symbol"].to_dict()
        return [eids[g] if g in eids else None for g in genes]
    raise ValueError("ensembl servers are down, try again later")
