import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'organism,lst_ens,lst_sym',
    [
        ['hsapiens_gene_ensembl', ['ENSG00000196092', 'ENSG00000115415'], ['PAX5', 'STAT1']],
        ['hsapiens_gene_ensembl', ['ENSG00000204655', 'ENSG00000184221'], ['MOG', 'OLIG1']],
        ['mmusculus_gene_ensembl', ['ENSMUSG00000076439', 'ENSMUSG00000046160'], ['Mog', 'Olig1']],
    ]
)
def test_ensmbl_to_symbol(
    organism,
    lst_ens,
    lst_sym,
):
    lst_trn = dc.ds.ensmbl_to_symbol(genes=lst_ens, organism=organism)
    assert all(s == t for s, t in zip(lst_trn, lst_sym))
