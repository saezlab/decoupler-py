import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'lst_ens,lst_sym',
    [
        [['ENSG00000196092', 'ENSG00000115415'], ['PAX5', 'STAT1']],
        [['ENSG00000204655', 'ENSG00000184221'], ['MOG', 'OLIG1']],
    ]
)
def test_ensmbl_to_symbol(
    lst_ens,
    lst_sym,
):
    lst_trn = dc.ds.ensmbl_to_symbol(genes=lst_ens)
    assert all(s == t for s, t in zip(lst_trn, lst_sym))
