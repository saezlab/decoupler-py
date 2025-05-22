import pandas as pd
import numpy as np
import pytest

import decoupler as dc


@pytest.mark.parametrize('remove_complexes', [True, False])
def test_collectri(
    remove_complexes,
):
    ct = dc.op.collectri(remove_complexes=remove_complexes)
    assert isinstance(ct, pd.DataFrame)
    cols = {'source', 'target', 'weight', 'resources', 'references', 'sign_decision'}
    assert cols.issubset(ct.columns)
    assert pd.api.types.is_numeric_dtype(ct['weight'])
    msk = np.isin(['AP1', 'NFKB'], ct['source']).all()
    if remove_complexes:
        assert not msk
    else:
        assert msk
    assert not ct.duplicated(['source', 'target']).any()
