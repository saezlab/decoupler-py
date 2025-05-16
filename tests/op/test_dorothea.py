import pandas as pd
import numpy as np
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'levels,dict_weights',
    [
        ['A', None],
        [['A', 'B'], dict(A=1, B=0.5)],
    ]
)
def test_dorothea(
    levels,
    dict_weights,
):
    do = dc.op.dorothea(levels=levels, dict_weights=dict_weights)
    assert isinstance(do, pd.DataFrame)
    cols = {'source', 'target', 'weight', 'confidence'}
    assert cols.issubset(do.columns)
    assert pd.api.types.is_numeric_dtype(do['weight'])
    assert not do.duplicated(['source', 'target']).any()
