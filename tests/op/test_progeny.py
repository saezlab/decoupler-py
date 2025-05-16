import pandas as pd
import numpy as np
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'top,thr_padj',
    [
        [100, 0.05],
        [100, 1],
        [np.inf, 0.05],
        [np.inf, 1],
    ]
)
def test_progeny(
    top,
    thr_padj,
):
    pr = dc.op.progeny(top=top, thr_padj=thr_padj)
    assert isinstance(pr, pd.DataFrame)
    cols = {'source', 'target', 'weight', 'padj'}
    assert cols.issubset(pr.columns)
    assert pd.api.types.is_numeric_dtype(pr['weight'])
    assert pd.api.types.is_numeric_dtype(pr['padj'])
    assert (pr['padj'] < thr_padj).all()
    assert (pr.groupby('source').size() <= top).all()
    assert not pr.duplicated(['source', 'target']).any()
