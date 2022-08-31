import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..method_ora import ora, get_ora_df, run_ora


def test_ora():
    m = csr_matrix(np.array([[1, 0, 2], [1., 0, 3], [0, 0, 0]], dtype=np.float32))
    net = pd.Series([np.array([1, 3], dtype=np.int32), np.array([1, 3], dtype=np.int32)], index=['T1', 'T2'])
    ora(m, net, 1, 0)


def test_get_ora_df():
    df = pd.DataFrame([
        ['GA', 'FA'],
        ['GB', 'FB'],
    ], columns=['groupby', 'features'])
    net = pd.DataFrame([
        ['SA', 'FA'],
        ['SA', 'FC'],
        ['SB', 'FB'],
        ['SB', 'FC'],
    ], columns=['source', 'target'])

    with pytest.raises(ValueError):
        get_ora_df(df, net, groupby='asd', features='features')
    with pytest.raises(ValueError):
        get_ora_df(df, net, groupby='groupby', features='asd')
    res = get_ora_df(df, net, groupby='groupby', features='features', min_n=0)
    assert res.loc['GA', 'SA'] < 0.05
    assert res.loc['GA', 'SB'] > 0.05
    assert res.loc['GB', 'SA'] > 0.05
    assert res.loc['GB', 'SB'] < 0.05


def test_run_ora():
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df)
    net = pd.DataFrame([['T1', 'G2'], ['T1', 'G4'], ['T2', 'G3'], ['T2', 'G1']],
                       columns=['source', 'target'])
    run_ora(adata, net, min_n=0, verbose=True, use_raw=False)
