import pytest
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..method_ora import ora, get_ora_df, run_ora, extract_c


def test_ora():
    m = csr_matrix(np.array([[1, 0, 2], [1., 0, 3], [0, 0, 0]], dtype=np.float32))
    net = pd.Series([np.array([1, 3], dtype=np.int64), np.array([1, 3], dtype=np.int64)], index=['T1', 'T2'])
    ora(m, net, 1, 0)


def test_extract_c():
    df = np.array(['G1', 'G2'], dtype='U')
    c = extract_c(pd.DataFrame(index=df))
    assert np.all(df == c)
    c = extract_c(list(df))
    assert np.all(df == c)
    c = extract_c(df)
    assert np.all(df == c)
    c = extract_c(pd.Index(df))
    assert np.all(df == c)
    with pytest.raises(ValueError):
        extract_c('asd')


def test_get_ora_df():
    df = pd.DataFrame([], index=['G1', 'G2', 'G3', 'G4', 'G5', 'G8', 'G9'])
    net = pd.DataFrame([
        ['T1', 'G1'],
        ['T1', 'G2'],
        ['T1', 'G3'],
        ['T2', 'G3'],
        ['T2', 'G4'],
        ['T2', 'G6'],
        ['T3', 'G5'],
        ['T3', 'G6'],
        ['T3', 'G7'],
    ], columns=['source', 'target'])

    res = get_ora_df(df, net, n_background=20000, verbose=True)
    assert_almost_equal(res.loc[0, 'Overlap ratio'], 1)
    assert_almost_equal(res.loc[0, 'p-value'], 2.625394e-11)
    assert_almost_equal(res.loc[0, 'FDR p-value'], 7.876183e-11)
    assert res.loc[0, 'Features'] == 'G1;G2;G3'
    res = get_ora_df(df, net, n_background=None, verbose=True)
    assert_almost_equal(res.loc[0, 'p-value'], 0.2857143)


def test_run_ora():
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    net = pd.DataFrame([['T1', 'G2'], ['T1', 'G4'], ['T2', 'G3'], ['T2', 'G1']],
                       columns=['source', 'target'])
    run_ora(adata, net, min_n=0, verbose=True, use_raw=False)
