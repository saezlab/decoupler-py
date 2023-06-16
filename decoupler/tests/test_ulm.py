import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from numpy.testing import assert_almost_equal
from ..method_ulm import ulm, run_ulm, mat_cov, mat_cor


def test_mat_cov():
    A = np.array([
        [4, 5, 0, 1],
        [6, 3, 1, 0],
        [2, 3, 5, 5]
    ])

    b = np.array([
        [1, 0],
        [1.7, 0],
        [0, 2.5],
        [0, 1]
    ])

    dc_cov = mat_cov(b, A.T)
    np_cov = np.cov(b, A.T, rowvar=False)[:2, 2:].T
    assert_almost_equal(dc_cov, np_cov)


def test_mat_cor():
    A = np.array([
        [4, 5, 0, 1],
        [6, 3, 1, 0],
        [2, 3, 5, 5]
    ])

    b = np.array([
        [1, 0],
        [1.7, 0],
        [0, 2.5],
        [0, 1]
    ])

    dc_cor = mat_cor(b, A.T)
    np_cor = np.corrcoef(b, A.T, rowvar=False)[:2, 2:].T
    assert_almost_equal(dc_cor, np_cor)
    assert np.all((dc_cor <= 1) * (dc_cor >= -1))


def test_ulm():
    m = csr_matrix(np.array([[7., 1., 1., 1.], [4., 2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]], dtype=np.float32))
    net = np.array([[1., 0.], [2, 0.], [0., -3.], [0., 4.]], dtype=np.float32)
    ulm(m, net)


def test_run_ulm():
    m = np.array([[7., 1., 1., 1.], [4., 2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3', 'G4'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df, dtype=np.float32)
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 2], ['T2', 'G3', -3], ['T2', 'G4', 4]],
                       columns=['source', 'target', 'weight'])
    run_ulm(adata, net, verbose=True, use_raw=False, min_n=0)
