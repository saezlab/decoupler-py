import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from anndata import AnnData
from ..method_udt import check_if_sklearn, fit_dt, udt, run_udt


def test_check_if_sklearn():
    sk = check_if_sklearn()
    assert sk is not None


def test_fit_dt():
    net = np.array([
        [1., 0.],
        [1., 1.],
        [.7, 0.],
        [0., 1.],
        [0., -.5],
        [0., -1.],
    ])
    sample = np.array([7., 6., 1., -3., -4., 0.])
    sr = check_if_sklearn()
    a = fit_dt(sr, net[:, 0], sample, seed=42, min_leaf=1)
    b = fit_dt(sr, net[:, 1], sample, seed=42, min_leaf=1)
    assert a > b


def test_udt():
    m = csr_matrix(np.array([[7., 6., 1., -3., -4., 0.]]))
    net = np.array([
        [1., 0.],
        [1., 1.],
        [.7, 0.],
        [0., 1.],
        [0., -.5],
        [0., -1.],
    ])
    a, b = udt(m, net, seed=42, min_leaf=1)[0]
    assert a > b
    a, b = udt(m.A, net, seed=42, min_leaf=1)[0]
    assert a > b


def test_run_mdt():
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df, dtype=np.float32)
    net = pd.DataFrame([['T1', 'G2', 1], ['T1', 'G4', 2], ['T2', 'G3', 3], ['T2', 'G1', 1]],
                       columns=['source', 'target', 'weight'])
    run_udt(adata, net, verbose=True, use_raw=False, min_n=0)
