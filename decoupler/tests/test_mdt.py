import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..method_mdt import check_if_skranger, fit_rf, mdt, run_mdt


def test_check_if_skranger():
    sr = check_if_skranger()
    assert sr is not None


def test_fit_rf():
    net = np.array([
        [1., 0.],
        [1., 1.],
        [.7, 0.],
        [0., 1.],
        [0., -.5],
        [0., -1.],
    ])
    sample = np.array([7., 6., 1., -3., -4., 0.])
    sr = check_if_skranger()
    a, b = fit_rf(sr, net, sample, min_leaf=2)
    assert a > b


def test_mdt():
    m = csr_matrix(np.array([[7., 6., 1., -3., -4., 0.]]))
    net = np.array([
        [1., 0.],
        [1., 1.],
        [.7, 0.],
        [0., 1.],
        [0., -.5],
        [0., -1.],
    ])
    a, b = mdt(m, net, seed=42, trees=100, min_leaf=2)[0]
    assert a > b
    a, b = mdt(m.A, net, seed=42, trees=100, min_leaf=2)[0]
    assert a > b


def test_run_mdt():
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    net = pd.DataFrame([['T1', 'G2', 1], ['T1', 'G4', 2], ['T2', 'G3', 3], ['T2', 'G1', 1]],
                       columns=['source', 'target', 'weight'])
    run_mdt(adata, net, verbose=True, use_raw=False, min_n=0)
