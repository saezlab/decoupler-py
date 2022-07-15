import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..method_aucell import nb_aucell, aucell, run_aucell


def test_nb_aucell():
    m = csr_matrix(np.array([[1, 0, 2], [1., 0, 3], [0, 0, 0]], dtype=np.float32))
    n_samples, n_features = np.array(m.shape, dtype=np.int32)
    net = np.array([1, 3, 2, 0], dtype=np.int32)
    offsets = np.array([2, 2], dtype=np.int32)
    n_up = np.array([1], dtype=np.int32)[0]
    nb_aucell(n_samples, n_features, m.data, m.indptr, m.indices, net, offsets, n_up)

def test_aucell():
    m = csr_matrix(np.array([[1, 0, 2], [1., 0, 3], [0, 0, 0]], dtype=np.float32))
    net = pd.Series([np.array([1, 3], dtype=np.int32), np.array([1, 3], dtype=np.int32)], index=['T1', 'T2'])
    n_up = np.array([1], dtype=np.int32)[0]
    aucell(m, net, n_up)

def test_run_aucell():
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df)
    net = pd.DataFrame([['T1', 'G2'], ['T1', 'G4'], ['T2', 'G3'], ['T2', 'G1']],
                   columns=['source', 'target'])
    run_aucell(adata, net, n_up=1.2, min_n=0, verbose=True, use_raw=False)
    with pytest.raises(ValueError):
        run_aucell(adata, net, n_up=-3, min_n=0, verbose=True, use_raw=False)
