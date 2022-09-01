import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..method_gsea import gsea, run_gsea


def test_gsea():
    m = csr_matrix(np.array([[1, 0, 2], [1., 0, 3], [0, 0, 0]], dtype=np.float32))
    net = pd.Series([np.array([1, 3], dtype=np.int32), np.array([1, 3], dtype=np.int32)], index=['T1', 'T2'])
    gsea(m, net, times=2)
    gsea(m, net, times=0)


def test_run_gsea():
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df)
    net = pd.DataFrame([['T1', 'G2'], ['T1', 'G4'], ['T2', 'G3'], ['T2', 'G1']],
                       columns=['source', 'target'])
    run_gsea(adata, net, min_n=0, use_raw=False, times=2, verbose=True)
    run_gsea(df, net, min_n=0, use_raw=False, times=2, verbose=True)
