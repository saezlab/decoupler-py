import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..method_gsva import init_cdfs, density, gsva, run_gsva


def test_init_cdfs():
    init_cdfs(pre_res=10000, max_pre=10)


def test_density():
    m = csr_matrix(np.array([[1, 0, 2], [1., 0, 3], [0, 0, 0]], dtype=np.float32)).A
    density(m, kcdf=True)
    density(m, kcdf=False)


def test_gsva():
    m = csr_matrix(np.array([[1, 0, 2], [1., 0, 3], [0, 0, 0]], dtype=np.float32)).A
    net = pd.Series([np.array([1, 3], dtype=np.int64), np.array([1, 3], dtype=np.int64)], index=['T1', 'T2'])
    gsva(m, net)


def test_run_gsva():
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df, dtype=np.float32)
    net = pd.DataFrame([['T1', 'G2'], ['T1', 'G4'], ['T2', 'G3'], ['T2', 'G1']],
                       columns=['source', 'target'])
    run_gsva(adata, net, min_n=0, use_raw=False, verbose=True)
