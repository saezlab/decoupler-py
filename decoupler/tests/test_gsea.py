import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..method_gsea import gsea, run_gsea, get_gsea_df


def test_get_gsea_df():
    df = pd.DataFrame([
        [7],
        [6],
        [1],
        [-3],
        [-4]
    ], columns=['stat'], index=['G1', 'G2', 'G3', 'G4', 'G5'])
    net = pd.DataFrame([['T1', 'G1'], ['T1', 'G2'], ['T2', 'G3'], ['T2', 'G4']],
                       columns=['source', 'target'])
    res = get_gsea_df(df, 'stat', net, min_n=0, times=10)
    assert res.loc[0, 'NES'] > 0
    assert res.loc[1, 'NES'] < 0
    assert res.loc[0, 'Leading edge'] == 'G2;G1'
    assert res.loc[1, 'Leading edge'] == 'G3;G4'


def test_gsea():
    m = csr_matrix(np.array([[7., 6., 1., -3., -4.]], dtype=np.float32))
    net = pd.Series([np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)], index=['T1', 'T2'])
    es, nes, pval = gsea(m, net, times=10)
    assert nes[0][0] > 0.
    assert nes[0][1] < 0.
    es, nes, pval = gsea(m.A, net, times=10)
    assert nes[0][0] > 0.
    assert nes[0][1] < 0.
    es, nes, pval = gsea(m.A, net, times=0)
    assert (nes is None) & (pval is None)
    assert es[0][0] > 0.
    assert es[0][1] < 0.


def test_run_gsea():
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    net = pd.DataFrame([['T1', 'G2'], ['T1', 'G4'], ['T2', 'G3'], ['T2', 'G1']],
                       columns=['source', 'target'])
    run_gsea(adata, net, min_n=0, use_raw=False, times=10, verbose=True)
    run_gsea(df, net, min_n=0, use_raw=False, times=10, verbose=True)
