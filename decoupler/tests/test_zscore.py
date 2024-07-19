import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import norm
from anndata import AnnData
from numpy.testing import assert_almost_equal
from ..method_zscore import zscore, run_zscore

def test_zscore():
    m = csr_matrix(np.array([[-7., -1., 1., 1.], [-4., -2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]], dtype=np.float32))
    net = np.array([[1., 0.], [1, 0.], [0., -1.], [0., -1.]], dtype=np.float32)
    act, pvl = zscore(m, net)
    assert act[0, 0] < 0
    assert act[1, 0] < 0
    assert act[2, 0] > 0
    assert act[3, 0] > 0
    assert np.all((0. <= pvl) * (pvl <= 1.))

def test_run_zscore():
    m = np.array([[7., 1., 1., 1.], [4., 2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3', 'G4'])
    df = pd.DataFrame(m, index=r, columns=c)
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 2], ['T2', 'G3', -3], ['T2', 'G4', 4]],
                       columns=['source', 'target', 'weight'])
    res = run_zscore(df, net, verbose=True, use_raw=False, min_n=0)
    assert res[0].loc['S1', 'T2'] > 0
    assert res[0].loc['S2', 'T2'] < 0
    assert res[0].loc['S3', 'T2'] > 0
    assert res[0].loc['S4', 'T2'] > 0
    assert res[1].map(lambda x: 0 <= x <= 1).all().all()