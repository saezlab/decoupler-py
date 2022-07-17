import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..method_udt import fit_dt, udt, run_udt


def test_fit_dt():
    sample = np.array([7., 1., 1.])
    net = np.array([1. , 1., 0.])
    fit_dt(net, sample)

def test_udt():
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    net = np.array([[1. , 0. ], [1. , 0. ], [0. , 1. ]])
    udt(m, net)

def test_run_mdt():
    m = np.array([[7., 1., 1.], [4., 2., 1.], [1., 2., 5.], [1., 1., 6.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df)
    net = pd.DataFrame([['T1', 'G2', 1], ['T1', 'G4', 2], ['T2', 'G3', 3], ['T2', 'G1', 1]],
                   columns=['source', 'target', 'weight'])
    run_udt(adata, net, verbose=True, use_raw=False, min_n=0)
