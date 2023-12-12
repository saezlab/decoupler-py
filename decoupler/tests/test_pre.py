import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..pre import check_mat, extract, filt_min_n, match, rename_net, get_net_mat, mask_features


def test_check_mat():
    m = csr_matrix(np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]]))
    r = np.array(['S1', 'S2', 'S3'])
    c = np.array(['G1', 'G2', 'G3'])
    nm, nr, nc = check_mat(m, r, c, verbose=True)
    assert nm.shape[0] == 2
    assert nm.shape[1] == 2
    assert nr.size == 2
    assert nc.size == 2
    assert type(nm) is csr_matrix
    nm, nr, nc = check_mat(m.A, r, c, verbose=True)
    assert nm.shape[0] == 2
    assert nm.shape[1] == 2
    assert nr.size == 2
    assert nc.size == 2
    assert type(nm) is not csr_matrix and isinstance(nm, np.ndarray)
    with pytest.raises(ValueError):
        check_mat(m, r, np.array(['G1', 'G2', 'G1']))
    m = csr_matrix(np.array([[1, 0, 2], [np.nan, 0, 3], [0, 0, 0]]))
    with pytest.raises(ValueError):
        check_mat(m, r, c)
    m = csr_matrix(np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]]))
    m = m.tocsc()
    nm, nr, nc = check_mat(m, r, c, verbose=True)
    assert isinstance(nm, csr_matrix)


def test_extract():
    m = np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]])
    r = np.array(['S1', 'S2', 'S3'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    adata_raw = adata.copy()
    adata_raw.raw = adata_raw
    extract([m, r, c])
    extract(df)
    extract(adata, use_raw=False)
    extract(adata_raw, use_raw=True)
    with pytest.raises(ValueError):
        extract('asdfg')
    with pytest.raises(ValueError):
        extract(adata, use_raw=True)


def test_filt_min_n():
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 1], ['T2', 'G3', 1], ['T2', 'G4', 0.5]],
                       columns=['source', 'target', 'weight'])
    c = np.array(['G1', 'G2', 'G3'])
    filt_min_n(c, net, min_n=2)
    with pytest.raises(ValueError):
        filt_min_n(c, net, min_n=5)


def test_match():
    c = np.array(['G1', 'G2', 'G3', 'G4'])
    targets = np.array(['G2', 'G1', 'G4', 'G3'])
    net = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 0.5]])
    match(c, targets, net)


def test_rename_net():
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 1], ['T2', 'G3', 1], ['T2', 'G4', 0.5]],
                       columns=['source', 'target', 'weight'])
    rename_net(net)
    rename_net(net, weight=None)
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G1', 1], ['T2', 'G3', 1], ['T2', 'G4', 0.5]],
                       columns=['source', 'target', 'weight'])
    with pytest.raises(ValueError):
        rename_net(net)


def test_get_net_mat():
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 1], ['T2', 'G3', 1], ['T2', 'G4', 0.5]],
                       columns=['source', 'target', 'weight'])
    get_net_mat(net)


def test_mask_features():
    m = np.array([[1, 0, 2], [1, 0, 3], [0, 0, 0]])
    r = np.array(['S1', 'S2', 'S3'])
    c = np.array(['G1', 'G2', 'G3'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    adata_raw = adata.copy()
    adata_raw.raw = adata_raw
    mask_features([m, r, c])
    mask_features([m, r, c], log=True)
    mask_features(df)
    mask_features(adata)
    mask_features(adata_raw, use_raw=True)
    with pytest.raises(ValueError):
        mask_features('asdfg')
    with pytest.raises(ValueError):
        mask_features(adata, use_raw=True)
