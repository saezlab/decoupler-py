import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..method_gsva import init_cdfs, density, ppois, norm_cdf, gsva, run_gsva


def test_init_cdfs():
    cdfs = init_cdfs()
    assert np.min(cdfs) == 0.5
    assert np.max(cdfs) == 1.0
    assert np.all(np.diff(cdfs) >= 0)


def test_density():
    m = np.array([
        [4, 5, 1, 0],
        [6, 4, 0, 2],
        [8, 8, 3, 1],
        [1, 1, 9, 8],
        [0, 2, 7, 5]
    ], dtype=np.float32)
    d_gauss = density(m, kcdf='gaussian')
    assert np.all(np.isclose(d_gauss[0, :], np.array([0.006604542, 0.77949374, -0.97600555, -1.9585940])))
    assert np.all(np.isclose(d_gauss[:, 0], np.array([0.006604542, 0.847297859, 2.178640849, -0.960265677, -1.962387678])))
    d_poiss = density(m, kcdf='poisson')
    assert np.all(np.isclose(d_poiss[0, :], np.array([0.2504135, 0.6946200, -0.74551495, -1.49476762])))
    assert np.all(np.isclose(d_poiss[:, 0], np.array([0.2504135, 0.9572208, 1.7733889, -0.8076761, -1.5963288])))
    d_ecdf = density(m, kcdf=None)
    assert np.all(np.isclose(d_ecdf[0, :], np.array([0.6, 0.8, 0.4, 0.2])))
    assert np.all(np.isclose(d_ecdf[:, 0], np.array([0.6, 0.8, 1., 0.4, 0.2])))
    with pytest.raises(ValueError):
        density(m, kcdf='asd')


def test_ppois():
    assert ppois(1000, 3) == 1.
    assert np.isclose(ppois(100, 75), 0.9975681)
    assert np.isclose(ppois(300000, 300000), 0.5004856)
    assert ppois(1, 1000) == 0.


def test_norm_cdf():
    assert np.isclose(norm_cdf(np.array([1, 2, 3], dtype=float), mu=0.0, sigma=1.0), np.array([0.8413447, 0.9772499, 0.9986501])).all()
    assert np.isclose(norm_cdf(np.array([0, 9, 1], dtype=float), mu=0.0, sigma=1.0), np.array([0.5, 1., 0.8413447])).all()


def test_gsva():
    m = np.array([
        [4, 5, 1, 0],
        [6, 4, 0, 2],
        [8, 8, 3, 1],
        [1, 1, 9, 8],
        [0, 2, 7, 5]
    ], dtype=np.float32)
    net = pd.Series([
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64)
    ], index=['T1', 'T2'])
    res = gsva(m, net, kcdf=None)
    assert np.isclose(res[0, :], np.array([1, 0.3333333])).all()
    assert np.isclose(res[:, 0], np.array([1, 0.5, 1, 0.3333333, 0.3333333])).all()


def test_run_gsva():
    m = np.array([
        [4, 5, 1, 0],
        [6, 4, 0, 2],
        [8, 8, 3, 1],
        [1, 1, 9, 8],
        [0, 2, 7, 5]
    ], dtype=np.float32)
    r = np.array(['S1', 'S2', 'S3', 'S4', 'S5'])
    c = np.array(['G1', 'G2', 'G3', 'G4'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    adata.X = csr_matrix(adata.X)
    net = pd.DataFrame([
        ['T1', 'G1'],
        ['T1', 'G2'],
        ['T1', 'G3'],
        ['T2', 'G2'],
        ['T2', 'G3'],
        ['T2', 'G4'],
    ], columns=['source', 'target'])
    run_gsva(adata, net, min_n=0, use_raw=False, verbose=True, kcdf='gaussian')
    res_adata = adata.obsm['gsva_estimate'].values
    assert np.isclose(res_adata[0, :], np.array([1, 0.3333333])).all()
    assert np.isclose(res_adata[:, 0], np.array([1, 0.5, 1, -1, 0.3333333])).all()
    res_df = run_gsva(df, net, min_n=0, use_raw=False, verbose=True, kcdf='poisson')[0]
    assert np.all(res_adata == res_df.values)
    res_lst = run_gsva([m, r, c], net, min_n=0, use_raw=False, verbose=True, kcdf=None)[0].values
    assert np.isclose(res_lst[0, :], np.array([1, 0.3333333])).all()
    assert np.isclose(res_lst[:, 0], np.array([1, 0.3333333, 1, 0.3333333, 0.3333333])).all()
