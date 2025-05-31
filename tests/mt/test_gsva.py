import pandas as pd
import numpy as np
import scipy.sparse as sps
import pytest

import decoupler as dc


"""
gs <- list(
    T1=c('G01', 'G02', 'G03'),
    T2=c('G04', 'G06', 'G07', 'G08'),
    T3=c('G06', 'G07', 'G08'),
    T4=c('G05', 'G10', 'G11', 'G09'),
    T5=c('G09', 'G10', 'G11')
)
mat <- matrix(c(
    0.879, 8.941, 1.951, 8.75, 0.128, 2.959, 2.369, 9.04, 0.853, 3.127, 0.017, 2.859, 0.316, 2.066, 2.05, 8.305, 0.778, 2.468, 1.302, 2.878,
    2.142, 8.155, 0.428, 9.223, 0.532, 2.84, 2.114, 8.681, 0.431, 2.814, 0.413, 3.129, 0.365, 2.512, 2.651, 8.185, 0.406, 2.616, 0.352, 2.824,
    1.729, 0.637, 8.341, 0.74, 8.084, 2.397, 3.093, 0.635, 1.682, 3.351, 1.28, 2.203, 8.556, 2.255, 3.303, 1.25, 1.359, 2.012, 9.784, 2.06,
    0.746, 0.894, 8.011, 1.798, 8.044, 3.059, 2.996, 0.08, 0.151, 2.391, 1.082, 2.123, 8.203, 2.511, 2.039, 0.051, 1.25, 3.787, 8.249, 3.026
), nrow=4, byrow=TRUE)
colnames(mat) <- c('G11', 'G04', 'G05', 'G03', 'G07', 'G18', 'G17', 'G02', 'G10',
       'G14', 'G09', 'G16', 'G08', 'G13', 'G20', 'G01', 'G12', 'G15',
       'G06', 'G19')
rownames(mat) <- c("S01", "S02", "S29", "S30")

gsvaPar <- GSVA::gsvaParam(t(mat), gs, kcdf='Gaussian', maxDiff=TRUE, absRanking=FALSE)
t(GSVA::gsva(gsvaPar, verbose=TRUE))

gsvaPar <- GSVA::gsvaParam(t(round(mat)), gs, kcdf='Poisson', maxDiff=TRUE, absRanking=FALSE)
t(GSVA::gsva(gsvaPar, verbose=TRUE))

gsvaPar <- GSVA::gsvaParam(t(mat), gs, kcdf='none', maxDiff=TRUE, absRanking=FALSE)
t(GSVA::gsva(gsvaPar, verbose=TRUE))

gsvaPar <- GSVA::gsvaParam(t(mat), gs, kcdf='none', maxDiff=FALSE, absRanking=FALSE)
t(GSVA::gsva(gsvaPar, verbose=TRUE))
"""


def test_erf(
    rng,
):
    x = rng.normal(size=10)
    e = dc.mt._gsva._erf.py_func(x=x)
    assert isinstance(e, np.ndarray)

@pytest.mark.parametrize(
    'k,lam',
    [
        [-3, 10],
        [0, 5],
        [3, 50],
    ]
)
def test_poisson_pmf(
    k,
    lam,
):
    p = dc.mt._gsva._poisson_pmf.py_func(k=k, lam=lam)
    assert isinstance(p, float)


def test_ecdf(
    rng
):
    arr = rng.normal(size=10)
    e = dc.mt._gsva._ecdf.py_func(arr)
    assert isinstance(e, np.ndarray)


def test_mat_ecdf(
    rng
):
    arr = rng.normal(size=(5, 10))
    e = dc.mt._gsva._mat_ecdf.py_func(arr)
    assert isinstance(e, np.ndarray)


@pytest.mark.parametrize('gauss', [True, False])
def test_col_d(
    rng,
    gauss,
):
    x = rng.normal(loc=5, size=20)
    pre_cdf = dc.mt._gsva._init_cdfs()
    arr = dc.mt._gsva._col_d.py_func(
        x=x,
        gauss=gauss,
        pre_cdf=pre_cdf
    )
    assert isinstance(arr, np.ndarray)


@pytest.mark.parametrize('gauss', [True, False])
def test_mat_d(
    rng,
    gauss,
):
    x = rng.normal(loc=5, size=(5, 15))
    d = dc.mt._gsva._mat_d.py_func(mat=x, gauss=gauss)
    assert isinstance(d, np.ndarray)


def test_dos_srs(
    rng,
):
    r = np.array(15)
    rng.shuffle(r)
    dos, srs = dc.mt._gsva._dos_srs.py_func(r=r)
    assert isinstance(dos, np.ndarray)
    assert isinstance(srs, np.ndarray)


def test_rankmat(
    rng,
):
    mat = rng.normal(size=(5, 15))
    dos_mat, srs_mat = dc.mt._gsva._rankmat.py_func(mat=mat)
    assert isinstance(dos_mat, np.ndarray)
    assert isinstance(srs_mat, np.ndarray)


@pytest.mark.parametrize(
    "gsetidx, decordstat, symrnkstat, n, tau",
    [
        (np.array([1, 3]), np.array([3, 1, 2, 4]), np.array([0.9, 0.1, 0.8, 0.2]), 4, 1.0),
        (np.array([2, 4]), np.array([1, 3, 2, 4]), np.array([0.5, 0.4, 0.6, 0.3]), 4, 2.0),
        (np.array([1]),    np.array([2, 1, 3]),     np.array([1.0, 0.5, 0.2]),      3, 0.5),
    ]
)
def test_rnd_walk(gsetidx, decordstat, symrnkstat, n, tau):
    k = len(gsetidx)
    pos, neg = dc.mt._gsva._rnd_walk.py_func(
        gsetidx=gsetidx,
        k=k,
        decordstat=decordstat,
        symrnkstat=symrnkstat,
        n=n,
        tau=tau,
    )
    assert isinstance(pos, float)
    assert isinstance(neg, float)
    assert -1.0 <= neg <= 1.0
    assert -1.0 <= pos <= 1.0


@pytest.mark.parametrize(
    "gsetidx, generanking, rankstat, maxdiff, absrnk, tau, expected_range",
    [
        (np.array([1, 3]), np.array([3, 1, 2, 4]), np.array([0.9, 0.1, 0.8, 0.2]), True, True, 1.0, (0.0, 2.0)),
        (np.array([2, 4]), np.array([1, 3, 2, 4]), np.array([0.5, 0.4, 0.6, 0.3]), True, False, 2.0, (-2.0, 2.0)),
        (np.array([1]),    np.array([2, 1, 3]),     np.array([1.0, 0.5, 0.2]),      False, True, 0.5, (-1.0, 1.0)),
    ]
)
def test_score_geneset(gsetidx, generanking, rankstat, maxdiff, absrnk, tau, expected_range):
    es = dc.mt._gsva._score_geneset.py_func(gsetidx, generanking, rankstat, maxdiff, absrnk, tau)
    assert isinstance(es, float)
    assert expected_range[0] <= es <= expected_range[1]
    

def test_init_cdfs():
    cdfs = dc.mt._gsva._init_cdfs.py_func()
    assert np.min(cdfs) == 0.5
    assert np.max(cdfs) == 1.0
    assert np.all(np.diff(cdfs) >= 0)


def test_ppois():
    assert dc.mt._gsva._ppois.py_func(1000, 3) == 1.
    assert np.isclose(dc.mt._gsva._ppois.py_func(100, 75), 0.9975681)
    assert np.isclose(dc.mt._gsva._ppois.py_func(300000, 300000), 0.5004856)
    assert dc.mt._gsva._ppois.py_func(1, 1000) == 0.


def test_norm_cdf():
    assert np.isclose(dc.mt._gsva._norm_cdf.py_func(np.array([1, 2, 3], dtype=float), mu=0.0, sigma=1.0),
                      np.array([0.8413447, 0.9772499, 0.9986501])).all()
    assert np.isclose(dc.mt._gsva._norm_cdf.py_func(np.array([0, 9, 1], dtype=float), mu=0.0, sigma=1.0),
                      np.array([0.5, 1., 0.8413447])).all()


def test_density():
    m = np.array([
        [4, 5, 1, 0],
        [6, 4, 0, 2],
        [8, 8, 3, 1],
        [1, 1, 9, 8],
        [0, 2, 7, 5]
    ])
    d_gauss = dc.mt._gsva._density(m, kcdf='gaussian')
    assert np.all(np.isclose(d_gauss[0, :], np.array([0.006604542, 0.77949374, -0.97600555, -1.9585940])))
    assert np.all(np.isclose(d_gauss[:, 0], np.array([0.006604542, 0.847297859, 2.178640849, -0.960265677, -1.962387678])))
    d_poiss = dc.mt._gsva._density(m, kcdf='poisson')
    assert np.all(np.isclose(d_poiss[0, :], np.array([0.2504135, 0.6946200, -0.74551495, -1.49476762])))
    assert np.all(np.isclose(d_poiss[:, 0], np.array([0.2504135, 0.9572208, 1.7733889, -0.8076761, -1.5963288])))
    d_ecdf = dc.mt._gsva._density(m, kcdf=None)
    print(d_ecdf[0, :])
    assert np.all(np.isclose(d_ecdf[0, :], np.array([0.6, 0.8, 0.4, 0.2])))
    assert np.all(np.isclose(d_ecdf[:, 0], np.array([0.6, 0.8, 1., 0.4, 0.2])))


def test_rankdata():
    arr = np.array([
        0.5, 1., 0.5, 0.75, 0.25, 0.75, 0.5, 1.,
        0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.5,
        1., 0.5, 0.5, 0.5, 0.75
    ])
    dc_rnk = dc.mt._gsva._rankdata.py_func(arr)
    gv_rnk = np.array([
        11, 20, 10, 17, 4, 16, 9, 19, 15, 14, 3, 13, 2, 1, 8, 18, 7, 6, 5, 12
    ])
    assert (dc_rnk == gv_rnk).all()


def test_dos_srs():
    arr = np.array([
        11, 20, 10, 17, 4, 16, 9, 19, 15, 14, 3, 13, 2, 1, 8, 18, 7, 6, 5, 12
    ])
    dc_dos, dc_srs = dc.mt._gsva._dos_srs.py_func(arr)
    gv_dos = np.array([10, 1, 11, 4, 17, 5, 12, 2, 6, 7, 18, 8, 19, 20, 13, 3, 14, 15, 16, 9])
    gv_srs = np.array([1, 10, 0, 7, 6, 6, 1, 9, 5, 4, 7, 3, 8, 9, 2, 8, 3, 4, 5, 2])
    assert (dc_dos == gv_dos).all()
    assert (dc_srs == gv_srs).all()


@pytest.mark.parametrize(
    'kcdf,maxdiff,absrnk,gv_es',
    [
        ['gaussian', True, False,
         np.array([
            [ 0.8823529,  0.009615385, -0.6470588, -0.4485294, -0.5058824],
            [ 0.7058824, -0.414473684, -0.6470588,  0.2215909,  0.6666667],
            [-0.5294118,  0.000000000,  0.3529412,  0.7250000,  0.8823529],
            [-0.5637255,  0.285714286,  0.4460784, -0.5902778, -0.8235294],
         ])],
        ['poisson', True, False,
         np.array([
            [ 0.8823529, -0.4117647, -1.0000000, -0.2467105, 0.04977376],
            [ 1.0000000, -0.5439815, -0.8235294, -0.4545455, 0.04977376],
            [-0.9411765,  0.5000000,  0.8823529,  0.8125000, 0.76470588],
            [-0.9411765,  0.4375000,  0.8235294,  0.5081522, 0.11764706],
         ])],
        [None, True, False,
         np.array([
            [ 0.8823529, -0.1853448, -0.8235294, -0.28846154, -0.3031674],
            [ 0.6470588, -0.4000000, -0.6932773,  0.36250000,  0.7142857],
            [-0.4973262,  0.3486842,  0.5756303,  0.81250000,  0.5882353],
            [-0.7058824,  0.2750000,  0.4117647,  0.07894737, -0.1372549],
         ])],
        [None, False, False,
         np.array([
            [ 0.9411765, -0.5301724, -0.8823529, -0.41346150, -0.4208145],
            [ 0.7058824, -0.4500000, -0.6932773,  0.50000000,  0.7142857],
            [-0.6737968,  0.4111842,  0.6344538,  0.81250000,  0.7647059],
            [-0.7058824,  0.5250000,  0.6470588,  0.26644740, -0.3725490],
         ])],
    ]
)
def test_func_gsva(
    mat,
    idxmat,
    kcdf,
    maxdiff,
    absrnk,
    gv_es,
):  
    X, obs, var = mat
    cnct, starts, offsets = idxmat
    obs = np.array(['S01', 'S02', 'S29', 'S30'])
    X = np.vstack((X[:2, :], X[-2:, :]))
    if kcdf == 'poisson':
        X = X.round()
    X = sps.csr_matrix(X)
    dc_es, _ = dc.mt._gsva._func_gsva(
        mat=X,
        cnct=cnct,
        starts=starts,
        offsets=offsets,
        kcdf=kcdf,
        maxdiff=maxdiff,
        absrnk=absrnk,
    )
    assert np.isclose(dc_es, gv_es).all()
