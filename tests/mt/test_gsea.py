import numpy as np
import pandas as pd
import pytest

import gseapy as gp
import decoupler as dc


def test_std():
    arr = np.array([0.1, -5.3, 3.8, 9.5, -0.4, 5.5])
    np_std = np.std(arr, ddof=1)
    dc_std = dc.mt._gsea._std.py_func(arr=arr, ddof=1)
    assert np_std == dc_std


def test_ridx():
    idx_a = dc.mt._gsea._ridx(times=5, nvar=10, seed=42)
    assert (~(np.diff(idx_a) == 1).all(axis=1)).all()
    idx_b = dc.mt._gsea._ridx(times=5, nvar=10, seed=2)
    assert (~(np.diff(idx_b) == 1).all(axis=1)).all()
    assert (~(idx_a == idx_b).all(axis=1)).all()


@pytest.mark.parametrize(
    'row,rnks,set_msk,dec,expected_value,expected_index',
    [
        (np.array([0.0, 2.0, 0.0]), np.array([0, 1, 2]), np.array([False, True, False]), 0.1, 0.9, 1),
        (np.array([1.0, 2.0, 3.0]), np.array([2, 1, 0]), np.array([True, True, True]), 0.1, 1.0, 0),
        (np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2]), np.array([False, False, False]), 0.1, 0, 0),
        (np.array([0.0, 0.0, 0.0]), np.array([0, 1, 2]), np.array([True, True, True]), 0.1, 0.0, 0),
        (np.array([1.0, -2.0, 3.0]), np.array([0, 1, 2]), np.array([True, False, True]), 0.5, 0.5, 2),
    ]
)
def test_esrank(
    row,
    rnks,
    set_msk,
    dec,
    expected_value,
    expected_index
):
    value, index, es = dc.mt._gsea._esrank.py_func(row=row, rnks=rnks, set_msk=set_msk, dec=dec)
    assert np.isclose(value, expected_value)
    assert index == expected_index
    assert isinstance(es, np.ndarray) and es.shape == rnks.shape


def test_nesrank(
    rng,
):
    ridx = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 2, 0],
        [1, 0, 2],
        [2, 0, 1],
        [2, 1, 0],
    ])
    row = np.array([0.0, 2.0, 0.0])
    rnks = np.array([0, 1, 2])
    set_msk = np.array([False, True, False])
    dec = 0.1
    es = 0.9
    nes, pval = dc.mt._gsea._nesrank.py_func(
        ridx=ridx,
        row=row,
        rnks=rnks,
        set_msk=set_msk,
        dec=dec,
        es=es
    )
    assert isinstance(nes, float)
    assert isinstance(pval, float)


def test_stsgsea(
    mat,
    idxmat,
):
    X, obs, var = mat
    cnct, starts, offsets = idxmat
    row = X[0, :]
    times = 10
    ridx = dc.mt._gsea._ridx(times=times, nvar=row.size, seed=42)
    es, nes, pv = dc.mt._gsea._stsgsea.py_func(
        row=row,
        cnct=cnct,
        starts=starts,
        offsets=offsets,
        ridx=ridx,
    )
    assert es.size == offsets.size
    assert nes.size == offsets.size
    assert pv.size == offsets.size


def test_func_gsea(
    mat,
    net,
    idxmat,
):
    times = 1000
    seed = 42
    X, obs, var = mat
    gene_sets = net.groupby('source')['target'].apply(lambda x: list(x)).to_dict()
    cnct, starts, offsets = idxmat
    res = gp.prerank(
        rnk=pd.DataFrame(X, index=obs, columns=var).T,
        gene_sets=gene_sets,
        permutation_num=times,
        permutation_type='gene_set',
        outdir=None,
        min_size=0,
        threads=4,
        seed=seed,
    ).res2d
    gp_es = res.pivot(index='Name', columns='Term', values='NES').astype(float)
    gp_pv = res.pivot(index='Name', columns='Term', values='FDR q-val').astype(float)
    dc_es, dc_pv = dc.mt._gsea._func_gsea(
        mat=X,
        cnct=cnct,
        starts=starts,
        offsets=offsets,
        times=times,
        seed=seed,
    )
    assert (gp_es - dc_es).abs().values.max() < 0.10
