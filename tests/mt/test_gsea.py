import numpy as np
import pandas as pd
import pytest

import gseapy as gp
import decoupler as dc


def test_std():
    arr = np.array([0.1, -5.3, 3.8, 9.5, -0.4, 5.5])
    np_std = np.std(arr, ddof=1)
    dc_std = dc.mt._gsea._std(arr=arr, ddof=1)
    assert np_std == dc_std


def test_ridx():
    idx_a = dc.mt._gsea._ridx(times=5, nvar=10, seed=42)
    assert (~(np.diff(idx_a) == 1).all(axis=1)).all()
    idx_b = dc.mt._gsea._ridx(times=5, nvar=10, seed=2)
    assert (~(np.diff(idx_b) == 1).all(axis=1)).all()
    assert (~(idx_a == idx_b).all(axis=1)).all()


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
