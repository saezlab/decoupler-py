import numpy as np
import pytest

import decoupler as dc


@pytest.mark.parametrize('sel', [np.array([0., 0., 0., 0.]), np.array([1., 3., 8., 2.])])
def test_zscore(
    sel,
):
    z = dc.mt._consensus._zscore.py_func(sel=sel)
    assert isinstance(z, np.ndarray)
    assert z.size  == sel.size


def test_mean_zscores(
    rng,
):
    scores = rng.normal(size=(2, 5, 10))
    es = dc.mt._consensus._mean_zscores.py_func(scores=scores)
    assert scores.shape[1:] == es.shape
    

def test_consensus(
    adata,
    net,
):
    dc.mt.decouple(data=adata, net=net, methods=['zscore', 'ulm'], cons=False, tmin=0)
    dc.mt.consensus(adata)
    assert 'score_consensus' in adata.obsm
    res = dc.mt.decouple(data=adata.to_df(), net=net, methods=['zscore', 'ulm'], cons=False, tmin=0)
    es, pv = dc.mt.consensus(res)
    assert np.isfinite(es.values).all()
    assert ((0 <= pv.values) & (pv.values <= 1)).all() 
