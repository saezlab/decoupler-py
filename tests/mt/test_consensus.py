import numpy as np
import pytest

import decoupler as dc


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
