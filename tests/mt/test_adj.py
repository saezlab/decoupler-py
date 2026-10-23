from __future__ import annotations

import numpy as np
import scipy.stats as sts

import decoupler as dc
from decoupler.mt._pv import _fdr_bh_axis1_numba


def test_func_mlm(
    adata,
    net,
):
    dc.mt.mlm(data=adata, net=net, tmin=3)
    dc_pv = adata.obsm["padj_mlm"]
    adj = _fdr_bh_axis1_numba(dc_pv.values)
    np.testing.assert_allclose(adj, sts.false_discovery_control(dc_pv.values, axis=1, method="bh"))
