import numpy as np
import pytest

import decoupler as dc


@pytest.mark.parametrize("flavor", ["KSEA", "RoKAI"])
def test_func_zscore(
    mat,
    adjmat,
    flavor,
):
    X, obs, var = mat
    es, pv = dc.mt._zscore._func_zscore(mat=X, adj=adjmat, flavor=flavor)
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()


def test_func_zscore_flavor():
    # KSEA subtracts the mean of all features, RoKAI does not
    mat = np.array([[11.0, 12.0, 13.0, 10.0, 9.0, 8.0]])
    adj = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
    std = np.std(mat, ddof=1)
    es_ksea, _ = dc.mt._zscore._func_zscore(mat=mat, adj=adj, flavor="KSEA")
    es_rokai, _ = dc.mt._zscore._func_zscore(mat=mat, adj=adj, flavor="RoKAI")
    np.testing.assert_allclose(es_ksea, [[(12.0 - 10.5) * np.sqrt(3) / std]])
    np.testing.assert_allclose(es_rokai, [[12.0 * np.sqrt(3) / std]])
