import pytest
import tempfile
import anndata as ad
import numpy as np

import decoupler as dc


@pytest.mark.parametrize(
    "methods,args,cons,anndata",
    [
        ["all", {}, True, True],
        ["aucell", {"aucell": {"n_up": 3}}, True, False],
        [["ulm"], {}, False, True],
        [["ulm", "ora"], {"ulm": {}, "ora": {"n_up": 3}}, False, False],
        [["ulm", "ora"], {"ulm": {}, "ora": {"n_up": 3}}, False, True],
    ],
)
def test_decouple(adata, net, methods, args, cons, anndata):
    if anndata:
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=True) as tf:
            adata.write(tf.name)
            bdata = ad.read_h5ad(tf.name, backed='r')
            dc.mt.decouple(data=bdata, net=net, methods=methods, args=args, cons=cons, tmin=0)
            dc.mt.decouple(data=adata, net=net, methods=methods, args=args, cons=cons, tmin=0)
            if cons:
                assert "score_consensus" in adata.obsm
            else:
                assert "score_consensus" not in adata.obsm
            assert set(adata.obsm).issubset(bdata.obsm)
            for k in adata.obsm:
                if 'waggr' not in k and 'consensus' not in k:  # has different seed, cannot compare
                    print(k)
                    assert np.allclose(adata.obsm[k].values, bdata.obsm[k].values)
    else:
        res = dc.mt.decouple(data=adata.to_df(), net=net, methods=methods, args=args, cons=cons, tmin=0)
        if cons:
            assert "score_consensus" in res
        else:
            assert "score_consensus" not in res
