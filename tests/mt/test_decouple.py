import pytest

import decoupler as dc


@pytest.mark.parametrize(
    "methods,args,cons,anndata",
    [
        ["all", {}, True, True],
        ["aucell", {"aucell": {"n_up": 3}}, True, False],
        [["ulm"], {}, False, True],
        [["ulm", "ora"], {"ulm": {}, "ora": {"n_up": 3}}, False, False],
    ],
)
def test_decouple(adata, net, methods, args, cons, anndata):
    if anndata:
        dc.mt.decouple(data=adata, net=net, methods=methods, args=args, cons=cons, tmin=0)
        if cons:
            assert "score_consensus" in adata.obsm
        else:
            assert "score_consensus" not in adata.obsm
    else:
        res = dc.mt.decouple(data=adata.to_df(), net=net, methods=methods, args=args, cons=cons, tmin=0)
        if cons:
            assert "score_consensus" in res
        else:
            assert "score_consensus" not in res
