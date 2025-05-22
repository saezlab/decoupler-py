import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'methods,args,consensus',
    [
        ['all', dict(), True],
        ['aucell', dict(aucell=dict(n_up=3)), True],
        [['ulm'], dict(), False],
        [['ulm', 'ora'], dict(ulm=dict(), ora=dict(n_up=3)), False]
    ]
)
def test_decouple(
    adata,
    net,
    methods,
    args,
    consensus,
):
    dc.mt.decouple(data=adata, net=net, methods=methods, args=args, cons=consensus, tmin=0)
    if consensus:
        assert 'score_consensus' in adata.obsm
    else:
        assert 'score_consensus' not in adata.obsm
