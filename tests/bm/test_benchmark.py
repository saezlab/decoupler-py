import pandas as pd
import scipy.sparse as sps
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'metrics,groupby,runby,sfilt,thr,emin,mnet',
    [
        ['auc', None, 'expr', False, 0.05, 5, False],
        ['auc', None, 'expr', True, 0.05, 5, False],
        [['auc'], None, 'expr', False, 0.05, 5, False],
        [['auc', 'fscore'], 'group', 'expr', False, 0.05, 5, False],
        [['auc', 'fscore', 'qrank'], None, 'source', False, 0.05, 2, False],
        [['auc', 'fscore', 'qrank'], 'group', 'source', False, 0.05, 1, False],
        [['auc', 'fscore', 'qrank'], 'bm_group', 'expr', True, 0.05, 5, False],
        [['auc', 'fscore', 'qrank'], 'source', 'expr', True, 0.05, 5, False],
    ]
)
def test_benchmark(
    bdata,
    net,
    metrics,
    groupby,
    runby,
    sfilt,
    thr,
    emin,
    mnet,
    rng,
):
    dc.mt.ulm(data=bdata, net=net, tmin=0)
    if mnet:
        net = {'w_net': net, 'unw_net': net.drop(columns=['weight'])}
        bdata = bdata.copy()
        bdata.obs['source'] = rng.choice(['x', 'y', 'z'], size=bdata.n_obs, replace=True)
        bdata.X = sps.csr_matrix(bdata.X)
    df = dc.bm.benchmark(
        adata=bdata,
        net=net,
        metrics=metrics,
        groupby=groupby,
        runby=runby,
        sfilt=sfilt,
        thr=thr,
        emin=emin,
        kws_decouple={
            'cons': True,
            'tmin': 3,
            'methods': ['ulm', 'zscore', 'aucell']
        },
        verbose=True
    )
    assert isinstance(df, pd.DataFrame)
    cols = {'method', 'metric', 'score'}
    assert cols.issubset(df.columns)
    hdf = dc.bm.metric.hmean(df, metrics=metrics)
    assert isinstance(hdf, pd.DataFrame)
