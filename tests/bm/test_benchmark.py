import pandas as pd
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'metrics,groupby,runby,sfilt,thr,emin',
    [
        ['auc', None, 'expr', False, 0.05, 5],
        ['auc', None, 'expr', True, 0.05, 5],
        [['auc'], None, 'expr', False, 0.05, 5],
        [['auc', 'fscore'], 'group', 'expr', False, 0.05, 5],
        [['auc', 'fscore', 'qrank'], None, 'source', False, 0.05, 2],
        [['auc', 'fscore', 'qrank'], 'group', 'source', False, 0.05, 1],
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
):
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
            'methods': ['ulm', 'zscore']
        },
    )
    assert isinstance(df, pd.DataFrame)
    cols = {'method', 'metric', 'score'}
    print(df.columns)
    assert cols.issubset(df.columns)
