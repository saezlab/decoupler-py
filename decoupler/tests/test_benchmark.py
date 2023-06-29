import pytest
import numpy as np
import pandas as pd
from ..utils import shuffle_net
from ..utils_benchmark import get_toy_benchmark_data
from ..benchmark import get_performances, format_benchmark_inputs, _benchmark, benchmark


def test_get_performances():
    exps = np.array(['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
    srcs = np.array(['T1', 'T2', 'T3'])
    mthds = np.array(['m_a', 'm_b'])
    m_a = pd.DataFrame([
        [4., 6., 5.],
        [1., 3., 2.],
        [4., 6., 5.],
        [1., 3., 2.],
        [4., 6., 5.],
        [1., 3., 2.]
    ], index=exps, columns=srcs)
    m_b = pd.DataFrame([
        [7., 9., 8.],
        [1., 3., 2.],
        [7., 9., 8.],
        [1., 3., 2.],
        [7., 9., 8.],
        [1., 3., 2.]
    ], index=exps, columns=srcs)
    res = {mthds[0]: m_a, mthds[1]: m_b}
    obs = pd.DataFrame([
            ['A1', 'B1', 'C1', ['T3', 'T2'], 1],
            ['A1', 'B2', 'C1', ['T3', 'T2'], 1],
            ['A1', 'B2', 'C2', ['T3', 'T2'], 1],
            ['A2', 'B1', 'C1', 'T1', 1],
            ['A2', 'B1', 'C2', 'T2', 1],
            ['A2', 'B2', 'C2', 'T3', 1],
        ], columns=['col_A', 'col_B', 'col_C', 'perturb', 'sign'], index=exps)

    get_performances(res, obs, groupby=['col_B'], by='experiment', metrics=['auroc'], min_exp=1)
    get_performances(res, obs, groupby=None, by='source', metrics=['auroc'], min_exp=1)


def test_format_benchmark_inputs():
    mat, net, obs = get_toy_benchmark_data(n_samples=6, shuffle_perc=0)
    decouple_kws = {'source': 'source', 'target': 'target', 'weight': 'weight', 'min_n': 1}

    out = format_benchmark_inputs(mat, obs, perturb='perturb', sign='sign', net=net, groupby='group',
                                  by='experiment', f_expr=True, f_srcs=True, min_n=1, verbose=True, decouple_kws=decouple_kws)
    assert out is not None


def test__benchmark():
    mat, net, obs = get_toy_benchmark_data(n_samples=6, shuffle_perc=0)
    decouple_kws = {'source': 'source', 'target': 'target', 'weight': 'weight', 'min_n': 1}

    out = _benchmark(mat, obs, net, perturb='perturb', sign='sign', metrics=['auroc'], groupby='group',
                     by='experiment', min_exp=1, verbose=True, decouple_kws=decouple_kws)
    assert type(out) is pd.DataFrame


def test_benchmark():
    mat, net, obs = get_toy_benchmark_data(n_samples=6, shuffle_perc=0)

    with pytest.raises(ValueError):
        benchmark(mat, obs, net, perturb='perturb', sign='sign', by='asd')
    with pytest.raises(ValueError):
        benchmark(mat, obs, net, perturb='perturb', sign='sign', pi0=-1)
    df = benchmark(mat, obs, net, metrics=['auroc'], perturb='perturb',
                   sign='sign', decouple_kws={'min_n': 0})
    assert np.all(df.score.values > 0.75)
    df = benchmark(mat, obs, net, metrics=['auroc'], perturb='perturb', sign='sign',
                   groupby='group', decouple_kws={'min_n': 0}, min_exp=1)
    assert np.all(df.score.values > 0.75)
    df = benchmark(mat, obs, net, metrics=['auroc'], perturb='perturb', sign='sign',
                   groupby='perturb', decouple_kws={'min_n': 0}, min_exp=1)
    assert df['group'].unique().size == 4
    assert df['ci'].unique() == 0.2

    rnet = shuffle_net(net, target='target',
                       weight='weight').drop_duplicates(['source', 'target'])
    nets = {'net': net, 'rnet': rnet}
    decouple_kws = {
        'net': {'min_n': 0},
        'rnet': {'min_n': 0}
    }
    df = benchmark(mat, obs, nets, metrics=['auroc'], perturb='perturb', sign='sign',
                   decouple_kws=decouple_kws, min_exp=1)
    msk = df['net'].values == 'net'
    assert np.all(df['score'].values[msk] > df['score'].values[~msk])
