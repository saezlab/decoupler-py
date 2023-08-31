import pytest
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from ..utils_benchmark import get_toy_benchmark_data, show_metrics, validate_metrics
from ..utils_benchmark import compute_metric, build_acts_tensor, build_grts_mat
from ..utils_benchmark import unique_obs, build_msks_tensor, append_by_experiment
from ..utils_benchmark import append_by_source, append_metrics_scores, check_groupby
from ..utils_benchmark import rename_obs, format_acts_grts, adjust_sign


def test_get_toy_benchmark_data():

    n = 24
    mat, net, obs = get_toy_benchmark_data(n_samples=n, shuffle_perc=0)
    assert mat.shape[0] == n and obs.shape[0] == n
    mean_pos_unsh = np.mean(mat.values[:12, 0])
    mean_neg_unsh = np.mean(mat.values[12:, 0])
    assert mean_pos_unsh > mean_neg_unsh
    mat, net, obs = get_toy_benchmark_data(n_samples=n, shuffle_perc=0.5)
    mean_pos_05 = np.mean(mat.values[:12, 0])
    mean_neg_05 = np.mean(mat.values[12:, 0])
    assert mean_pos_unsh > mean_pos_05
    assert mean_neg_unsh < mean_neg_05
    assert mean_pos_05 > mean_neg_05


def test_show_metrics():
    m = show_metrics()
    r, c = m.shape
    assert r > 0 and c > 0


def test_validate_metrics():
    metrics = 'auroc'
    validate_metrics(metrics)
    metrics = ['auroc', 'auprc', 'mcauroc', 'mcauprc']
    validate_metrics(metrics)
    metrics = ['auroc', 'asd', 'mcauroc', 'mcauprc']
    with pytest.raises(ValueError):
        validate_metrics(metrics)


def test_compute_metric():
    act = [6., 5., 4., 3., 2., 1., 0.]
    grt = [1., 0., 1., 1., 0., 0., 0.]
    metric = 'auroc'
    res = compute_metric(act, grt, metric)
    assert type(res) is np.ndarray
    metric = 'auprc'
    res = compute_metric(act, grt, metric)
    assert type(res) is np.ndarray
    metric = 'mcauroc'
    res = compute_metric(act, grt, metric)
    assert type(res) is np.ndarray
    metric = 'mcauprc'
    res = compute_metric(act, grt, metric)
    assert type(res) is np.ndarray


def test_adjust_sign():
    mat = np.array([
        [1, 2, -3],
        [-1, 0, 4],
        [-5, 0, 6]
        ])
    v_sign = np.array([1, -1, 1])

    adj_m = adjust_sign(mat, v_sign)
    assert np.all(adj_m[1] == (mat[1] * -1))
    assert np.all(adj_m[0] == mat[0])

    adj_m = adjust_sign(csr_matrix(mat), v_sign)
    assert np.all(adj_m[1] == (mat[1] * -1))
    assert np.all(adj_m[0] == mat[0])
    assert isinstance(adj_m, csr_matrix)


def test_build_acts_tensor():
    exps = np.array(['S2', 'S1'])
    srcs = np.array(['T1', 'T3', 'T2'])
    mthds = np.array(['m_a', 'm_b'])

    m_a = pd.DataFrame([
        [4., 6., 5.],
        [1., 3., 2.]
    ], index=exps, columns=srcs)
    m_b = pd.DataFrame([
        [7., 9., 8.],
        [1., 3., 2.]
    ], index=exps, columns=srcs)
    res = {mthds[0]: m_a, mthds[1]: m_b}

    racts, rexps, rsrcs, rmthds = build_acts_tensor(res)
    assert np.all(racts[0, :, 0] == np.array([4., 5., 6.]))
    assert np.all(racts[1, :, 0] == np.array([1., 2., 3.]))
    assert np.all(racts[0, :, 1] == np.array([7., 8., 9.]))
    assert np.all(rexps == np.sort(exps))
    assert np.all(rsrcs == np.sort(srcs))
    assert np.all(rmthds == mthds)


def test_build_grts_mat():
    exps = np.array(['S1', 'S2'])
    srcs = np.array(['T1', 'T2', 'T3'])
    obs = pd.DataFrame([
        [['T3', 'T2'], 1],
        ['T4', -1],
    ], columns=['perturb', 'sign'], index=['S2', 'S1'])

    grts = build_grts_mat(obs, exps, srcs)
    assert np.all(grts.columns == np.array(['T2', 'T3']))
    assert np.all(grts.index == np.array(['S1', 'S2']))
    assert np.all(grts.values == np.array([[0., 0.], [1., 1.]]))


def test_unique_obs():
    col = ['T1', 'T1', 'T2', 'T2']
    assert len(unique_obs(col)) == 2

    col = ['T1', ['T1', 'T2'], 'T3', 'T2']
    assert len(unique_obs(col)) == 3

    col = [['T1', 'T2'], ['T1', 'T2'], ['T3', 'T4'], ['T3', 'T4']]
    assert len(unique_obs(col)) == 4


def test_build_msks_tensor():
    obs = pd.DataFrame([
            ['A1', 'B1', 'C1'],
            ['A1', 'B2', 'C1'],
            ['A1', 'B2', 'C2'],
            ['A2', 'B1', 'C1'],
            ['A2', 'B1', 'C2'],
            ['A2', 'B2', 'C2'],
        ], columns=['col_A', 'col_B', 'col_C'])

    msks, grpbys, grps = build_msks_tensor(obs, groupby=['col_C'])
    assert np.all(msks[0][0] == np.array([True, True, False, True, False, False]))
    assert grpbys[0] == 'col_C'
    assert np.all(grps == np.array(['C1', 'C2']))
    msks, grpbys, grps = build_msks_tensor(obs, groupby=[['col_A']])
    assert np.all(msks[0][0] == np.array([True, True, True, False, False, False]))
    assert grpbys[0] == 'col_A'
    assert np.all(grps == np.array(['A1', 'A2']))
    msks, grpbys, grps = build_msks_tensor(obs, groupby=[['col_A', 'col_B']])
    assert np.all(msks[0][0] == np.array([True, False, False, False, False, False]))
    assert np.all(msks[0][1] == np.array([False, True, True, False, False, False]))
    assert grpbys[0] == 'col_A|col_B'
    assert np.all(grps == np.array(['A1|B1', 'A1|B2', 'A2|B1', 'A2|B2']))
    msks, grpbys, grps = build_msks_tensor(obs, groupby=None)
    assert msks is None and grpbys is None and grps is None


def test_append_by_experiment():
    act = np.array([
        [[2, 5],
         [1, 4],
         [2, 3],
         [4, 2],
         [5, 1]],
        [[5, 1],
         [4, 2],
         [3, 3],
         [1, 5],
         [2, 4]],
        [[2, 5],
         [1, 4],
         [2, 3],
         [4, 2],
         [5, 1]]
    ])
    grt = np.array([
            [1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0.]
    ])
    srcs = np.array(['T1', 'T2', 'T3', 'T4', 'T5'])
    mthds = np.array(['M1', 'M2'])
    metrics = ['auroc']
    df = []

    append_by_experiment(df, grpby_i=None, grp=None, act=act, grt=grt, srcs=srcs,
                         mthds=mthds, metrics=metrics, min_exp=1)

    act_na = act.astype(float)
    act_na[1, 0, 0] = np.nan
    act_na[1, 1, 0] = np.nan
    act_na[1, 2, 0] = np.nan

    df_na = []

    append_by_experiment(df_na, grpby_i=None, grp=None, act=act_na, grt=grt, srcs=srcs,
                         mthds=mthds, metrics=metrics, min_exp=1)

    assert len(df) == 2
    assert df[0][5] < df[1][5]
    assert df[0][5] < df_na[0][5]  # check improvement of performance due to removal of NAs
    assert df[0][6] < df_na[0][6]  # check change of class imbalance due to removal of NAs


def test_append_by_source():
    act = np.array([
        [[2, 5],
         [1, 4],
         [2, 3],
         [4, 2],
         [5, 1]],
        [[5, 1],
         [4, 2],
         [3, 3],
         [1, 5],
         [2, 4]],
        [[2, 5],
         [1, 4],
         [2, 3],
         [4, 2],
         [5, 1]]
    ])
    grt = np.array([
            [1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0.]
    ])
    srcs = np.array(['T1', 'T2', 'T3', 'T4', 'T5'])
    mthds = np.array(['M1', 'M2'])
    metrics = ['auroc']
    df = []
    append_by_source(df, grpby_i=None, grp=None, act=act, grt=grt, srcs=srcs,
                     mthds=mthds, metrics=metrics, min_exp=1)
    assert len(df) == 4
    assert df[0][5] < df[2][5]

    act_na = act.astype(float)
    act_na[1, 4, 0] = np.nan

    df_na = []

    append_by_source(df_na, grpby_i=None, grp=None, act=act_na, grt=grt, srcs=srcs,
                     mthds=mthds, metrics=metrics, min_exp=1)

    assert len(df_na) == 3
    assert df_na[0][2] == 'T1'

    act_na[1, 0, 0] = np.nan

    df_na_2 = []
    append_by_source(df_na_2, grpby_i=None, grp=None, act=act_na, grt=grt, srcs=srcs,
                     mthds=mthds, metrics=metrics, min_exp=1)

    assert len(df_na_2) == 2

    act_na_3 = act.astype(float)
    act_na_3[1, 0, 0] = np.nan

    df_na_3 = []

    append_by_source(df_na_3, grpby_i=None, grp=None, act=act_na_3, grt=grt, srcs=srcs,
                     mthds=mthds, metrics=metrics, min_exp=1)

    assert len(df_na_3) == 3
    assert df_na_3[0][2] == 'T5'


def test_append_metrics_scores():
    act = np.array([
        [[2, 5],
         [1, 4],
         [2, 3],
         [4, 2],
         [5, 1]],
        [[5, 1],
         [4, 2],
         [3, 3],
         [1, 5],
         [2, 4]],
        [[2, 5],
         [1, 4],
         [2, 3],
         [4, 2],
         [5, 1]]
    ])
    grt = np.array([
            [1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0.]
    ])
    srcs = np.array(['T1', 'T2', 'T3', 'T4', 'T5'])
    mthds = np.array(['M1', 'M2'])
    metrics = ['auroc']
    by = 'experiment'
    df = []

    with pytest.raises(ValueError):
        append_metrics_scores(df, grpby_i=None, grp=None, act=act, grt=grt, srcs=srcs,
                              mthds=mthds, metrics=metrics, by=by, min_exp=0)
    df = []
    append_metrics_scores(df, grpby_i=None, grp=None, act=act, grt=grt, srcs=srcs,
                          mthds=mthds, metrics=metrics, by=by, min_exp=1)
    assert len(df) == 2
    assert df[0][5] < df[1][5]
    df = []
    append_metrics_scores(df, grpby_i=None, grp=None, act=act, grt=grt, srcs=srcs,
                          mthds=mthds, metrics=metrics, by='source', min_exp=1)
    assert len(df) == 4
    assert df[0][5] < df[2][5]


def test_check_groupby():

    obs = pd.DataFrame([
        ['A1', 'B1', 'C1'],
        ['A1', 'B2', 'C1'],
        ['A1', 'B2', 'C2'],
        ['A2', 'B1', 'C1'],
        ['A2', 'B1', 'C2'],
        ['A2', 'B2', 'C2'],
    ], columns=['A', 'B', 'C'])
    perturb = 'A'

    check_groupby(obs=obs, groupby='A', perturb=perturb, by='experiment')
    with pytest.raises(AssertionError):
        check_groupby(obs=obs, groupby='A', perturb=perturb, by='source')
    with pytest.raises(AssertionError):
        check_groupby(obs=obs, groupby='D', perturb=perturb, by='experiment')

    check_groupby(obs=obs, groupby=['A', 'B'], perturb=perturb, by='experiment')
    with pytest.raises(AssertionError):
        check_groupby(obs=obs, groupby=['A', 'B'], perturb=perturb, by='source')
    with pytest.raises(AssertionError):
        check_groupby(obs=obs, groupby=['D', 'B'], perturb=perturb, by='experiment')

    check_groupby(obs=obs, groupby=['A', ['B', 'C']], perturb=perturb, by='experiment')
    with pytest.raises(AssertionError):
        check_groupby(obs=obs, groupby=['A', ['B', 'C']], perturb=perturb, by='source')
    with pytest.raises(AssertionError):
        check_groupby(obs=obs, groupby=['D', ['B', 'C']], perturb=perturb, by='experiment')
    obs = obs.rename({'A': 'A|Z'}, axis=1)
    with pytest.raises(AssertionError):
        check_groupby(obs=obs, groupby=['A|Z'], perturb=perturb, by='experiment')


def test_rename_obs():
    meta = pd.DataFrame([
        ['TF1', -1],
        [np.array(['TF1', 'TF2']), 0],
        ['TF2', 1]
    ], columns=['perturb', 'sign'])

    with pytest.raises(AssertionError):
        rename_obs(meta, 'asd', 'sign')
    with pytest.raises(ValueError):
        rename_obs(meta, 'perturb', 'perturb')
    with pytest.raises(AssertionError):
        rename_obs(meta, 'perturb', 'asd')
    with pytest.raises(AssertionError):
        rename_obs(meta, 'perturb', 'sign')
    rename_obs(meta, 'perturb', -1)
    with pytest.raises(ValueError):
        rename_obs(meta, 'perturb', 0)


def test_format_acts_grts():
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
            ['A1', 'B2', 'C1', ['T3', 'T2'], -1],
            ['A1', 'B2', 'C2', ['T3', 'T2'], 1],
            ['A2', 'B1', 'C1', 'T1', -1],
            ['A2', 'B1', 'C2', 'T2', 1],
            ['A2', 'B2', 'C2', 'T3', -1],
        ], columns=['col_A', 'col_B', 'col_C', 'perturb', 'sign'], index=exps)

    out = format_acts_grts(res, obs, groupby=['col_C'])
    assert out is not None
