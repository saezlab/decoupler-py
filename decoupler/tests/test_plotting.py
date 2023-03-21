import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from anndata import AnnData
from ..plotting import check_if_matplotlib, check_if_seaborn, save_plot, set_limits, plot_volcano, plot_violins
from ..plotting import plot_barplot, build_msks, write_labels, plot_metrics_scatter, plot_metrics_scatter_cols
from ..plotting import plot_metrics_boxplot, plot_psbulk_samples, plot_filter_by_expr, plot_filter_by_prop, plot_volcano_df
from ..plotting import plot_targets


def test_check_if_matplotlib():
    check_if_matplotlib(return_mpl=False)
    check_if_matplotlib(return_mpl=True)


def test_check_if_seaborn():
    check_if_seaborn()


def test_save_plot():
    fig, ax = plt.subplots(1, 1)
    with pytest.raises(AttributeError):
        save_plot(fig, ax, True)


def test_set_limits():
    values = pd.Series([1, 2, 3])
    set_limits(None, None, None, values)


def test_plot_volcano():
    logFCs = pd.DataFrame([[3, 0, -3], [1, 2, -5]], index=['C1', 'C2'], columns=['G1', 'G2', 'G3'])
    logFCs.name = 'contrast_logFCs'
    pvals = pd.DataFrame([[.3, .02, .01], [.9, .1, .003]], index=['C1', 'C2'], columns=['G1', 'G2', 'G3'])
    pvals.name = 'contrast_pvals'
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 1], ['T2', 'G3', 1], ['T2', 'G4', 0.5]],
                       columns=['source', 'target', 'weight'])
    fig, ax = plt.subplots(1, 1)
    plot_volcano(logFCs, pvals, 'C1', name=None, net=None, ax=None, return_fig=True)
    plot_volcano(logFCs, pvals, 'C1', name=None, net=None, ax=ax, return_fig=False)
    plot_volcano(logFCs, pvals, 'C1', name='T1', net=net, ax=None, return_fig=True)


def test_plot_volcano_df():
    data = pd.DataFrame([
        ['G1', 3, 0.04],
        ['G2', -2, 0.03],
        ['G3', 4, 0.01],
        ['G4', 0.5, 0.09],
        ['G5', -0.25, 0.20],
        ['G6', -0.20, 0.27],
    ], columns=['names', 'logFCs', 'pvals']).set_index('names')

    fig, ax = plt.subplots(1, 1)
    plot_volcano_df(data, x='logFCs', y='pvals', ax=None, return_fig=True)
    plot_volcano_df(data, x='logFCs', y='pvals', ax=ax, return_fig=False)


def test_plot_targets():
    data = pd.DataFrame([
        ['G1', 3],
        ['G2', -2],
        ['G3', 4],
        ['G4', 0.5],
        ['G5', -0.25],
        ['G6', -0.20],
    ], columns=['names', 'stat']).set_index('names')

    net = pd.DataFrame([
        ['S1', 'G1', 0.7],
        ['S1', 'G3', 1.5],
        ['S1', 'G4', 1.2],
        ['S1', 'G5', 0.9],
        ['S1', 'G2', -4],
        ['S2', 'G1', 8],
        ['S2', 'G2', -8],
    ], columns=['source', 'target', 'weight'])

    fig, ax = plt.subplots(1, 1)
    plot_targets(data, stat='stat', source_name='S1', net=net, ax=None, return_fig=True)
    plot_targets(data, stat='stat', source_name='S1', net=net, ax=ax, return_fig=False)


def test_plot_violins():
    m = [[1, 2, 3], [4, 5, 6]]
    c = ['G1', 'G2', 'G3']
    r = ['S1', 'S2']
    mat = pd.DataFrame(m, index=r, columns=c)
    plot_violins(mat, thr=1, log=True, use_raw=False, ax=None, title='Title', ylabel='Ylabel', return_fig=True)


def test_plot_barplot():
    estimate = pd.DataFrame([[3.5, -0.5, 0.3], [3.6, -0.6, 0.04], [-1, 2, -1.8]],
                            columns=['T1', 'T2', 'T3'], index=['C1', 'C2', 'C3'])
    plot_barplot(estimate, 'C1', vertical=False, return_fig=False)
    plot_barplot(estimate, 'C1', vertical=True, return_fig=True)


def test_build_msks():
    df = pd.DataFrame([
        ['mlm_estimate', 'mcauroc', 0.7, 'A'],
        ['mlm_estimate', 'mcauprc', 0.7, 'A'],
        ['ulm_estimate', 'mcauroc', 0.6, 'A'],
        ['ulm_estimate', 'mcauprc', 0.6, 'A'],
        ['mlm_estimate', 'mcauroc', 0.5, 'B'],
        ['mlm_estimate', 'mcauprc', 0.5, 'B'],
        ['ulm_estimate', 'mcauroc', 0.4, 'B'],
        ['ulm_estimate', 'mcauprc', 0.4, 'B'],
    ], columns=['method', 'metric', 'score', 'net'])

    msks, cats = build_msks(df, groupby=None)
    assert np.all(msks)
    assert msks[0].size == df.shape[0]
    assert cats[0] is None
    msks, cats = build_msks(df, groupby='net')
    assert len(msks) == 2
    assert not np.all(msks[0])
    assert not np.all(msks[1])
    assert not np.any(msks[0] * msks[1])
    assert cats.size == 2


def test_write_labels():
    fig, ax = plt.subplots(1, 1)
    assert ax.title.get_text() == ''
    assert ax.get_xlabel() == ''
    assert ax.get_ylabel() == ''
    write_labels(ax, title='title', xlabel=None, ylabel=None, x='x', y='y')
    assert ax.title.get_text() == 'title'
    assert ax.get_xlabel() == 'X'
    assert ax.get_ylabel() == 'Y'


def test_plot_metrics_scatter():
    df = pd.DataFrame([
        ['mlm_estimate', 'mcauroc', 0.7, 'A'],
        ['mlm_estimate', 'mcauprc', 0.7, 'A'],
        ['ulm_estimate', 'mcauroc', 0.6, 'A'],
        ['ulm_estimate', 'mcauprc', 0.6, 'A'],
        ['mlm_estimate', 'mcauroc', 0.5, 'B'],
        ['mlm_estimate', 'mcauprc', 0.5, 'B'],
        ['ulm_estimate', 'mcauroc', 0.4, 'B'],
        ['ulm_estimate', 'mcauprc', 0.4, 'B'],
    ], columns=['method', 'metric', 'score', 'net'])

    plot_metrics_scatter(df, x='mcauroc', y='mcauprc', groupby=None, title='title', return_fig=False)
    plot_metrics_scatter(df, x='mcauroc', y='mcauprc', groupby='net', title='title', return_fig=True)


def test_plot_metrics_scatter_cols():
    df = pd.DataFrame([
        ['mlm_estimate', 'mcauroc', 0.7, 'A'],
        ['mlm_estimate', 'mcauprc', 0.7, 'A'],
        ['ulm_estimate', 'mcauroc', 0.6, 'A'],
        ['ulm_estimate', 'mcauprc', 0.6, 'A'],
        ['mlm_estimate', 'mcauroc', 0.5, 'B'],
        ['mlm_estimate', 'mcauprc', 0.5, 'B'],
        ['ulm_estimate', 'mcauroc', 0.4, 'B'],
        ['ulm_estimate', 'mcauprc', 0.4, 'B'],
    ], columns=['method', 'metric', 'score', 'net'])

    plot_metrics_scatter_cols(df, col='net', x='mcauroc', y='mcauprc', groupby='method', return_fig=True)

    df = pd.DataFrame([
        ['mlm_estimate', 'mcauroc', 0.7, 'A', 'TF1'],
        ['mlm_estimate', 'mcauprc', 0.7, 'A', 'TF1'],
        ['ulm_estimate', 'mcauroc', 0.6, 'A', 'TF2'],
        ['ulm_estimate', 'mcauprc', 0.6, 'A', 'TF2'],
        ['mlm_estimate', 'mcauroc', 0.5, 'B', 'TF1'],
        ['mlm_estimate', 'mcauprc', 0.5, 'B', 'TF1'],
        ['ulm_estimate', 'mcauroc', 0.4, 'B', 'TF1'],
        ['ulm_estimate', 'mcauprc', 0.4, 'B', 'TF1'],
    ], columns=['method', 'metric', 'score', 'net', 'source'])

    plot_metrics_scatter_cols(df, col='source', x='mcauroc', y='mcauprc', groupby='net', return_fig=True)
    plot_metrics_scatter_cols(df, col='method', x='mcauroc', y='mcauprc', groupby='net', return_fig=True)


def test_plot_metrics_boxplot():
    df = pd.DataFrame([
        ['mlm_estimate', 'mcauroc', 0.7, 'A'],
        ['mlm_estimate', 'mcauroc', 0.6, 'A'],
        ['ulm_estimate', 'mcauroc', 0.5, 'A'],
        ['ulm_estimate', 'mcauroc', 0.6, 'A'],
        ['mlm_estimate', 'mcauroc', 0.5, 'B'],
        ['mlm_estimate', 'mcauroc', 0.4, 'B'],
        ['ulm_estimate', 'mcauroc', 0.5, 'B'],
        ['ulm_estimate', 'mcauroc', 0.4, 'B'],
    ], columns=['method', 'metric', 'score', 'net'])

    with pytest.raises(ValueError):
        plot_metrics_boxplot(df, 'auroc')
    plot_metrics_boxplot(df, 'mcauroc', groupby='net', title='title', return_fig=True)
    plot_metrics_boxplot(df, 'mcauroc', groupby=None)
    with pytest.raises(ValueError):
        plot_metrics_boxplot(df, 'mcauroc', groupby='method')


def test_plot_psbulk_samples():
    obs = pd.DataFrame([
        ['S1', 'C1', 10, 1023],
        ['S1', 'C2', 12,  956],
        ['S2', 'C1',  7, 1374],
        ['S2', 'C2', 16,  977],
    ], columns=['sample_id', 'cell_type', 'psbulk_n_cells', 'psbulk_counts'], index=['1', '2', '3', '4'])
    var = pd.DataFrame(index=['G1', 'G2'])
    X = np.zeros((4, 2))
    adata = AnnData(X, obs=obs, var=var, dtype=np.float32)

    with pytest.raises(ValueError):
        plot_psbulk_samples(adata, groupby=['cell_type', 'sample_id'], ax='ax')
    plot_psbulk_samples(adata, groupby=['cell_type', 'sample_id'])
    plot_psbulk_samples(adata, groupby='sample_id')


def test_plot_filter_by_expr():
    index = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    columns = ['G1', 'G2', 'G3', 'G4', 'G5']
    # empty, BG, SG
    df = pd.DataFrame([
        [0, 0, 8, 4, 1],
        [0, 0, 8, 2, 1],
        [0, 0, 8, 4, 1],
        [0, 0, 2, 4, 1],
        [0, 8, 0, 2, 1],
        [0, 4, 0, 8, 1],
        [0, 4, 0, 8, 1],
    ], index=index, columns=columns)
    obs = pd.DataFrame([
        ['A'],
        ['A'],
        ['A'],
        ['A'],
        ['B'],
        ['B'],
        ['B'],
    ], index=index, columns=['group'])
    adata = AnnData(df, obs=obs, dtype=np.float32)

    plot_filter_by_expr(adata, obs=None, group=None, lib_size=None, min_count=10,
                        min_total_count=15, large_n=10, min_prop=0.7)
    plot_filter_by_expr(adata, obs=None, group=None, lib_size=40, min_count=1,
                        min_total_count=15, large_n=10, min_prop=0.7)
    plot_filter_by_expr(adata, obs=None, group=None, lib_size=40, min_count=1,
                        min_total_count=15, large_n=10, min_prop=0.7, return_fig=True)

def test_plot_filter_by_prop():
    index = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    columns = ['G1', 'G2', 'G3', 'G4', 'G5']
    # empty, BG, SG
    df = pd.DataFrame([
        [0, 0, 8, 4, 1],
        [0, 0, 8, 2, 1],
        [0, 0, 8, 4, 1],
        [0, 0, 2, 4, 1],
        [0, 8, 0, 2, 1],
        [0, 4, 0, 8, 1],
        [0, 4, 0, 8, 1],
    ], index=index, columns=columns)
    props = pd.DataFrame([
        [0, .0, .8, .4, .1],
        [0, .0, .8, .2, .1],
        [0, .0, .8, .4, .1],
        [0, .0, .2, .4, .1],
        [0, .8, .0, .2, .1],
        [0, .4, .0, .8, .1],
        [0, .4, .0, .8, .1],
    ], index=index, columns=columns, dtype=np.float32)
    obs = pd.DataFrame([
        ['A'],
        ['A'],
        ['A'],
        ['A'],
        ['B'],
        ['B'],
        ['B'],
    ], index=index, columns=['group'])
    adata = AnnData(df, obs=obs, layers={'psbulk_props': props}, dtype=np.float32)

    plot_filter_by_prop(adata, min_prop=0.2, min_smpls=2)
    plot_filter_by_prop(adata, min_prop=0.2, min_smpls=2, return_fig=True)
    with pytest.raises(ValueError):
        plot_filter_by_prop(df, min_prop=0.2, min_smpls=2)
    del adata.layers['psbulk_props']
    with pytest.raises(ValueError):
        plot_filter_by_prop(adata, min_prop=0.2, min_smpls=2)