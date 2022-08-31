import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ..plotting import check_if_matplotlib, check_if_seaborn, save_plot, set_limits, plot_volcano, plot_violins
from ..plotting import plot_barplot, build_msks, write_labels, plot_metrics_scatter, plot_metrics_scatter_cols
from ..plotting import plot_metrics_boxplot


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
