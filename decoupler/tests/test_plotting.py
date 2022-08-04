import pytest
import matplotlib.pyplot as plt
import pandas as pd
from ..plotting import check_if_matplotlib, check_if_seaborn, save_plot, set_limits, plot_volcano, plot_violins, plot_barplot


def test_check_if_matplotlib():
    check_if_matplotlib(return_mpl=False)
    check_if_matplotlib(return_mpl=True)

def test_check_if_seaborn():
    check_if_seaborn()

def test_save_plot():
    fig, ax = plt.subplots(1,1)
    with pytest.raises(AttributeError):
        save_plot(fig, ax, True)

def test_set_limits():
    values = pd.Series([1,2,3])
    set_limits(None, None, None, values)

def test_plot_volcano():
    logFCs = pd.DataFrame([[3, 0, -3], [1, 2, -5]], index=['C1', 'C2'], columns=['G1', 'G2', 'G3'])
    logFCs.name = 'contrast_logFCs'
    pvals = pd.DataFrame([[.3, .02, .01], [.9, .1, .003]], index=['C1', 'C2'], columns=['G1', 'G2', 'G3'])
    pvals.name = 'contrast_pvals'
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 1], ['T2', 'G3', 1], ['T2', 'G4', 0.5]],
                       columns=['source', 'target', 'weight'])
    fig, ax = plt.subplots(1,1)
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
