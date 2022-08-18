import pytest
import pandas as pd
from anndata import AnnData
from ..utils import m_rename, melt, show_methods, check_corr, get_toy_data, summarize_acts
from ..utils import assign_groups, p_adjust_fdr, dense_run
from ..method_mlm import run_mlm


def test_m_rename():
    e_tmp = pd.DataFrame([['S01', 'T1', 3.51], ['S02', 'T1', 3.53]],
                         columns=['index', 'variable', 'value'])
    m_rename(e_tmp, 'mlm_estimate')
    p_tmp = pd.DataFrame([['S01', 'T1', 0.06], ['S02', 'T1', 0.05]],
                         columns=['index', 'variable', 'value'])
    m_rename(p_tmp, 'mlm_pvals')


def test_melt():
    estimate = pd.DataFrame([[3.5, -0.5, 0.3], [3.6, -0.6, 0.04], [-1, 2, -1.8]],
                    columns=['T1', 'T2', 'T3'], index=['S01', 'S02', 'S03'])
    estimate.name = 'mlm_estimate'
    pvals = pd.DataFrame([[.005, .5, .7], [.006, .6, .9], [.004, .3, .7]],
                            columns=['T1', 'T2', 'T3'], index=['S01', 'S02', 'S03'])
    pvals.name = 'mlm_pvals'
    pvals
    res_dict = {estimate.name: estimate, pvals.name: pvals}
    res_list = [estimate, pvals]
    melt(estimate)
    melt(pvals)
    melt(res_dict)
    melt(res_list)
    with pytest.raises(ValueError):   
        melt({0, 1, 2, 3})


def test_show_methods():

    methods = show_methods()

    assert methods.shape[0] > 0


def test_check_corr():

    mat = pd.DataFrame([[1,2,3,4,5,6]], columns=['G01','G02','G03','G06','G07','G08'])
    net = pd.DataFrame([['T1', 'G01', 1], ['T1', 'G02', 1], ['T2', 'G06', 1], ['T2', 'G07', 0.5],
                        ['T3', 'G06', -0.5], ['T3', 'G07', -3]], columns=['source', 'target', 'weight'])
    check_corr(net, min_n=2)
    check_corr(net, mat=mat, min_n=2)


def test_get_toy_data():

    get_toy_data()


def test_summarize_acts():

    estimate = pd.DataFrame([[3.5, -0.5, 0.3], [3.6, -0.6, 0.04], [-1, 2, -1.8]],
                    columns=['T1', 'T2', 'T3'], index=['S01', 'S02', 'S03'])
    obs = pd.DataFrame([['C01', 'C01', 'C02']], columns=estimate.index, index=['celltype']).T
    adata = AnnData(estimate, obs=obs)
    summarize_acts(estimate, obs=obs, groupby='celltype', mode='median', min_std=0)
    acts = summarize_acts(adata, groupby='celltype', mode='mean', min_std=0)
    assert np.unique(acts.values).size > 2
    with pytest.raises(ValueError):
        summarize_acts(adata, 'celltype', obs, 'mean', 0)


def test_assign_groups():
    estimate = pd.DataFrame([[3.5, -0.5, 0.3], [3.6, -0.6, 0.04], [-1, 2, -1.8]],
                    columns=['T1', 'T2', 'T3'], index=['S01', 'S02', 'S03'])
    obs = pd.DataFrame([['C01', 'C01', 'C02']], columns=estimate.index, index=['celltype']).T
    sum_acts = summarize_acts(estimate, obs=obs, groupby='celltype', min_std=0)
    assign_groups(sum_acts)


def test_p_adjust_fdr():
    p_adjust_fdr([1, 0.8, 0.8, 0.3, 0.01])


def test_denserun():
    mat = pd.DataFrame([[1,2,3,4,5,6]], columns=['G01','G02','G03','G06','G07','G08'])
    net = pd.DataFrame([['T1', 'G01', 1], ['T1', 'G02', 1], ['T2', 'G06', 1], ['T2', 'G07', 0.5],
                        ['T3', 'G06', -0.5], ['T3', 'G07', -3]], columns=['source', 'target', 'weight'])
    dense_run(run_mlm, mat, net, min_n=0)
