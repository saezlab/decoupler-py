import pytest
import numpy as np
import pandas as pd
import os
from anndata import AnnData
from ..utils import m_rename, melt, show_methods, check_corr, get_toy_data, summarize_acts
from ..utils import assign_groups, p_adjust_fdr, dense_run, shuffle_net, read_gmt
from ..method_mlm import run_mlm
from ..method_ora import run_ora


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

    mat = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=['G01', 'G02', 'G03', 'G06', 'G07', 'G08'])
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
    adata = AnnData(estimate.astype(np.float32), obs=obs)
    acts = summarize_acts(estimate, obs=obs, groupby='celltype', mode='median', min_std=0)
    acts = summarize_acts(adata, obs=None, groupby='celltype', mode='mean', min_std=0)
    assert np.unique(acts.values).size > 2
    with pytest.raises(ValueError):
        summarize_acts(adata, groupby='celltype', obs=obs, mode='mean', min_std=0)
    with pytest.raises(ValueError):
        summarize_acts(adata, groupby='celltype', mode='asd', min_std=0)


def test_assign_groups():
    estimate = pd.DataFrame([[3.5, -0.5, 0.3], [3.6, -0.6, 0.04], [-1, 2, -1.8]],
                            columns=['T1', 'T2', 'T3'], index=['S01', 'S02', 'S03'])
    obs = pd.DataFrame([['C01', 'C01', 'C02']], columns=estimate.index, index=['celltype']).T
    sum_acts = summarize_acts(estimate, obs=obs, groupby='celltype', min_std=0)
    assign_groups(sum_acts)


def test_p_adjust_fdr():
    p_adjust_fdr([1, 0.8, 0.8, 0.3, 0.01])


def test_denserun():
    mat = pd.DataFrame([[0, 2, 3, 4, 5, 6], [1, 0, 0, 0, 0, 0]], columns=['G01', 'G02', 'G03', 'G06', 'G07', 'G08'])
    net = pd.DataFrame([['T1', 'G01', 1], ['T1', 'G02', 1], ['T2', 'G06', 1], ['T2', 'G07', 0.5],
                        ['T3', 'G06', -0.5], ['T3', 'G07', -3]], columns=['source', 'target', 'weight'])
    acts, _ = dense_run(run_mlm, mat, net, min_n=2, verbose=True)
    assert acts.shape[1] == 2
    assert not np.all(np.isnan(acts.values[0]))
    assert np.all(np.isnan(acts.values[1]))

    mat = AnnData(mat.astype(np.float32))
    dense_run(run_ora, mat, net, min_n=2, verbose=True, use_raw=False)
    acts = mat.obsm['ora_estimate']
    assert acts.shape[1] == 2
    assert not np.all(np.isnan(acts.values[0]))
    assert np.all(np.isnan(acts.values[1]))


def test_shuffle_net():
    net = pd.DataFrame([['T1', 'G01', 1], ['T1', 'G02', 1], ['T2', 'G03', 1], ['T2', 'G04', 0.5],
                        ['T3', 'G05', -0.5], ['T3', 'G06', -3]], columns=['source', 'target', 'weight'])

    with pytest.raises(ValueError):
        shuffle_net(net, target=None, weight=None, seed=42, same_seed=True)
    with pytest.raises(ValueError):
        shuffle_net(net, target='asd', weight=None, seed=42, same_seed=True)
    rnet = shuffle_net(net, target='target', weight=None, seed=42, same_seed=True)
    assert np.any(net.target.values != rnet.target.values)
    with pytest.raises(ValueError):
        shuffle_net(net, target=None, weight='asd', seed=42, same_seed=True)
    rnet = shuffle_net(net, target=None, weight='weight', seed=42, same_seed=True)
    assert np.any(net.weight.values != rnet.weight.values)
    rnet = shuffle_net(net, target='target', weight='weight', seed=42, same_seed=True)
    net_dict = {k: v for k, v in zip(net.target, net.weight)}
    rnet_dict = {k: v for k, v in zip(rnet.target, rnet.weight)}
    assert net_dict == rnet_dict
    rnet = shuffle_net(net, target='target', weight='weight', seed=42, same_seed=False)
    net_dict = {k: v for k, v in zip(net.target, net.weight)}
    rnet_dict = {k: v for k, v in zip(rnet.target, rnet.weight)}
    assert net_dict != rnet_dict


def test_read_gmt():
    gmt = """gset_1 link A B C
    gset_2 link C D
    gset_3 link A B C E
    """.rstrip().split('\n')

    # Write line per line
    path = 'tmp.gmt'
    with open(path, 'w') as f:
        for line in gmt:
            f.write(line + '\n')

    # Read gmt
    df = read_gmt(path)

    # Check
    assert df['source'].unique().size == 3
    assert df['target'].unique().size == 5
    counts = df.groupby('source', group_keys=False)['target'].apply(lambda x: np.array(x))
    assert counts['gset_1'].size == 3
    assert counts['gset_2'].size == 2
    assert counts['gset_3'].size == 4

    # Remove tmp file
    os.remove(path)
