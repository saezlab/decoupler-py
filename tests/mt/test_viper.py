import numpy as np
import pandas as pd
import pytest

import decoupler as dc


"""
mat <- matrix(c(
    0.879, 8.941, 1.951, 8.75, 0.128, 2.959, 2.369, 9.04, 0.853, 3.127, 0.017, 2.859, 0.316, 2.066, 2.05, 8.305, 0.778, 2.468, 1.302, 2.878,
    2.142, 8.155, 0.428, 9.223, 0.532, 2.84, 2.114, 8.681, 0.431, 2.814, 0.413, 3.129, 0.365, 2.512, 2.651, 8.185, 0.406, 2.616, 0.352, 2.824,
    1.729, 0.637, 8.341, 0.74, 8.084, 2.397, 3.093, 0.635, 1.682, 3.351, 1.28, 2.203, 8.556, 2.255, 3.303, 1.25, 1.359, 2.012, 9.784, 2.06,
    0.746, 0.894, 8.011, 1.798, 8.044, 3.059, 2.996, 0.08, 0.151, 2.391, 1.082, 2.123, 8.203, 2.511, 2.039, 0.051, 1.25, 3.787, 8.249, 3.026
), nrow=4, byrow=TRUE)
colnames(mat) <- c('G11', 'G04', 'G05', 'G03', 'G07', 'G18', 'G17', 'G02', 'G10',
       'G14', 'G09', 'G16', 'G08', 'G13', 'G20', 'G01', 'G12', 'G15',
       'G06', 'G19')
rownames(mat) <- c("S01", "S02", "S29", "S30")
gs <- list(
    T1 = list(
        tfmode = c(G01 = 1, G02 = 1, G03 = 0.7, G04 = 1, G06 = -0.5, G07 = -3, G08 = -1),
        likelihood = c(1, 1, 1, 1, 1, 1, 1)
    ),
    T2 = list(
        tfmode = c(G06 = 1, G07 = 0.5, G08 = 1, G05 = 1.9, G10 = -1.5, G11 = -2, G09 = 3.1),
        likelihood = c(1, 1, 1, 1, 1, 1, 1)
    ),
    T3 = list(
        tfmode = c(G09 = 0.7, G10 = 1.1, G11 = 0.1),
        likelihood = c(1, 1, 1)
    ),
    T4 = list(
        tfmode = c(G06 = 1, G07 = 0.5, G08 = 1, G05 = 1.9, G10 = -1.5, G11 = -2, G09 = 3.1, G03 = -1.2),
        likelihood = c(1, 1, 1, 1, 1, 1, 1, 1)
    )
)
t(viper::viper(eset=t(mat), regulon=gs, minsize=1, eset.filter=F, pleiotropy=F))
pargs=list(regulators = 0.05, shadow = 0.05, targets = 1, penalty = 20, method = "adaptive")
t(viper::viper(eset=t(mat), regulon=gs, minsize=1, eset.filter=F, pleiotropy=T, pleiotropyArgs=pargs))

"""


def test_get_tmp_idxs(
    rng,
):
    pval = rng.random((5, 5))
    np.fill_diagonal(pval, np.nan)
    dc.mt._viper._get_tmp_idxs.py_func(pval)


def test_func_viper(
    adata,
    net,
):
    dict_net = {
        'T1': 'T1',
        'T2': 'T1',
        'T3': 'T2',
        'T4': 'T2',
        'T5': 'T3',
    }
    net['source'] = [dict_net[s] for s in net['source']]
    net = pd.concat([
        net,
        net[net['source'] == 'T2'].assign(source='T4'),
        pd.DataFrame([['T4', 'G03', -1.2]], columns=['source', 'target', 'weight'], index=[0])
    ])
    mat = dc.pp.extract(data=adata)
    X, obs, var = mat
    sources, targets, adjmat = dc.pp.adjmat(features=var, net=net, verbose=False)
    obs = np.array(['S01', 'S02', 'S29', 'S30'])
    X = np.vstack((X[:2, :], X[-2:, :]))
    pf_dc_es, pf_dc_pv = dc.mt._viper._func_viper(mat=X, adj=adjmat, pleiotropy=False)
    pt_dc_es, pt_dc_pv = dc.mt._viper._func_viper(mat=X, adj=adjmat, n_targets=1, pleiotropy=True)
    pf_vp_es = np.array([
        [ 3.708381, -2.154396, -1.4069603, -2.468185],
        [ 3.702911, -2.288070, -0.7239077, -2.848132],
        [-3.613066,  1.696114, -0.5789716,  2.039502],
        [-3.495480,  2.560792, -1.1296442,  2.523946],
    ])
    pt_vp_es = np.array([
        [ 2.224856, -2.154396, -1.4069603, -1.131059],
        [ 1.880012, -2.288070, -0.7239077, -2.848132],
        [-3.177418,  1.696114, -0.5789716,  2.039502],
        [-2.073186,  2.560792, -1.1296442,  2.523946],
    ])
    assert np.isclose(pf_vp_es, pf_dc_es).all()
    assert np.isclose(pt_vp_es, pt_dc_es).all()
