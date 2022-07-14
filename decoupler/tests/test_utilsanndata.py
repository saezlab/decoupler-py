import unittest
import scanpy as sc
from anndata import AnnData
import numpy as np
import pandas as pd
from ..utils import get_toy_data, format_contrast_results, get_top_targets
from ..utils_anndata import get_pseudobulk, get_contrast
from ..plotting import plot_volcano, plot_barplot, plot_violins


class TestAnnDataUtils(unittest.TestCase):

    n_samples = 64
    n = 8
    sample_col = 'sample_id'
    groups_col = 'cell_type'
    condition_col = 'condition'
    mat, net = get_toy_data(n_samples=n_samples)
    mat = np.floor(mat)

    pid = ['P{:02d}'.format(i+1) for i in range(int(n_samples/n))]
    cid = ['C{:02d}'.format(i+1) for i in range(int(n/4))]
    sid = np.array([np.repeat(['Healthy'], n_samples//2), np.repeat(['Disease'], n_samples//2)]).flatten()
    pid = np.repeat(pid, n)
    cid = np.tile(cid, n*4)
    obs = pd.DataFrame([pid, cid, sid],
                       index=[sample_col, groups_col, condition_col],
                       columns=mat.index).T
    adata = AnnData(mat, obs=obs)

    def test_pseudobulk(self):
        get_pseudobulk(self.adata, self.sample_col, self.groups_col,
                       min_prop=0.2, min_cells=1, min_counts=10, min_smpls=1)

    def test_contrast(self):
        pdata = get_pseudobulk(self.adata, self.sample_col, self.groups_col,
                               min_prop=0.2, min_cells=1, min_counts=10, min_smpls=1)
        sc.pp.normalize_total(pdata, target_sum=1e4)
        sc.pp.log1p(pdata)
        logFCs, pvals = get_contrast(pdata, self.groups_col, self.condition_col,
                                     'Disease', reference='Healthy', method='t-test')
        format_contrast_results(logFCs, pvals)
        get_top_targets(logFCs, pvals, 'C01', sign_thr=1, lFCs_thr=0.5)
        get_top_targets(logFCs, pvals, 'C01', name='T3', net=self.net, sign_thr=1, lFCs_thr=0.5)
        plot_volcano(logFCs, pvals, 'C01', top=5, sign_thr=0.05, lFCs_thr=1)
        plot_volcano(logFCs, pvals, 'C01', name='T3', net=self.net, top=5, sign_thr=0.05, lFCs_thr=1)
        plot_barplot(logFCs, 'C01', top=25)
        plot_violins(pdata)
