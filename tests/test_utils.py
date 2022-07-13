import unittest
from numpy.testing import assert_allclose
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
import decoupler as dc


class TestUtils(unittest.TestCase):
    
    mat, net = dc.get_toy_data()

    def test_get_toy_data(self):

        dc.get_toy_data()

    def test_show_methods(self):

        methods = dc.show_methods()

        self.assertTrue(methods.shape[0] > 0)

    def tes_check_corr(self):

        corrs = dc.check_corr(self.net)
        n_reg = self.net['source'].unique()
        n_pairs = (n_reg * (n_reg - 1)) / 2

        self.assertTrue(n_pairs == corrs.shape[0])

    def test_get_acts(self):

        m, r, c = dc.extract(self.mat)
        var = pd.DataFrame(index=c)
        obs = pd.DataFrame(index=r)
        adata = AnnData(csr_matrix(m), var=var, obs=obs)
        dc.run_mlm(adata, self.net, min_n=0, use_raw=False)

        estimate = dc.get_acts(adata, 'mlm_estimate')
        pvals = dc.get_acts(adata, 'mlm_pvals')

        assert_allclose(estimate.X, adata.obsm['mlm_estimate'].values)
        assert_allclose(pvals.X, adata.obsm['mlm_pvals'].values)

    def test_melt(self):

        estimate, pvals = dc.run_mlm(self.mat, self.net, min_n=0)
        dc.melt(estimate)
        dc.melt(pvals)
        
    def test_denserun(self):
        dc.dense_run(dc.run_consensus, self.mat, self.net, min_n=0)
        
        
