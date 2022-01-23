import unittest
from numpy.testing import assert_allclose
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
import decoupler as dc


class TestUtils(unittest.TestCase):
    
    def test_get_toy_data(self):
        
        self.mat, self.net = dc.get_toy_data()
    
    def test_show_methods(self):
        
        methods = dc.show_methods()
        
        self.assertTrue(methods.shape[0] > 0)
        
    def tes_check_corr(self):
        
        net = self.net
        
        corrs = dc.check_corr(net)
        n_reg = net['source'].unique()
        n_pairs = (n_reg * (n_reg - 1)) / 2
        
        self.assertTrue(n_pairs == corrs.shape[0])
        
    def test_get_acts(self):
        
        # Run act with AnnData
        mat, net = dc.get_toy_data()
        m, r, c = dc.extract(mat, net)
        var = pd.DataFrame(index=c)
        obs = pd.DataFrame(index=r)
        adata = AnnData(csr_matrix(m), var=var, obs=obs)
        dc.run_mlm(adata, net, min_n=0, use_raw=False)
        
        estimate = dc.get_acts(adata, 'mlm_estimate')
        pvals = dc.get_acts(adata, 'mlm_pvals')
        
        assert_allclose(estimate.X, adata.obsm['mlm_estimate'].values)
        assert_allclose(pvals.X, adata.obsm['mlm_pvals'].values)
        
    def test_melt(self):
        
        mat, net = dc.get_toy_data()
        
        estimate, pvals = dc.run_mlm(mat, net, min_n=0)