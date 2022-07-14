import unittest
from numpy.testing import assert_allclose
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..utils import get_toy_data, show_methods, check_corr, get_acts, melt, dense_run
from ..pre import extract
from ..method_mlm import run_mlm
from ..decouple import run_consensus


class TestUtils(unittest.TestCase):

    mat, net = get_toy_data()

    def test_get_toy_data(self):

        get_toy_data()

    def test_show_methods(self):

        methods = show_methods()

        self.assertTrue(methods.shape[0] > 0)

    def tes_check_corr(self):

        corrs = check_corr(self.net)
        n_reg = self.net['source'].unique()
        n_pairs = (n_reg * (n_reg - 1)) / 2

        self.assertTrue(n_pairs == corrs.shape[0])

    def test_get_acts(self):

        m, r, c = extract(self.mat)
        var = pd.DataFrame(index=c)
        obs = pd.DataFrame(index=r)
        adata = AnnData(csr_matrix(m), var=var, obs=obs)
        run_mlm(adata, self.net, min_n=0, use_raw=False)

        estimate = get_acts(adata, 'mlm_estimate')
        pvals = get_acts(adata, 'mlm_pvals')

        assert_allclose(estimate.X, adata.obsm['mlm_estimate'].values)
        assert_allclose(pvals.X, adata.obsm['mlm_pvals'].values)

    def test_melt(self):

        estimate, pvals = run_mlm(self.mat, self.net, min_n=0)
        melt(estimate)
        melt(pvals)

    def test_denserun(self):
        dense_run(run_consensus, self.mat, self.net, min_n=0)
