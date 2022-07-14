import unittest
from ..method_aucell import run_aucell
from ..method_gsea import run_gsea
from ..method_gsva import run_gsva
from ..method_mdt import run_mdt
from ..method_mlm import run_mlm
from ..method_ora import run_ora
from ..method_udt import run_udt
from ..method_ulm import run_ulm
from ..method_viper import run_viper
from ..method_wmean import run_wmean
from ..method_wsum import run_wsum
from ..decouple import decouple, run_consensus
from anndata import AnnData
import decoupler as dc


class TestMethods(unittest.TestCase):

    mat, net = dc.get_toy_data()

    def test_aucell(self):
        run_aucell(self.mat, self.net, min_n=0, n_up=2)

    def test_gsea(self):
        run_gsea(self.mat, self.net, min_n=0)

    def test_gsva(self):
        run_gsva(self.mat, self.net, min_n=0)

    def test_mdt(self):
        run_mdt(self.mat, self.net, min_n=0)

    def test_mlm(self):
        run_mlm(self.mat, self.net, min_n=0)

    def test_ora(self):
        run_ora(self.mat, self.net, min_n=0)

    def test_udt(self):
        run_udt(self.mat, self.net, min_n=0)

    def test_ulm(self):
        run_ulm(self.mat, self.net, min_n=0)

    def test_viper(self):
        run_viper(self.mat, self.net, min_n=0)

    def test_wmean(self):
        run_wmean(self.mat, self.net, min_n=0)

    def test_wsum(self):
        run_wsum(self.mat, self.net, min_n=0)

    def test_decouple(self):
        decouple(self.mat, self.net, min_n=0, verbose=False, methods='all')
        decouple(AnnData(self.mat), self.net, min_n=0, verbose=False, use_raw=False, methods='all')

    def test_consensus(self):
        run_consensus(self.mat, self.net, min_n=0)
