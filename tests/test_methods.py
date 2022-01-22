import unittest
import decoupler as dc


class TestPre(unittest.TestCase):
    
    mat, net = dc.get_toy_data()
    
    def test_aucell(self):
        dc.run_aucell(self.mat, self.net, min_n=0, n_up=2)
    
    def test_gsea(self):
        dc.run_gsea(self.mat, self.net, min_n=0)
        
    def test_gsva(self):
        dc.run_gsva(self.mat, self.net, min_n=0)
        
    def test_mdt(self):
        dc.run_mdt(self.mat, self.net, min_n=0)
        
    def test_mlm(self):
        dc.run_mlm(self.mat, self.net, min_n=0)
    
    def test_ora(self):
        dc.run_ora(self.mat, self.net, min_n=0)
        
    def test_udt(self):
        dc.run_udt(self.mat, self.net, min_n=0)
        
    def test_ulm(self):
        dc.run_ulm(self.mat, self.net, min_n=0)
        
    def test_viper(self):
        dc.run_viper(self.mat, self.net, min_n=0)
        
    def test_wmean(self):
        dc.run_wmean(self.mat, self.net, min_n=0)
        
    def test_wsum(self):
        dc.run_wsum(self.mat, self.net, min_n=0)
        
    def test_decouple(self):
        dc.decouple(self.mat, self.net, min_n=0, 
                    verbose=False)
