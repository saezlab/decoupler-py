import unittest
from pandas._testing import assert_frame_equal

import pandas as pd
import numpy as np

from anndata import AnnData


class TestPre(unittest.TestCase):
    exp_net = pd.DataFrame(
        [
        ['T1', 'G1', 1.0],
        ['T1', 'G2', 0.5],
        ['T1', 'G3',-1.0],
        ['T2', 'G3', 0.2],
        ['T2', 'G4', 0.1],
        ['T2', 'G5',-0.5]
        ],
        columns = ['source', 'target', 'weight']
    )
    
    net = pd.DataFrame(
        [
        ['G1', 'T1', 1.0],
        ['G2', 'T1', 0.5],
        ['G3', 'T1',-1.0],
        ['G3', 'T2', 0.2],
        ['G4', 'T2', 0.1],
        ['G5', 'T2',-0.5]
        ],
        columns = ['gene', 'tf', 'mor']
    )
    
    mat = np.array([
        [0,1,2,3,4],
        [5,6,7,8,9]
    ])
    g = ['G1','G2','G3','G4','G5']
    
    def test_extract(self):
        """
        Test mat extraction with different inputs.
        """
        
        from decoupler import extract
        
        mat = self.mat
        g = self.g
        
        # List
        pred_mat, pred_g = extract([mat, g])
        expt_mat, expt_g = [mat, g]
        pred_mat = list(pred_mat.A.flatten())
        expt_mat = list(expt_mat.flatten())
        self.assertEqual(pred_mat, expt_mat)
        self.assertEqual(list(pred_g), list(expt_g))
        
        # Data Frame
        df = pd.DataFrame(mat, columns=g)
        
        pred_mat, pred_g = extract(df)
        expt_mat, expt_g = [mat, g]
        pred_mat = list(pred_mat.A.flatten())
        expt_mat = list(expt_mat.flatten())
        self.assertEqual(pred_mat, expt_mat)
        self.assertEqual(list(pred_g), list(expt_g))
        
        # AnnData
        adata = AnnData(mat, var=pd.DataFrame(index=g))
        
        pred_mat, pred_g = extract(adata)
        expt_mat, expt_g = [mat, g]
        pred_mat = list(pred_mat.A.flatten())
        expt_mat = list(expt_mat.flatten())
        
        self.assertEqual(pred_mat, expt_mat)
        self.assertEqual(list(pred_g), list(expt_g))
    
    def test_rename_net(self):
        """
        Test net rename to default names.
        """
        
        from decoupler.pre import rename_net
        
        net = self.net.copy()
        exp_net = self.exp_net
        
        # Rename
        net = rename_net(net, source='tf', target='gene', weight='mor')
        
        assert_frame_equal(net, exp_net)
        
    def test_get_net_mat(self):
        """
        Test expand net to adjacency matrix between sources and targets.
        """
        
        from decoupler.pre import get_net_mat
        
        net = self.exp_net
        
        # Extract
        sources, targets, X = get_net_mat(net)
        
        self.assertEqual(list(sources), list(net['source'].unique()))
        self.assertEqual(list(targets), list(net['target'].unique()))