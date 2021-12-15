import unittest
from pandas._testing import assert_frame_equal

import pandas as pd
import numpy as np

from anndata import AnnData

# Define toy data
from numpy.random import default_rng

net = pd.DataFrame(
    [
    ['M1', 'G2', 1],
    ['M1', 'G3', 1],
    ['M1', 'G4', 1],
        
    ['M2', 'G5', 1],
    ['M2', 'G6', 1],
    ['M2', 'G7', 1],
        
    ['T1', 'G1', 1], 
    ['T1', 'G2', 1], 
    ['T1', 'G3', 1],
        
    ['T2', 'G2', 1], 
    ['T2', 'G3', 1], 
    ['T2', 'G4', 1], 
    ['T2', 'G5', -1],
    ['T2', 'G6', -1], 
        
    ['T3', 'G3', -1], 
    ['T3', 'G4', -1], 
    ['T3', 'G5', 1], 
    ['T3', 'G6', 1],
    ['T3', 'G7', 1], 
        
    ['T4', 'G6', 1], 
    ['T4', 'G7', 1], 
    ['T4', 'G8', 1],
    ],
    columns = ['source', 'target', 'weight']
)

rng = default_rng(seed=42)
mat = np.array([
    np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
    np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
    np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
    np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
    np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
    np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
    np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
    np.array([0,0,0,0,1,1,1,1]) * np.abs(rng.normal(size=8)),
    np.array([0,0,0,0,1,1,1,1]) * np.abs(rng.normal(size=8)),
    np.array([0,0,0,0,1,1,1,1]) * np.abs(rng.normal(size=8)),
    np.array([0,0,0,0,1,1,1,1]) * np.abs(rng.normal(size=8)),
    np.array([0,0,0,0,1,1,1,1]) * np.abs(rng.normal(size=8))
])
mat = pd.DataFrame(mat, columns=['G1','G2','G3','G4','G5','G6','G7','G8'])


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
    
    rng = default_rng(seed=42)
    mat = np.array([
        np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
        np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
        np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
        np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
        np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
        np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
        np.array([1,1,1,1,0,0,0,0]) * np.abs(rng.normal(size=8)),
        np.array([0,0,0,0,1,1,1,1]) * np.abs(rng.normal(size=8)),
        np.array([0,0,0,0,1,1,1,1]) * np.abs(rng.normal(size=8)),
        np.array([0,0,0,0,1,1,1,1]) * np.abs(rng.normal(size=8)),
        np.array([0,0,0,0,1,1,1,1]) * np.abs(rng.normal(size=8)),
        np.array([0,0,0,0,1,1,1,1]) * np.abs(rng.normal(size=8))
    ], dtype=np.float32)
    r = ['S{0}'.format(i+1) for i in range(mat.shape[0])]
    c = ['G1','G2','G3','G4','G5','G6','G7','G8']
    
    def test_extract(self):
        """
        Test mat extraction with different inputs.
        """
        
        from decoupler import extract
        
        mat = self.mat
        r = self.r
        c = self.c
        
        # List
        pred_mat, pred_r, pred_c = extract([mat, r, c])
        expt_mat, expt_r, expt_c = [mat, r, c]
        pred_mat = list(pred_mat.A.flatten())
        expt_mat = list(expt_mat.flatten())
        self.assertEqual(pred_mat, expt_mat)
        self.assertEqual(list(pred_r), list(expt_r))
        self.assertEqual(list(pred_c), list(expt_c))
        
        # Data Frame
        df = pd.DataFrame(mat, index=r, columns=c)
        
        pred_mat, pred_r, pred_c = extract(df)
        expt_mat, expt_r, expt_c = [mat, r, c]
        pred_mat = list(pred_mat.A.flatten())
        expt_mat = list(expt_mat.flatten())
        self.assertEqual(pred_mat, expt_mat)
        self.assertEqual(list(pred_r), list(expt_r))
        self.assertEqual(list(pred_c), list(expt_c))
        
        # AnnData
        adata = AnnData(mat, obs=pd.DataFrame(index=r), var=pd.DataFrame(index=c))
        
        pred_mat, pred_r, pred_c = extract(adata)
        expt_mat, expt_r, expt_c = [mat, r, c]
        pred_mat = list(pred_mat.A.flatten())
        expt_mat = list(expt_mat.flatten())
        self.assertEqual(pred_mat, expt_mat)
        self.assertEqual(list(pred_r), list(expt_r))
        self.assertEqual(list(pred_c), list(expt_c))
    
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
