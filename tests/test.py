import unittest

import pandas as pd
import numpy as np


class TestPre(unittest.TestCase):
    net = pd.DataFrame(
        [
        ['T1', 'G1', 1],
        ['T1', 'G2', 1],
        ['T1', 'G3',-1],
        ['T2', 'G3', 1],
        ['T2', 'G4', 1],
        ['T2', 'G5',-1]
        ],
        columns = ['tf', 'target', 'weight']
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
