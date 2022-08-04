import pytest
import numpy as np
import pandas as pd
from ..consensus import z_score, mean_z_scores, cons


def test_z_score():
    arr = np.array([1., 2., 6.], dtype=np.float32)
    z_score(arr)

def test_mean_z_scores():
    arr = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [0., 1., 2.]]], dtype=np.float32)
    mean_z_scores(arr)

def test_cons():
    mlm_estimate = pd.DataFrame([[3.5, -0.5, 0.3], [3.6, -0.6, 0.04], [-1, 2, -1.8]],
                                columns=['T1', 'T2', 'T3'], index=['C1', 'C2', 'C3'])
    mlm_estimate.name = 'mlm_estimate'
    mlm_pvals = pd.DataFrame([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]],
                             columns=['T1', 'T2', 'T3'], index=['C1', 'C2', 'C3'])
    mlm_pvals.name = 'mlm_pvals'
    ulm_estimate = pd.DataFrame([[3.9, -0.2, 0.8], [3.2, -0.1, 0.09], [-2, 3, -2.3]],
                                columns=['T1', 'T2', 'T3'], index=['C1', 'C2', 'C3'])
    ulm_estimate.name = 'ulm_estimate'
    ulm_pvals = pd.DataFrame([[.2, .1, .3], [.5, .3, .2], [.4, .5, .3]],
                             columns=['T1', 'T2', 'T3'], index=['C1', 'C2', 'C3'])
    ulm_pvals.name = 'ulm_pvals'
    res = {mlm_estimate.name: mlm_estimate, mlm_pvals.name: mlm_pvals,
           ulm_estimate.name: ulm_estimate, ulm_pvals.name: ulm_pvals}
    cons(res)
