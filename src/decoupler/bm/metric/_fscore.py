from typing import Tuple

import numpy as np

from decoupler.bm._pp import _validate_bool


def fscore(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float = 1.,
) -> Tuple[float, float, float]:
    """
    F-beta score
    """
    # Validate
    _validate_bool(y_true=y_true, y_score=y_score)
    assert y_score.dtype == np.bool_, \
    'y_score must be bool numpy.ndarray'
    assert isinstance(beta, (int, float)) and 0. <= beta <= 1., \
    'beta must be numeric and between 0 and 1'
    y_true = y_true.astype(np.bool_)
    # Compute
    tp = np.sum(y_true * y_score)
    fp = np.sum((~y_true) * y_score)
    fn = np.sum(y_true * (~y_score))
    if tp > 0:
        prc = tp / (tp + fp)
        rcl = tp / (tp + fn)
        score = (1 + beta**2) * (prc * rcl) / ((prc * beta**2) + rcl)
    else:
        prc = 0.
        rcl = 0.
        score = 0.
    return prc, rcl, score

fscore.scores = ['precision', 'recall', 'fscore']
