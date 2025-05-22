from typing import Tuple

import numpy as np

from decoupler.bm._pp import _validate_bool


def fscore(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float, float]:
    """
    F-beta score
    """
    # Validate
    _validate_bool(y_true=y_true, y_score=y_score)
    assert y_score.dtype == np.bool_, \
    'y_score must be bool numpy.ndarray'
    y_true = y_true.astype(np.bool_)
    # Compute
    tp = np.sum(y_true * y_score)
    fp = np.sum((~y_true) * y_score)
    fn = np.sum(y_true * (~y_score))
    if tp > 0:
        prc = tp / (tp + fp)
        rcl = tp / (tp + fn)
    else:
        prc = 0.
        rcl = 0.
    return prc, rcl

fscore.scores = ['precision', 'recall']
