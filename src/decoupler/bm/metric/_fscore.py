import numpy as np

from decoupler.bm._pp import _validate_bool
from decoupler.bm.metric._Metric import Metric


def f_fscore(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float]:
    """F-beta score"""
    # Validate
    _validate_bool(y_true=y_true, y_score=y_score)
    assert y_score.dtype == np.bool_, "y_score must be bool numpy.ndarray"
    y_true = y_true.astype(np.bool_)
    # Compute
    tp: float = np.sum(y_true * y_score)
    fp: float = np.sum((~y_true) * y_score)
    fn: float = np.sum(y_true * (~y_score))
    if tp > 0:
        prc = tp / (tp + fp)
        rcl = tp / (tp + fn)
    else:
        prc = 0.0
        rcl = 0.0
    return prc, rcl


fscore = Metric(func=f_fscore, scores=["precision", "recall"])
