from typing import Tuple

import numpy as np

from decoupler.bm._pp import _validate_bool


def _binary_clf_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sort scores
    idx = np.flip(np.argsort(y_score))
    y_score = y_score[idx]
    y_true = y_true[idx]
    # Find unique value idxs
    idx = np.where(np.diff(y_score))[0]
    # Append a value for the end of the curve
    idx = np.append(idx, y_true.size - 1)
    # Acucmulate TP with decreasing threshold
    tps = np.cumsum(y_true)[idx]
    fps = 1 + idx - tps
    return fps, tps, y_score[idx]


def auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    _validate_bool(y_true=y_true, y_score=y_score)
    # Compute binary curve
    fps, tps, thr = _binary_clf_curve(y_true, y_score)
    # Add limits
    fps = np.append(0., fps)
    tps = np.append(0., tps)
    thr = np.append(thr[0] + 1., thr)
    # Compute ratios
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    # Compute auc
    dx = np.diff(np.ascontiguousarray(fpr))
    # Get direction slope
    if np.all(dx <= 0):
        d = -1.
    else:
        d = 1.
    # Compute area
    ret = np.sum((dx * (tpr[1:] + tpr[:-1]) / 2.0))
    auc = d * ret
    return auc


def auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pi0: float = 0.5
) -> float:
    _validate_bool(y_true=y_true, y_score=y_score)
    assert isinstance(pi0, (int, float)) and 0. <= pi0 <= 1., \
    'pi0 must be numeric and between 0 and 1'
    # Compute binary curve
    fps, tps, thr = _binary_clf_curve(y_true, y_score)
    # Compute prc
    ps = tps + fps
    msk = ps != 0
    # Siblini W., Fréry J., He-Guelton L., Oblé F., Wang YQ. (2020) Master
    # Your Metrics with Calibration. In: Berthold M., Feelders A., Krempl G.
    # (eds) Advances in Intelligent Data Analysis XVIII. IDA 2020. Lecture
    # Notes in Computer Science, vol 12080. Springer, Cham
    pi = np.sum(y_true) / y_true.size
    ratio = pi * (1 - pi0) / (pi0 * (1 - pi))
    prc = tps[msk] / (tps[msk] + ratio * fps[msk])
    # Compute rcl
    rcl = tps / tps[-1]
    # Flip and add limits
    prc = np.append(np.flip(prc), 1)
    rcl = np.append(np.flip(rcl), 0)
    thr = np.flip(thr)
    dx = np.diff(np.ascontiguousarray(rcl))
    auc = -np.sum(dx * prc[:-1])
    return auc


def auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pi0: float = 0.5,
) -> Tuple[float, float]:
    """
    Area Under the Curve.
    """
    # Normalize to make comparable
    norm = np.nanmax(np.abs(y_score), axis=1)
    msk = norm == 0.
    norm[msk] = 1.
    y_score = y_score / norm.reshape(-1, 1)
    assert ((-1. <= y_score) & (y_score <= 1.)).all()
    # Flatten and remove nans
    y_true, y_score = y_true.ravel(), y_score.ravel()
    msk_nan = ~np.isnan(y_score)
    y_true, y_score = y_true[msk_nan], y_score[msk_nan]
    auc_roc = auroc(y_true=y_true, y_score=y_score)
    auc_prc = auprc(y_true=y_true, y_score=y_score, pi0=pi0)
    return auc_roc, auc_prc

auc.scores = ['auroc', 'auprc']
