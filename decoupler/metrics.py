"""
Metric functions to evaluate benchamrk performance.
Functions to compute metrics for evaluating methods and networks.
"""

import numpy as np
import numba as nb
from scipy.stats import rankdata


@nb.njit(nb.types.UniTuple(nb.f4[:], 3)(nb.f4[:], nb.f4[:]), cache=True)
def binary_clf_curve(y_true, y_score):

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


@nb.njit(nb.types.UniTuple(nb.f4[:], 3)(nb.f4[:], nb.f4[:]), cache=True)
def roc_curve(y_true, y_score):

    # Compute binary curve
    fps, tps, thr = binary_clf_curve(y_true, y_score)

    # Add limits
    fps = np.append(nb.float32(0), fps)
    tps = np.append(nb.float32(0), tps)
    thr = np.append(thr[0] + nb.float32(1), thr)

    # Compute ratios
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    return fpr, tpr, thr


@nb.njit(nb.types.UniTuple(nb.f4[:], 3)(nb.f4[:], nb.f4[:], nb.f4), cache=True)
def prc_curve(y_true, y_score, pi0):

    # Compute binary curve
    fps, tps, thr = binary_clf_curve(y_true, y_score)

    # Compute prc
    ps = tps + fps
    msk = ps != 0
    if pi0 > 0 and pi0 < 1:
        # Siblini W., Fréry J., He-Guelton L., Oblé F., Wang YQ. (2020) Master
        # Your Metrics with Calibration. In: Berthold M., Feelders A., Krempl G.
        # (eds) Advances in Intelligent Data Analysis XVIII. IDA 2020. Lecture
        # Notes in Computer Science, vol 12080. Springer, Cham
        pi = np.sum(y_true) / nb.float32(y_true.size)
        ratio = pi * (nb.float32(1) - pi0) / (pi0 * (nb.float32(1) - pi))
        prc = tps[msk] / (tps[msk] + ratio * fps[msk])
    else:
        prc = np.divide(tps[msk], ps[msk])

    # Compute rcl
    rcl = tps / tps[-1]

    # Flip and add limits
    prc = np.append(np.flip(prc), nb.float32(1))
    rcl = np.append(np.flip(rcl), nb.float32(0))
    thr = np.flip(thr)

    return prc, rcl, thr


@nb.njit(nb.f4(nb.f4[:], nb.f4[:]), cache=True)
def auc(x, y):

    # Compute diff
    dx = np.diff(np.ascontiguousarray(x))

    # Get direction slope
    if np.all(dx <= 0):
        d = nb.float32(-1)
    else:
        d = nb.float32(1)

    # Compute area
    ret = np.sum((dx * (y[1:] + y[:-1]) / nb.float32(2.0)))
    area = d * ret

    return area


@nb.njit(nb.f4(nb.f4[:], nb.f4[:]), cache=True)
def roc_auc(y_true, y_score):

    # Compute roc curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Compute auc
    return auc(fpr, tpr)


@nb.njit(nb.f4(nb.f4[:], nb.f4[:], nb.f4), cache=True)
def prc_auc(y_true, y_score, pi0):

    # Compute prc curve
    prc, rcl, _ = prc_curve(y_true, y_score, pi0)

    # Compute auc
    dx = np.diff(np.ascontiguousarray(rcl))
    return -np.sum(dx * prc[:-1])


def check_m_inputs(y_true, y_score):
    unq = np.sort(np.unique(y_true))
    lbl = np.array([0, 1])
    if unq.size > 2:
        raise ValueError("""Ground truth vector contains more than one
        class. Only binary classes are supported.""")
    else:
        if not np.all(unq == lbl):
            raise ValueError("""Ground truth binary classes must be 0 and 1.""")
    assert y_true.size == y_score.size, 'y_true and y_score must have the same length.'


def metric_rank(y_true, y_score):
    """
    Rank (from 1 to N)
    """

    check_m_inputs(y_true, y_score)

    return rankdata(-y_score, axis=1, nan_policy='omit')[y_true.astype(bool)]


def metric_nrank(y_true, y_score):
    """
    Min-max normalized rank (from 0 to 1)
    """

    check_m_inputs(y_true, y_score)

    rnks = rankdata(-y_score, axis=1, nan_policy='omit')
    mins = np.nanmin(rnks, axis=1)
    maxs = np.nanmax(rnks, axis=1)
    nrnks = (rnks - mins.reshape(-1, 1)) / (maxs - mins).reshape(-1, 1)
    return nrnks[y_true.astype(bool)]


def metric_auroc(y_true, y_score):
    """
    Area Under the Receiver Operating characteristic Curve (AUROC)
    """

    # Flatten
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_score = np.asarray(y_score, dtype=np.float32).flatten()

    # Check inputs
    check_m_inputs(y_true, y_score)

    return roc_auc(y_true, y_score)


def metric_auprc(y_true, y_score, pi0=None):
    """
    Area Under the Precision-Recall Curve (AUPRC)
    """

    # Flatten
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_score = np.asarray(y_score, dtype=np.float32).flatten()

    # Check inputs
    check_m_inputs(y_true, y_score)

    if pi0 is None:
        pi0 = 0.0

    return prc_auc(y_true, y_score, pi0)


@nb.njit(nb.types.Tuple((nb.f4[:], nb.f4[:, :]))(nb.f4[:], nb.f4[:], nb.i8, nb.i8), cache=True)
def mc_perm(y_true, y_score, n_iter, seed):

    # Separate TPs from TNs
    msk = y_true != 0

    # Split TP and TN
    tp_score = y_score[msk]
    tn_score = y_score[~msk]
    tp_true = y_true[msk]
    tn_true = y_true[~msk]
    n_tp = tp_score.size

    # Init y_true
    y_true = np.append(tp_true, tn_true[:n_tp])

    # Set seed
    np.random.seed(seed)

    # Generate random shuffling matrix
    idx = np.arange(tn_score.size, dtype=nb.i8)
    scores = np.zeros((n_iter, n_tp * 2), dtype=nb.f4)
    for i in range(n_iter):
        r_i = np.random.choice(idx, n_tp)
        scores[i] = np.append(tp_score, tn_score[r_i])

    return y_true, scores


@nb.njit(nb.f4[:](nb.f4[:], nb.f4[:, :]), parallel=True, cache=True)
def mcauroc(y_true, y_score):
    n_iter = y_score.shape[0]
    total = np.zeros(n_iter, dtype=nb.f4)
    for i in nb.prange(n_iter):
        total[i] = roc_auc(y_true, y_score[i])
    return total


@nb.njit(nb.f4[:](nb.f4[:], nb.f4[:, :]), parallel=True, cache=True)
def mcauprc(y_true, y_score):
    n_iter = y_score.shape[0]
    total = np.zeros(n_iter, dtype=nb.f4)
    for i in nb.prange(n_iter):
        total[i] = prc_auc(y_true, y_score[i], nb.f4(0.0))
    return total


def metric_mcauroc(y_true, y_score, n_iter=1000, seed=42):
    """
    Monte-Carlo Area Under the Receiver Operating characteristic Curve (AUROC)
    """

    # Flatten
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_score = np.asarray(y_score, dtype=np.float32).flatten()

    # Check inputs
    check_m_inputs(y_true, y_score)

    # Perform MC permutations
    y_true, y_score = mc_perm(y_true, y_score, n_iter, seed)

    # Compute AUC per permutation
    return mcauroc(y_true, y_score)


def metric_mcauprc(y_true, y_score, n_iter=1000, seed=42):
    """
    Monte-Carlo Area Under the Precision-Recall Curve (AUPRC)
    """

    # Flatten
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_score = np.asarray(y_score, dtype=np.float32).flatten()

    # Check inputs
    check_m_inputs(y_true, y_score)

    # Perform MC permutations
    y_true, y_score = mc_perm(y_true, y_score, n_iter, seed)

    # Compute AUC per permutation
    return mcauprc(y_true, y_score)
