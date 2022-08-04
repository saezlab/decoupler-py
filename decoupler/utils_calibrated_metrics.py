# taken directly from: https://github.com/wissam-sib/calibrated_metrics/blob/f1c262d0aaac16356a143596a832030abb87d48c/calibrated_metrics.py
# Siblini W., Fréry J., He-Guelton L., Oblé F., Wang YQ. (2020) Master Your Metrics with Calibration. In: Berthold M., Feelders A., Krempl G. (eds) Advances in Intelligent Data Analysis XVIII. IDA 2020. Lecture Notes in Computer Science, vol 12080. Springer, Cham

import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.metrics import confusion_matrix

def precision_recall_curve(y_true, y_pred, pos_label=None,
                           sample_weight=None,pi0=None):
    """Compute precision-recall (with optional calibration) pairs for different probability thresholds
    This implementation is a modification of scikit-learn "precision_recall_curve" function that adds calibration
    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    calib_precision : array, shape = [n_thresholds + 1]
        Calibrated Precision values such that element i is the calibrated precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """
    
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    
   
    
    
    if pi0 is not None:
        pi = np.sum(y_true)/float(np.array(y_true).shape[0])
        ratio = pi*(1-pi0)/(pi0*(1-pi))
        precision = tps / (tps + ratio*fps)
    else:
        precision = tps / (tps + fps)
    
    precision[np.isnan(precision)] = 0
        
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision(y_true, y_pred, pos_label=1, sample_weight=None,pi0=None):
        precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight, pi0=pi0)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])
    
    
def f1score(y_true, y_pred, pi0=None):
    """
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier. (must be binary)
    pi0 : float, None by default
        The reference ratio for calibration
    """
    CM = confusion_matrix(y_true, y_pred)
    
    tn = CM[0][0]
    fn = CM[1][0]
    tp = CM[1][1]
    fp = CM[0][1] 
        
    pos = fn + tp
    
    recall = tp / float(pos)
    
    if pi0 is not None:
        pi = pos/float(tn + fn + tp + fp)
        ratio = pi*(1-pi0)/(pi0*(1-pi))
        precision = tp / float(tp + ratio*fp)
    else:
        precision = tp / float(tp + fp)
    
    if np.isnan(precision):
        precision = 0
    
    if (precision+recall)==0.0:
        f=0.0
    else:
        f = (2*precision*recall)/(precision+recall)

    return f


def bestf1score(y_true, y_pred, pi0=None):
    """
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    pi0 : float, None by default
        The reference ratio for calibration
    """
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pi0=pi0)
    
    fscores = (2*precision*recall)/(precision+recall)
    fscores = np.nan_to_num(fscores,nan=0, posinf=0, neginf=0)
    
    return np.max(fscores)