import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics._ranking import _binary_clf_curve
from ..metrics import (
    check_m_inputs, binary_clf_curve, mc_perm, metric_auroc, metric_auprc, metric_mcauroc, metric_mcauprc, metric_rank,
    metric_nrank, metric_recall
)


def test_check_m_inputs():
    act = np.array([7, 6, 5, 5, 4, 3, 2, 1, 0])
    grt = np.array([1, 0, -1, 0, 1, 1, 0, 0, 0])
    with pytest.raises(ValueError):
        check_m_inputs(y_true=grt, y_score=act)

    grt = np.array([-1, 0, -1, 0, -1, -1, 0, 0, 0])
    with pytest.raises(ValueError):
        check_m_inputs(y_true=grt, y_score=act)

    act = np.array([7, 6, 5, 5, 4, 3, 2, 1])
    grt = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0])
    with pytest.raises(AssertionError):
        check_m_inputs(y_true=grt, y_score=act)


def test_binary_clf_curve():
    act = np.array([7, 6, 5, 5, 4, 3, 2, 1, 0], dtype=np.float32)
    grt = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0], dtype=np.float32)

    a = _binary_clf_curve(y_true=grt, y_score=act)
    b = binary_clf_curve(y_true=grt, y_score=act)
    assert np.allclose(a, b)


def test_mc_perm():
    act = np.array([7, 6, 5, 5, 4, 3, 2, 1, 0], dtype=np.float32)
    grt = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0], dtype=np.float32)

    y_true, y_score = mc_perm(y_true=grt, y_score=act, n_iter=100, seed=42)
    assert np.all(y_score[:, 0] == 7.)
    assert np.all(y_score[:, 1] == 4.)
    assert np.all(y_score[:, 2] == 3.)
    _, counts = np.unique(y_score[:, 3:], axis=0, return_counts=True)
    assert np.all(counts < 15)


def test_metric_auroc():
    act = np.array([7, 6, 5, 5, 4, 3, 2, 1, 0])
    grt = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0])

    a = roc_auc_score(y_true=grt, y_score=act)
    b = metric_auroc(y_true=grt, y_score=act)
    assert np.isclose(a, b)

    act = np.array([7, -6, 5, 5, -4, 3, 2, 1, 0])
    grt = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0])
    a = roc_auc_score(y_true=grt, y_score=act)
    b = metric_auroc(y_true=grt, y_score=act)
    assert np.isclose(a, b)


def test_metric_auprc():
    act = np.array([7, 6, 5, 5, 4, 3, 2, 1, 0])
    grt = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0])

    a = average_precision_score(y_true=grt, y_score=act)
    b = metric_auprc(y_true=grt, y_score=act)
    assert np.isclose(a, b)
    b = metric_auprc(y_true=grt, y_score=act, pi0=0.5)
    assert np.isclose(0.7460317, b)

    act = np.array([7, -6, 5, 5, -4, 3, 2, 1, 0])
    grt = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0])
    a = average_precision_score(y_true=grt, y_score=act)
    b = metric_auprc(y_true=grt, y_score=act)
    assert np.isclose(a, b)
    b = metric_auprc(y_true=grt, y_score=act, pi0=0.5)
    assert np.isclose(0.7373737, b)


def test_metric_mcauroc():
    act = np.array([7, 6, 5, 5, 4, 3, 2, 1, 0])
    grt = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0])

    a = metric_auroc(y_true=grt, y_score=act)
    b = metric_mcauroc(y_true=grt, y_score=act)
    assert np.isclose(a, np.mean(b), rtol=1e-01)


def test_metric_mcauprc():
    act = np.array([7, 6, 5, 5, 4, 3, 2, 1, 0])
    grt = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0])

    a = metric_auprc(y_true=grt, y_score=act, pi0=0.5)
    b = metric_mcauprc(y_true=grt, y_score=act)
    assert np.isclose(a, np.mean(b), rtol=1e-01)


def test_metric_rank():
    act = np.array([[7, 6, 5], [5, 4, 3], [2, 1, 0]])
    grt = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])

    a = metric_rank(y_true=grt, y_score=act)
    assert a.size == 4
    assert np.all(a == np.array([1., 2., 3., 3.]))


def test_metric_nrank():
    act = np.array([[7, 6, 5], [5, 4, 3], [2, 1, 0]])
    grt = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])

    a = metric_nrank(y_true=grt, y_score=act)
    assert a.size == 4
    assert np.all(a == np.array([0., 0.5, 1., 1.]))


def test_metric_recall():
    act = np.array([7, 0, 5, 0, 0, 3, 0, 1, 0])
    grt = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0])
    a = metric_recall(y_true=grt, y_score=act)
    assert isinstance(a, float)
