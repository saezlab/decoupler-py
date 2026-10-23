import numpy as np
import scipy.stats as sts

from decoupler.bm._pp import _validate_bool
from decoupler.bm.metric._Metric import Metric


def f_qrank(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float]:
    """1 - quantile normalized rank"""
    _validate_bool(y_true=y_true, y_score=y_score)
    y_rank = sts.rankdata(y_score, axis=1, nan_policy="omit", method="average")
    y_rank = y_rank / np.sum(~np.isnan(y_rank), axis=1).reshape(-1, 1)
    msk = y_true.astype(np.bool_)
    score = y_rank[msk]
    rest = y_rank[~msk]
    _, pval = sts.ranksums(score, rest, alternative="greater")
    score = float(np.nanmean(score))
    pval = float(-np.log10(pval))
    return score, pval


qrank = Metric(func=f_qrank, scores=["1-qrank", "-log10(pval)"])
