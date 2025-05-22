from typing import Tuple

import numpy as np
import scipy.stats as sts

from decoupler.bm._pp import _validate_bool


def qrank(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float]:
    """
    1 - quantile normalized rank
    """
    _validate_bool(y_true=y_true, y_score=y_score)
    y_rank = sts.rankdata(y_score, axis=1, nan_policy='omit', method='average')
    y_rank = y_rank / np.sum(~np.isnan(y_rank), axis=1).reshape(-1, 1)
    msk = y_true.astype(np.bool_)
    score = y_rank[msk]
    rest = y_rank[~msk]
    _, pval = sts.ranksums(score, rest, alternative='greater')
    score = np.nanmean(score)
    return score, -np.log10(pval)

qrank.scores = ['1-qrank', '-log10(pval)']
