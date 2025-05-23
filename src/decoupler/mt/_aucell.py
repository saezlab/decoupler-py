from typing import Tuple

import numpy as np
import scipy.stats as sts
import scipy.sparse as sps
from tqdm.auto import tqdm
import numba as nb

from decoupler._docs import docs
from decoupler._log import _log
from decoupler._Method import MethodMeta, Method
from decoupler.pp.net import _getset


@nb.njit(parallel=True, cache=True)
def _auc(
    row: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_up: int,
    nsrc: int,
) -> np.ndarray:
    # Empty acts
    es = np.zeros(nsrc)
    # For each feature set
    for j in nb.prange(nsrc):
        # Extract feature set
        fset = _getset(cnct, starts, offsets, j)
        # Compute max AUC for fset
        x_th = np.arange(1, stop=fset.shape[0] + 1)
        x_th = x_th[x_th < n_up]
        max_auc = np.sum(np.diff(np.append(x_th, n_up)) * x_th)
        # Compute AUC
        x = row[fset]
        x = np.sort(x[x <= n_up])
        y = np.arange(x.shape[0]) + 1
        x = np.append(x, n_up)
        # Update acts matrix
        es[j] = np.sum(np.diff(x) * y) / max_auc
    return es


def _validate_n_up(
    nvar: int,
    n_up: int | float | None = None,
) -> int:
    assert isinstance(n_up, (int, float)) or n_up is None, 'n_up must be numerical or None'
    if n_up is None:
        n_up = np.ceil(0.05 * nvar)
        n_up = int(np.clip(n_up, a_min=2, a_max=nvar))
    else:
        n_up = int(np.ceil(n_up))
    assert nvar >= n_up > 1, f'For nvar={nvar}, n_up={n_up} must be between 1 and {nvar}'
    return n_up


@docs.dedent
def _func_aucell(
    mat: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_up: int | float | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, None]:
    r"""
    Area Under the Curve for set enrichment within single cells (AUCell) :cite:`aucell`.

    Given a ranked list of features per observation, AUCell calculates the AUC by measuring how early the features in
    the set appear in this ranking. Specifically, the enrichment score :math:`ES` is:

    .. math::
    
       {ES}_{i, F} = \int_0^1 {RecoveryCurve}_{i, F}(r_i) \, dr
    
    Where:
    
    - :math:`i` is the obervation  
    - :math:`F` is the feature set  
    - :math:`{RecoveryCurve}_{i, F}(r_i)` is the proportion of features from :math:`F` recovered in the top :math:`r_i`-fraction of the ranked list for observation :math:`i`

    %(notest)s

    %(params)s
    n_up
        Number of features to include in the AUC calculation.
        If ``None``, the top 5% of features based on their magnitude are selected.

    %(returns)s
    """
    nobs, nvar = mat.shape
    nsrc = starts.size
    n_up = _validate_n_up(nvar, n_up)
    m = f'aucell - calculating {nsrc} AUCs for {nvar} targets across {nobs} observations, categorizing features at rank={n_up}' 
    _log(m, level='info', verbose=verbose)
    es = np.zeros(shape=(nobs, nsrc))
    for i in tqdm(range(mat.shape[0]), disable=not verbose):
        if isinstance(mat, sps.csr_matrix):
            row = mat[i].toarray()[0]
        else:
            row = mat[i]
        row = sts.rankdata(a=-row, method='ordinal')
        es[i] = _auc(row=row, cnct=cnct, starts=starts, offsets=offsets, n_up=n_up, nsrc=nsrc)
    return es, None


_aucell = MethodMeta(
    name='aucell',
    desc='AUCell',
    func=_func_aucell,
    stype='categorical',
    adj=False,
    weight=False,
    test=False,
    limits=(0, 1),
    reference='https://doi.org/10.1038/nmeth.4463',
)
aucell = Method(_method=_aucell)
