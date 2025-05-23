from typing import Tuple, Callable
import inspect
from functools import partial

import numpy as np
import scipy.stats as sts
import numba as nb

from decoupler._docs import docs
from decoupler._log import _log
from decoupler._Method import MethodMeta, Method
from decoupler.mt._gsea import _std, _ridx


@nb.njit(cache=True)
def _wsum(
    x: np.ndarray,
    w: np.ndarray,
) -> float:
    return np.sum(x * w)


@nb.njit(cache=True)
def _wmean(
    x: np.ndarray,
    w: np.ndarray,
) -> float:
    agg = _wsum(x, w)
    div = np.sum(np.abs(w))
    return agg / div


def _fun(
    f: Callable,
    verbose: bool = False,
):
    @nb.njit(parallel=True, cache=True)
    def _f(mat, adj):
        nobs, nvar = mat.shape
        nvar, nsrc = adj.shape
        es = np.zeros((nobs, nsrc))
        for i in nb.prange(nobs):
            x = mat[i]
            for j in range(nsrc):
                w = adj[:, j]
                es[i, j] = f(x, w)
        return es
    _f.__name__ = f.__name__
    if _f.__name__ not in _cfuncs:
        _cfuncs[f.__name__] = _f
        m = f'waggr - using {_f.__name__} for the first time, will need to be compiled'
        _log(m, level='info', verbose=verbose)


_fun_dict = {
    'wsum': _wsum,
    'wmean': _wmean,
}

_cfuncs = dict()

def _validate_args(
    fun: Callable,
    verbose: bool,
) -> bool:
    args = inspect.signature(fun).parameters
    required_args = ['x', 'w']
    for arg in required_args:
        if arg not in args:
            assert False, f'fun={fun.__name__} must contain arguments x and w'
    # Check if any additional arguments have default values
    for param in args.values():
        if param.name not in required_args and param.default == inspect.Parameter.empty:
            assert False, f'fun={fun.__name__} has an argument {param.name} without a default value'
    if not hasattr(fun, 'func_code'):
        m = f'waggr - {fun.__name__} will be compiled into numba code'
        _log(m, level='info', verbose=verbose)
        fun = nb.njit(fun)
    return fun


def _validate_func(
    fun: str | Callable,
    verbose: bool,
) -> None:
    fun = _validate_args(fun=fun, verbose=verbose)
    x = np.array([1., 2., 3.])
    w = np.array([-1., 0., 2.])
    try:
        res = fun(x=x, w=w)
        assert isinstance(res, (int, float)), 'output of fun must be a single numerical value'
    except:
        raise ValueError(f'fun failed to run with test data: fun(x={x}), w={w}')
    m = f'waggr - using function {fun.__name__}'
    _log(m, level='info', verbose=verbose)
    _fun(f=fun, verbose=verbose)


@nb.njit(parallel=True, cache=True)
def _perm(
    fun: Callable,
    es: np.ndarray,
    mat: np.ndarray,
    adj: np.ndarray,
    idx: np.ndarray,
):
    # Init
    nobs, nvar = mat.shape
    nvar, nsrc = adj.shape
    times, nvar = idx.shape
    null_dst = np.zeros((nobs, nsrc, times))
    pvals = np.zeros((nobs, nsrc))
    # Permute
    for i in nb.prange(times):
        null_dst[:, :, i] = fun(mat[:, idx[i]], adj)
        pvals += np.abs(null_dst[:, :, i]) > np.abs(es)
    # Compute z-score
    nes = np.zeros(es.shape)
    for i in nb.prange(nobs):
        for j in range(nsrc):
            e = es[i, j]
            d = _std(null_dst[i, j, :], 1)
            if d != 0.:
                n = (e - np.mean(null_dst[i, j, :])) / d
            else:
                if e != 0.:
                    n = np.inf
                else:
                    n = 0.
            nes[i, j] = n
    # Compute empirical p-value
    pvals = np.where(pvals == 0.0, 1.0, pvals)
    pvals = np.where(pvals == times, times - 1, pvals)
    pvals = pvals / times
    pvals = np.where(pvals >= 0.5, 1 - (pvals), pvals)
    pvals = pvals * 2
    return nes, pvals


@docs.dedent
def _func_waggr(
    mat: np.ndarray,
    adj: np.ndarray,
    fun: str | Callable = 'wmean',
    times: int | float = 1000,
    seed: int | float = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Weighted Aggregate (WAGGR) :cite:`decoupler`.

    This approach aggregates the molecular features :math:`x_i` from one observation :math:`i` with
    the feature weights :math:`w` of a given feature set :math:`j` into an enrichment score :math:`ES`.

    This method can use any aggregation function, which by default is the weighted mean.

    .. math::

        ES = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}

    Another simpler option is the weighted sum.

    .. math::

        ES = \sum_{i=1}^{n} w_i x_i

    Alternatively, this method can also take any defined function :math:`f` as long at it aggregates :math:`x_i` and
    :math:`w` into a single :math:`ES`.

    .. math::

        ES = f(w_i, x_i)

    This functionality makes it relatively easy to implement and try new enrichment methods.

    When multiple random permutations are done (``times > 1``), statistical significance is assessed via empirical testing.

    .. math::

        p_{value}=\frac{ES_{rand} \geq ES}{P}

    Where:

    - :math:`ES_{rand}` are the enrichment scores of the random permutations
    - :math:`P` is the total number of permutations

    Additionaly, :math:`ES` is updated to a normalized enrichment score :math:`NES`.

    .. math::

        NES = \frac{ES - \mu(ES_{rand})}{\sigma(ES_{rand})}

    Where:

    - :math:`\mu` is the mean
    - :math:`\sigma` is the standard deviation

    %(yestest)s

    %(params)s
    
    fun
        Function to compute enrichment statistic from omics readouts (``x``) and feature weights (``w``).
        Provided function must contain ``x`` and ``w`` arguments and ouput a single float.
        By default, 'wmean' and 'wsum' are implemented.
    %(times)s
    %(seed)s

    %(returns)s
    """
    assert isinstance(fun, str) or callable(fun), 'fun must be str or callable'
    if isinstance(fun, str):
        assert fun in _fun_dict, 'when fun is str, it must be wmean or wsum'
        fun = _fun_dict[fun]
    _validate_func(fun, verbose=verbose)
    vfun = _cfuncs[fun.__name__]
    assert isinstance(times, (int, float)) and times >= 0, 'times must be numeric and >= 0'
    assert isinstance(seed, (int, float)) and seed >= 0, 'seed must be numeric and >= 0'
    times, seed = int(times), int(seed)
    nobs, nvar = mat.shape
    nvar, nsrc = adj.shape
    m = f'waggr - calculating scores for {nsrc} sources across {nobs} observations'
    _log(m, level='info', verbose=verbose)
    es = vfun(mat, adj)
    if times > 1:
        m = f'waggr - comparing estimates against {times} random permutations'
        _log(m, level='info', verbose=verbose)
        idx = _ridx(times=times, nvar=nvar, seed=seed)
        es, pv = _perm(fun=vfun, es=es, mat=mat, adj=adj, idx=idx)
    else:
        pv = np.ones(es.shape)
    return es, pv


_waggr = MethodMeta(
    name='waggr',
    desc='Weighted Aggregate (WAGGR)',
    func=_func_waggr,
    stype='numerical',
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference='https://doi.org/10.1093/bioadv/vbac016',
)
waggr = Method(_method=_waggr)
