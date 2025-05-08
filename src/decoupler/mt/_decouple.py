import pandas as pd

from decoupler._docs import docs
from decoupler._datatype import DataType
from decoupler.mt._methods import _methods


@docs.dedent
def decouple(
    data: DataType,
    net: pd.DataFrame,
    methods: str | list = 'all',
    args: dict = dict(),
    **kwargs
) -> dict | None:
    """
    Runs multiple enrichment methods sequentially.

    Parameters
    ----------
    %(data)s
    %(net)s
    methods
        List of methods to run.
    args
        Dictionary of dictionaries containing method-specific keyword arguments.
    %(tmin)s
    %(raw)s
    %(empty)s
    %(bsize)s
    %(verbose)s
    """
    # Validate
    _mdict = {m.name: m for m in _methods}
    if isinstance(methods, str):
        if methods == 'all':
            methods = _mdict.keys()
        else:
            methods = [methods]
    assert set(methods).issubset(_mdict), \
    f'methods={methods} must be in decoupler.\nUse decoupler.mt.show_methods to check which ones are available'
    assert all(k in methods for k in args), \
    f'All keys in args={args.keys()} must belong to a method in methods={methods}'
    # Run each method
    all_res = {}
    for name in methods:
        mth = _mdict[name]
        arg = args.setdefault(name, {})
        res = mth(data=data, net=net, **arg, **kwargs)
        if res:
            res = {
                f'score_{mth.name}': res[0],
                f'padj_{mth.name}': res[1],
            }
            all_res = all_res | res
    if all_res:
        return all_res
