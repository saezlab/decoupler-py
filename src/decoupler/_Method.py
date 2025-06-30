from collections.abc import Callable

import pandas as pd

from decoupler._datatype import DataType
from decoupler.mt._run import _run


class MethodMeta:
    def __init__(
        self,
        name: str,
        desc: str,
        func: Callable,
        stype: str,
        adj: bool,
        weight: bool,
        test: bool,
        limits: tuple,
        reference: str,
    ):
        self.name = name
        self.desc = desc
        self.func = func
        self.stype = stype
        self.adj = adj
        self.weight = weight
        self.test = test
        self.limits = limits
        self.reference = reference

    def meta(self) -> pd.DataFrame:
        meta = pd.DataFrame(
            [
                {
                    "name": self.name,
                    "desc": self.desc,
                    "stype": self.stype,
                    "weight": self.weight,
                    "test": self.test,
                    "limits": self.limits,
                    "reference": self.reference,
                }
            ]
        )
        return meta


# @docs.dedent
class Method(MethodMeta):
    def __init__(
        self,
        _method: MethodMeta,
    ):
        super().__init__(
            name=_method.name,
            desc=_method.desc,
            func=_method.func,
            stype=_method.stype,
            adj=_method.adj,
            weight=_method.weight,
            test=_method.test,
            limits=_method.limits,
            reference=_method.reference,
        )
        self._method = _method
        self.__doc__ = self.func.__doc__

    def __call__(
        self,
        data: DataType,
        net: pd.DataFrame,
        tmin: int | float = 5,
        raw: bool = False,
        empty: bool = True,
        bsize: int | float = 250_000,
        verbose: bool = False,
        **kwargs,
    ):
        return _run(
            name=self.name,
            func=self.func,
            adj=self.adj,
            test=self.test,
            data=data,
            net=net,
            tmin=tmin,
            raw=raw,
            empty=empty,
            bsize=bsize,
            verbose=verbose,
            **kwargs,
        )


def _show_methods(methods):
    return pd.concat([method.meta() for method in methods]).reset_index(drop=True)
