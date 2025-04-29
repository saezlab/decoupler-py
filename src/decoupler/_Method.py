from typing import Callable
import textwrap

import pandas as pd

from decoupler._docs import docs
from decoupler._datatype import DataType
from decoupler.mt._run import _run


class MethodMeta:
    """
    A Class used to store Method Metadata
    """
    def __init__(
        self,
        name: str,
        func: Callable,
        stype: str,
        adj: bool,
        weight: bool,
        test: bool,
        limits: tuple,
        reference: str,
        params: str,
    ):
        self.name = name
        self.func = func
        self.stype = stype
        self.adj = adj
        self.weight = weight
        self.test = test
        self.limits = limits
        self.reference = reference
        self.params = params

    def meta(self) -> pd.DataFrame:
        meta = pd.DataFrame([{
            'name': self.name,
            'stype': self.stype,
            'weight': self.weight,
            'test': self.test,
            'limits': self.limits,
            'reference': self.reference
        }])
        return meta

mparams = {
    'aucell': 'aucell',
    'gsea': 'gsea',
    'gsva': 'gsva',
    'mdt': 'mdt',
    'mlm': 'mlm',
    'ora': 'ora',
    'udt': 'udt',
    'ulm': 'ulm',
    'viper': 'viper',
    'waggr': 'waggr',
    'zscore': 'zscore'
}

@docs.dedent
class Method(MethodMeta):
    """
    Enrichment Method Class

    Parameters
    ----------
    %(data)s
    %(net)s
    %(tmin)s
    %(raw)s
    %(empty)s
    bsize
        For large datasets in sparse format, this parameter controls how many observations are processed at once.
        Increasing this value speeds up computation but uses more memory.
    %(verbose)s
    """
    def __init__(
        self,
        _method: MethodMeta,
    ):
        super().__init__(
            name=_method.name,
            func=_method.func,
            stype=_method.stype,
            adj=_method.adj,
            weight=_method.weight,
            test=_method.test,
            limits=_method.limits,
            reference=_method.reference,
            params=_method.params
        )
        self._method = _method
        self.__doc__ = f"{self.__doc__}\n\nMethod parameters\n-----------------\n{self.params}\n\n"


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

    def __repr__(self):
        doc = f"""
        Method
        ------
        Name: {self.name}
        Type of enrichment statistic: {self.stype}
        Models feature weights: {self.weight}
        Performs statistical test: {self.test}
        Range of values: {self.limits}
        Reference: {self.reference}

        """
        return doc
