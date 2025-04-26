from typing import Callable

import pandas as pd

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
    ):
        """
        Parameters
        ----------
        name
            Name of the Method.
        func
            Enrichment scoring function.
        stype
            Statistic type, either categorical or numerical.
        adj
            Wether the method transforms net to an adjacency matrix.
        weight
            Wether the method models feature weights.
        test
            Wether the method performs any test, returning a p-value.
        limits
            Range of values that the enrichment score can reach.
        reference
            Publication of method.
        """
        self.name = name
        self.func = func
        self.stype = stype
        self.adj = adj
        self.weight = weight
        self.test = test
        self.limits = limits
        self.reference = reference

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


class Method(MethodMeta):
    """
    Enrichment Method Class
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
        )
        self._method = _method

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
        """
        Run an enrichment method.

        Parameters
        ----------
        data
            AnnData instance, DataFrame or tuple of [matrix, samples, features].
        net
            Network in long format.
        tmin
            Minimum of number of targets per source after overlaping with ``mat``. If less, sources are removed.
        raw
            Whether to use the ``.raw`` attribute of ``AnnData``.
        empty
            Whether to remove empty observations (rows) or features (columns).
        bsize
            For large datasets in sparse format, this parameter controls how many observations are processed at once.
            Increasing this value speeds up computation but uses more memory.
        verbose
            If True, print progress messages or additional information during execution.
        """
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
