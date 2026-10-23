from decoupler._Method import _show_methods
from decoupler.mt._consensus import consensus
from decoupler.mt._decouple import decouple
from decoupler.mt._methods import _methods, aucell, gsea, gsva, mdt, mlm, ora, udt, ulm, viper, waggr, zscore
from decoupler.mt._query_set import query_set


def show() -> None:
    """Displays the methods available in decoupler"""
    return _show_methods(_methods)
