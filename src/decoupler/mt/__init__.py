from decoupler._Method import _show_methods
from decoupler.mt._methods import aucell
from decoupler.mt._methods import gsea
from decoupler.mt._methods import gsva
from decoupler.mt._methods import mdt
from decoupler.mt._methods import mlm
from decoupler.mt._methods import ora
from decoupler.mt._methods import udt
from decoupler.mt._methods import ulm
from decoupler.mt._methods import viper
from decoupler.mt._methods import waggr
from decoupler.mt._methods import zscore
from decoupler.mt._methods import _methods
from decoupler.mt._decouple import decouple
from decoupler.mt._consensus import consensus

def show() -> None:
    """Displays the methods available in decoupler"""
    return _show_methods(_methods)
