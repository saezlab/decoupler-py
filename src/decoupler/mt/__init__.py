from decoupler._Method import _show_methods
from decoupler.mt._methods import aucell
from decoupler.mt._methods import gsea  # p-vals are weird compared to old version
from decoupler.mt._methods import gsva  # division by zero sometimes
from decoupler.mt._methods import mdt
from decoupler.mt._methods import mlm  # singular matrix
from decoupler.mt._methods import ora  # Check that p-values are actually two sided and not one sided
from decoupler.mt._methods import udt
from decoupler.mt._methods import ulm
from decoupler.mt._methods import viper # TypeError: expected a sequence of integers or a single integer, got '146.0'
from decoupler.mt._methods import waggr
from decoupler.mt._methods import zscore
from decoupler.mt._methods import _methods
from decoupler.mt._decouple import decouple
from decoupler.mt._consensus import consensus

def show() -> None:
    """Displays the methods available in decoupler"""
    return _show_methods(_methods)
