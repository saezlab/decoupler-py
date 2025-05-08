from importlib.metadata import version

from . import bm, ds, mt, op, pl, pp, tl

__all__ = ['bm', 'ds', 'mt', 'op', 'pl', 'pp', 'tl']

__version__ = version('decoupler')
