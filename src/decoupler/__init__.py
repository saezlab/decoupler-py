from importlib.metadata import version

from . import ds, mt, op, pl, pp, tl

__all__ = ['ds', 'mt', 'op', 'pl', 'pp', 'tl']

__version__ = version('decoupler')
