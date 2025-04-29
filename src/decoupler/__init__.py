from importlib.metadata import version

from . import ds, mt, op, pl, pp

__all__ = ['ds', 'mt', 'op', 'pl', 'pp']

__version__ = version('decoupler')
