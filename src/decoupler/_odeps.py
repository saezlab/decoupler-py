import warnings
import types
from typing import TYPE_CHECKING


def _try_import(
    name: str
) -> types.ModuleType | None:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module=name)
            module = __import__(name, fromlist=[""])
        return module
    except ImportError:
        return None


def _check_import(
    module: types.ModuleType
) -> None:
    if module is None:
        name = module.__name__
        raise ImportError(
            f"{name} is not installed. Please install it using:\n"
            f"  pip install {name}"
            "or install decoupler with full dependencies:\n"
            "  pip install 'decoupler[full]'"
        )


# Handle optional dependencies
ig = _try_import("igraph")
if ig is not None:
    if TYPE_CHECKING:
        from igraph import Graph
    else:
        Graph = ig.Graph
else:
    if TYPE_CHECKING:
        from typing import Any as Graph
    else:
        Graph = None

xgboost = _try_import("xgboost")
dcor = _try_import("dcor")
