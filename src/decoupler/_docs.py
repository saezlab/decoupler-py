from docrep import DocstringProcessor

_features = """\
features
    Column names of ``mat``."""

_net = """\
net
    Network in long format. Must include ``source`` and ``target`` columns, and optionally a ``weight`` column."""

_tmin = """\
tmin
    Minimum number of targets per source. Sources with fewer targets will be removed."""

_verbose = """\
verbose
    Whether to display progress messages and additional execution details."""

_data = """\
data
    AnnData instance, DataFrame or tuple of [matrix, samples, features]."""

_layer = """\
layer
    Layer key name of an ``anndata.AnnData`` instance."""

_raw = """\
raw
    Whether to use the ``.raw`` attribute of ``anndata.AnnData``."""

_empty = """\
empty
    Whether to remove empty observations (rows) or features (columns)."""

_adata = """\
adata
    Annotated data matrix with observations (rows) and features (columns)."""

_times = """\
    Number of random permutations to do.
"""

_seed = """\
    Random seed to use.
"""


docs = DocstringProcessor(
    net=_net,
    tmin=_tmin,
    verbose=_verbose,
    features=_features,
    data=_data,
    layer=_layer,
    raw=_raw,
    empty=_empty,
    adata=_adata,
    times=_times,
    seed=_seed,
)

