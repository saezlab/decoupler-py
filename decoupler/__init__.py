__version__ = '1.1.0'  # noqa: F401
__version_info__ = tuple([int(num) for num in __version__.split('.')])  # noqa: F401

from .pre import extract, match, rename_net, get_net_mat, filt_min_n  # noqa: F401
from .utils import melt, show_methods, check_corr, get_acts, get_toy_data, summarize_acts, assign_groups  # noqa: F401
from .utils import get_top_targets, format_contrast_results  # noqa: F401
from .utils_anndata import get_pseudobulk, get_contrast  # noqa: F401
from .method_wmean import run_wmean  # noqa: F401
from .method_wsum import run_wsum  # noqa: F401
from .method_ulm import run_ulm  # noqa: F401
from .method_mlm import run_mlm  # noqa: F401
from .method_ora import run_ora  # noqa: F401
from .method_gsva import run_gsva  # noqa: F401
from .method_gsea import run_gsea  # noqa: F401
from .method_viper import run_viper  # noqa: F401
from .method_aucell import run_aucell  # noqa: F401
from .decouple import decouple  # noqa: F401
from .consensus import run_consensus  # noqa: F401
from .omnip import show_resources, get_resource, get_progeny, get_dorothea  # noqa: F401
from .plotting import plot_volcano  # noqa: F401

# External libraries go out of main setup
try:
    from .method_mdt import run_mdt  # noqa: F401
except Exception:
    pass

try:
    from .method_udt import run_udt  # noqa: F401
except Exception:
    pass
