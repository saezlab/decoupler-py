__version__ = '1.3.5'  # noqa: F401
__version_info__ = tuple([int(num) for num in __version__.split('.')])  # noqa: F401

from .pre import extract, match, rename_net, get_net_mat, filt_min_n, mask_features  # noqa: F401
from .utils import melt, show_methods, check_corr, get_toy_data, summarize_acts, assign_groups  # noqa: F401
from .utils import dense_run, p_adjust_fdr, shuffle_net, read_gmt  # noqa: F401
from .utils_anndata import get_acts, swap_layer, get_pseudobulk, get_contrast  # noqa: F401
from .utils_anndata import get_top_targets, format_contrast_results  # noqa: F401
from .method_wmean import run_wmean  # noqa: F401
from .method_wsum import run_wsum  # noqa: F401
from .method_ulm import run_ulm  # noqa: F401
from .method_mdt import run_mdt  # noqa: F401
from .method_mlm import run_mlm  # noqa: F401
from .method_udt import run_udt  # noqa: F401
from .method_ora import run_ora, test1r, get_ora_df  # noqa: F401
from .method_gsva import run_gsva  # noqa: F401
from .method_gsea import run_gsea  # noqa: F401
from .method_viper import run_viper  # noqa: F401
from .method_aucell import run_aucell  # noqa: F401
from .decouple import decouple, run_consensus  # noqa: F401
from .consensus import cons  # noqa: F401
from .omnip import show_resources, get_resource, get_progeny, get_dorothea, translate_net  # noqa: F401
from .plotting import plot_volcano, plot_violins, plot_barplot  # noqa: F401
from .plotting import plot_metrics_scatter, plot_metrics_boxplot, plot_metrics_scatter_cols  # noqa: F401
from .benchmark import benchmark, format_benchmark_inputs, get_performances  # noqa: F401
from .utils_benchmark import get_toy_benchmark_data, show_metrics  # noqa: F401
from .metrics import metric_auroc, metric_auprc, metric_mcauroc, metric_mcauprc  # noqa: F401
