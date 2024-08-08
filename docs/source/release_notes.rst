Release notes
=============

1.8.0
-----
Bug fixes
~~~~~~~~~
- Fixed ``scipy >= 1.14.0`` errors due to deprecated ``.A`` attribute.
- ``get_pseudobulk`` now handles categorical columns.

Changes
~~~~~~~
- ``pre`` functions now are more efficient.
- Refactored ``run_gsva``:
   -``kcdf`` density argument now accepts either ``'gaussian'``, ``'poisson'`` or ``None``.
   -``mx_diff`` and ``abs_rnk`` now can be set to different values.
   -Deprecated ``tau`` parameter

Additions
~~~~~~~~~
- Added method ``zscore``.
- Added ``pre`` function ``break_ties``.

1.7.0
-----

Bug fixes
~~~~~~~~~
- Fixed error in in ``get_contrast`` by reverting use of ``copy.deepcopy`` to ``copy``.
- Fixed verbose error regarding the number of unique sources being used in ``benchmark``.
- Added check for minimum version of ``igraph>=0.10.0`` to properly render ``plot_network``.
- Fixed return error of methods triggered when an observation was empty and input was ``AnnData``.

Changes
~~~~~~~
- Resource functions such as ``get_resource`` or ``get_collectri`` now accept different ``genesymbol_resource`` than UniProt for gene translation to other organisms.
- Deprecated ``sklearn`` and switched to ``sklearn`` for ``udt``.

Additions
~~~~~~~~~
- Added metric ``recall`` to the benchmark pipeline.

1.6.0
-----

Bug fixes
~~~~~~~~~
- Fixed wrong corrected p-values of ``get_metadata_associations`` whenever more than 10 factors were present.
- Added static fallback for the most used ``omnipath`` resources, enabling their retrieval when internet connection is unstable.
- Fixed ZeroDivisionError of ``gsea`` when many 0s were present.
- ``dense_run`` now returns sorted features, which allows to run the benchmark pipeline ignoring zeros.

Changes
~~~~~~~
- Methods now throw an error if there are repeated feature names in ``mat``.
- Functions can now accept ``AnnData`` objects with any sparse format than just ``scipy.sparse.csr_matrix``.
- ``get_pseudobulk`` now can return empty observations and features with the parameter ``remove_empty=False``.
- Fixed ``pandas`` warnings for categorical data across several functions.
- ``plot_volcano`` and ``plot_volcano_df`` now can use different colors for positive, negative and null genes.
- ``benchmark`` now has increased verbosity.
- ``get_metadata_associations`` now has a ``verbose`` argument.
- ``plot_associations`` now uses the ``marsilea`` plotting library instead of ``PyComplexHeatmap``.

Additions
~~~~~~~~~
- Added ``plot_network`` to visualize the network structure of top sources and their targets.
- Added metrics ``rank`` and ``nrank`` to the benchmark pipeline.
- Added ``dense`` argument to ``decouple`` to run methods ignoring zeroes per observation.

1.5.0
-----

Bug fixes
~~~~~~~~~
- ``run_mlm`` now uses the correct number of degrees of freedom to compute the t-value (one less than before), scores might change slightly.
- ``rank_sources_groups`` now handles ``nan`` p-value results and skips whenever the group is equal to the reference.
- ``get_pseudobulk`` now handles any sparse matrix format, can use individual ``lambda`` functions as ``mode`` and handles negative values when filtering for ``min_counts`` and ``min_cells``.
- Plotting functions with a grid now show it below dots instead of above.

Changes
~~~~~~~
- All functions that deal with ``mat`` now do not enforce ``csr_matrix`` format automatically.
- The benchmark pipeline now handles methods that provide only positive enrichment scores. Also added support for ``nan`` scores.
- Modified ``get_ora_df`` to make it easier to use results of differential expression analysis, now returns different statistics.
- ``gsea`` normalized score now is computed as in the original publication ``(ES+/mean(null+) and ES-/mean(null-))`` instead of ``(ES - mean(null)) / sd(null)``.
- ``run_ulm`` now runs faster.
- ``get_collectri`` now returns the PMIDs supporting each interaction when available.

Additions
~~~~~~~~~
- Added ``get_gsea_df`` function to run ``gsea`` directly on results of differential expression analysis, returns different statistics and the leading edge.
- Added ``plot_dotplot`` function to plot results of ``get_ora_df``.
- Added ``plot_barplot_df`` function to plot results of ``get_ora_df``.
- Added ``plot_running_score`` function to plot results of ``get_gsea_df``.
- Added ``plot_associations`` function to check for significant covariates in the metadata.

1.4.0
-----

Changes
~~~~~~~
- ``get_pseudobulk`` changes:
   - Default values now do not filter features. For feature filtering check the new functions ``filter_by_expr`` or ``filter_by_prop``.
   - If feature filters are used, it may return more genes than before due to a change of ``> min_props`` to ``>= min_props``.
   - Now it returns quality control metrics such as ``psbulk_n_cells``, ``psbulk_counts`` and ``psbulk_props``.
   - Now ``groups_col`` accepts take multiple keys.
   - Now ``mode`` accepts a dictionary of callable functions. The resulting profiles will be stored in ``.layers``.
- ``swap_layer`` now has a new argument ``X_layer_key``, a ``.layers`` key where to move and store the original ``.X``.
- Pseudobulk and bulk vignettes have been updated to use the PyDESeq2 package
- ``run_consensus`` now accepts extra arguments with the new parametter ``args`` that are passed down to ``decouple``.
- Omnipath functions now return resources with sorted indexes and throw a warning if the version is too old.
- ``run_wsum`` and ``run_wmean`` now correctly accept empty null distributions.

Additions
~~~~~~~~~
- Added ``filter_by_expr`` feature filtering function from edgeR.
- Added ``filter_by_prop`` feature filtering function. In previous versions it was incorporated inside ``get_pseudobulk``.
- Added ``plot_psbulk_samples`` to assess the quality of pseudobulk samples.
- Added ``plot_filter_by_expr`` to assess which filtering thresholds to use in ``filter_by_expr``.
- Added ``plot_filter_by_prop`` to assess which filtering thresholds to use in ``filter_by_prop``.
- Added ``plot_volcano_df`` to plot volcano plots from long format dataframes.
- Added ``plot_targets`` to plot downstream target genes of a source by their change and weight.
- Added ``get_collectri`` to retrieve the CollecTRI gene regulatory network.
- Added ``get_ksn_omnipath`` to retrieve the Kinase-Substrate network from omnipath.
- Added ``rank_sources_groups`` to identify marker sources (TFs, pathways, etc.) per group of samples/cells.

1.3.4
-----

Changes
~~~~~~~
- ``get_pseudobulk`` now has new arguments: ``mode`` to change how to summarize profiles and ``skip_checks`` to bypass checks.
- OmniPath functions now accept more organism synonyms.

Bug fixes
~~~~~~~~~
- Fixed empty text labels bug for ``adjustText==0.8``.


1.3.3
-----

Bug fixes
~~~~~~~~~
- ``read_gmt`` is now properly exported.

1.3.2
-----

Bug fixes
~~~~~~~~~
- ``plot_metrics_scatter_cols`` now deals with missing sources when comparing nets.

Changes
~~~~~~~
- ``get_pseudobulk`` and ``get_acts`` now have a ``dtype`` argument due to future ``AnnData`` changes.
- ``plot_metrics_scatter`` and ``plot_metrics_boxplot`` now use ``GroupBy.mean(numeric_only=True)``.

Additions
~~~~~~~~~
- Added ``swap_layer`` function to easily move ``adata`` layers to ``.X``.
- Added ``read_gmt`` function to read GMT files containing gene sets.

1.3.1
-----

Changes
~~~~~~~
- Omnipath wrappers (``get_resource``, ``get_dorothea`` and ``get_progeny``) now accept any organism name.

1.3.0
-----

Bug fixes
~~~~~~~~~
- Fixed change in api from ``sklearn.tree``.
- Forced gene names in ``extract`` to be in ``unicode`` format.
- Changed integer format from ``int32`` to ``int64`` to accommodate larger datasets across methods.

Additions
~~~~~~~~~
- Added conversion utility function ``translate_net`` to translate nets across organisms.

1.2.0
-----

Bug fixes
~~~~~~~~~
- Removed ``python <3.10`` limitation.
- Forced ``np.float32`` type to output of ``get_contrast``.
- Made ``summarize_acts`` compatible with older versions of numpy.
- ``extract_psbulk_inputs`` now checks if mat and obs have matching indexes.
- ``plot_volcano`` now correctly can plot networks with different source names.

Changes
~~~~~~~
- ``extract`` now removes empty samples and features.
- ``run_consensus`` now follows the same format as other methods, old function is now called ``cons``.
- ``get_pseudobulk`` now checks if input are raw integer counts.
- ``plot_volcano`` now can plot without subsetting features by a network and can save plots to disk.
- ``plot_volcano`` now uses ``adjustText`` to better plot text labels.
- ``plot_volcano`` now can set logFCs and p-value limits for outliers.
- ``get_top_targets`` now can also work without subsetting features by a network and returns significant adjusted p-values.
- ``get_contrast`` now can also work without needing to group.
- ``udt`` and ``mdt`` now check if ``skranger`` and ``sklearn`` are installed, respectively.
- ``get_toy_data`` now contains more example TFs.
- ``get_top_targets`` now returns ``logFCs`` and ``pvals`` as column names instead of ``logFC`` and ``pval``.
- ``format_contrast_results`` now returns also the adjusted p-value.

Additions
~~~~~~~~~
- Added ``dense_run`` util function which runs methods ignoring zeros in the data.
- Added ``plot_violins`` and ``plot_barplot`` functions.
- Added ``p_adjust_fdr`` util function to correct p-values for FDR.
- Added ``get_ora_df`` function to infer ora from lists of genes instead of an input mat.
- Added ``shuffle_net`` function to randomize networks.
- Added benchmarking metrics ``metric_auroc``, ``metric_auprc``, ``metric_mcauroc`` and ``metric_mcauprc``.
- Added ``get_toy_benchmark_data`` function to generate a toy example for benchmarking.
- Added ``show_metrics`` function to show available metrics.
- Added  ``benchmark``, ``format_benchmark_inputs`` and ``get_performances`` functions to benchmark methods and nets.
- Added ``plot_metrics_scatter`` function to plot the results of running the benchmarking pipeline.
- Added ``plot_metrics_scatter_cols`` function to plot the results of running the benchmarking pipeline grouped by two levels.
- Added ``plot_metrics_scatter`` function to plot the results of running the benchmarking pipeline.
- Added ``plot_metrics_boxplot`` function to plot the distributions of Monte-Carlo benchmarking metrics.

1.1.0
-----
Bug fixes
~~~~~~~~~
- Fixed ``get_pseudobulk`` errors.
- Fixed ``get_progeny`` to correctly return non duplicate entries.
- Fixed ``run_viper`` parallelization error.
- Fixed ``run_ora`` to correctly deal with random ties.

Changes
~~~~~~~
- ``get_dorothea`` now returns an ordered dataframe.
- ``get_contrast`` now prints warnings instead of returning an empty dataframe.

Additions
~~~~~~~~~
- Added ``get_top_targets`` util function.
- Added ``format_contrast_results`` util function.
