API
===
.. module:: decoupler
.. automodule:: decoupler
   :noindex:
   
Import decoupler as::

   import decoupler as dc

Preprocessing:
--------------
.. autosummary::
   :toctree: generated

   extract
   filt_min_n
   match
   rename_net
   get_net_mat
   

Methods:
--------
.. autosummary::
   :toctree: generated

   run_aucell
   run_gsea
   run_gsva
   run_mdt
   run_mlm
   run_ora
   run_udt
   run_ulm
   run_viper
   run_wmean
   run_wsum
   run_consensus

Running multiple methods:
-------------------------
.. autosummary::
   :toctree: generated

   decouple
   cons
   dense_run
   
General utils:
--------------
.. autosummary::
   :toctree: generated

   melt
   show_methods
   check_corr
   get_toy_data
   summarize_acts
   assign_groups
   dense_run
   p_adjust_fdr
   shuffle_net

AnnData utils:
--------------
.. autosummary::
   :toctree: generated
   
   get_acts
   get_pseudobulk
   get_contrast
   get_top_targets
   format_contrast_results
   
Omnipath wrappers:
------------------
.. autosummary::
   :toctree: generated
   
   show_resources
   get_resource
   get_progeny
   get_dorothea
   translate_net
   
Plotting
--------
.. autosummary::
   :toctree: generated
   
   plot_volcano
   plot_violins
   plot_barplot
   plot_metrics_scatter
   plot_metrics_scatter_cols
   plot_metrics_boxplot

Metrics
-------
.. autosummary::
   :toctree: generated
   
   metric_auroc
   metric_auprc
   metric_mcauroc
   metric_mcauprc

Benchmark utils
---------------
.. autosummary::
   :toctree: generated
   
   get_toy_benchmark_data
   show_metrics

Benchmark
---------
.. autosummary::
   :toctree: generated
   
   benchmark
   format_benchmark_inputs
   get_performances
