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
   
Utils:
------
.. autosummary::
   :toctree: generated

   show_methods
   check_corr
   melt
   get_acts
   get_toy_data
   summarize_acts
   assign_groups
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
   
Plotting
--------
.. autosummary::
   :toctree: generated
   
   plot_volcano